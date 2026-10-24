"""Componente Qiskit: QAP -> QuadraticProgram -> QUBO -> Ising -> QAOA (simulador).

Qué hace Qiskit:
  * qiskit_optimization: construye el QuadraticProgram, lo convierte a QUBO
    (restricciones como penalización lambda*(sum x - 1)^2) y a Ising (SparsePauliOp).
  * qiskit.circuit.library.QAOAAnsatz: construye el circuito QAOA de p capas.
  * qiskit_aer EstimatorV2 (statevector, precisión 0): calcula
    <psi(gamma,beta)|H|psi(gamma,beta)> de forma exacta. Da el mismo valor que
    qiskit.primitives.StatevectorEstimator, pero ~5x más rápido con 16 qubits.
  * qiskit_aer SamplerV2: muestrea bitstrings del circuito optimizado.
Qué es clásico:
  * El optimizador de los ángulos (COBYLA de scipy), la decodificación de
    bitstrings, el chequeo de factibilidad y el cálculo del costo QAP.

Convención de variables: x_ij (ruta i -> estación j) es la variable/qubit i*N + j.
Qiskit ordena los bits de un bitstring en little-endian: el qubit q es bitstring[-1-q].
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
from qiskit import transpile
from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from scipy.optimize import minimize

from data import CONTROLS, load_qap, qap_cost, subinstance

RESULTS_DIR = Path(__file__).parent / "results"
SEED = 42
SHOTS = 2048
MAXITER = 200
RESTARTS = 3
LAMBDA_FACTORS = (0.25, 1.0, 3.0)
REPS = (1, 2)


# ---------------------------------------------------------------- modelo
def build_quadratic_program(F, D):
    """QAP como QuadraticProgram: min sum_{i<k, j!=l} F_ik D_jl x_ij x_kl."""
    N = len(F)
    qp = QuadraticProgram(f"qap_N{N}")
    names = [[f"x_{i}_{j}" for j in range(N)] for i in range(N)]
    for row in names:
        for name in row:
            qp.binary_var(name)
    quad = {}
    for i in range(N):
        for k in range(i + 1, N):
            if F[i][k] == 0:
                continue
            for j in range(N):
                for l in range(N):
                    if j != l and D[j][l]:
                        quad[(names[i][j], names[k][l])] = int(F[i][k] * D[j][l])
    qp.minimize(quadratic=quad)
    for i in range(N):
        qp.linear_constraint({names[i][j]: 1 for j in range(N)}, "==", 1, f"ruta_{i}")
    for j in range(N):
        qp.linear_constraint({names[i][j]: 1 for i in range(N)}, "==", 1, f"estacion_{j}")
    return qp


def penalty_bound(F, D):
    """Delta = máximo ahorro de costo al quitar una asignación x_ij = 1.

    Todos los coeficientes son >= 0, así que quitar la ruta i ahorra como mucho
    sum_k F_ik * max(D). Quitarla rompe una fila y una columna: penalización 2*lambda.
    Con lambda >= Delta la penalización siempre supera el ahorro (con margen x2).
    """
    return int(np.max(F.sum(axis=1)) * np.max(D))


def to_qubo_ising(qp, lam):
    """Devuelve (qubo, operador Ising, offset). E_qubo(x) = <z|op|z> + offset."""
    qubo = QuadraticProgramToQubo(penalty=lam).convert(qp)
    op, offset = qubo.to_ising()
    return qubo, op, offset


def qubo_arrays(qubo):
    n = qubo.get_num_vars()
    lin = qubo.objective.linear.to_array()
    quad = qubo.objective.quadratic.to_array(symmetric=False)
    return n, qubo.objective.constant, lin, quad


def exact_qubo_ground(qubo):
    """Mínimo exacto del QUBO enumerando los 2^n bitstrings (n <= 16)."""
    n, const, lin, quad = qubo_arrays(qubo)
    X = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1).astype(float)
    E = const + X @ lin + np.einsum("bi,ij,bj->b", X, quad, X)
    idx = int(E.argmin())
    return X[idx].astype(int), float(E[idx])


# ---------------------------------------------------------- decodificación
def bits_from_bitstring(bitstring):
    """Bitstring de Qiskit (little-endian) -> lista x[q]."""
    return [int(b) for b in reversed(bitstring)]


def decode(x, N):
    """Si x es una asignación válida devuelve la permutación, si no None."""
    X = np.asarray(x).reshape(N, N)
    if (X.sum(axis=0) == 1).all() and (X.sum(axis=1) == 1).all():
        return X.argmax(axis=1).tolist()
    return None


def evaluate_samples(counts, qubo, F, D):
    """Métricas de un conjunto de muestras. Separa energía QUBO y costo factible."""
    N = len(F)
    shots = sum(counts.values())
    n_feas, best_cost, best_perm = 0, None, None
    energies, opt_hits = [], 0
    for bitstring, c in counts.items():
        x = bits_from_bitstring(bitstring)
        e = qubo.objective.evaluate(x)
        energies.append((e, c))
        perm = decode(x, N)
        if perm is None:
            continue
        n_feas += c
        cost = qap_cost(perm, F, D)
        if cost == CONTROLS.get(N):
            opt_hits += c
        if best_cost is None or cost < best_cost:
            best_cost, best_perm = cost, perm
    e_vals = np.array([e for e, _ in energies])
    e_w = np.array([c for _, c in energies])
    return {
        "shots": shots,
        "feasible_fraction": n_feas / shots,
        "optimal_fraction": opt_hits / shots,
        "best_feasible_cost": best_cost,
        "best_feasible_perm": best_perm,
        "min_qubo_energy": float(e_vals.min()),
        "mean_qubo_energy": float((e_vals * e_w).sum() / shots),
    }


# ------------------------------------------------------ subproblemas (N=10)
def build_subproblem(F, D, perm, S):
    """QAP reducido: las rutas S se reparten sus propias estaciones T; el resto fijo.

    Variable x_a_b: ruta S[a] -> estación T[b]  (m^2 qubits).
    Costo total = const + sum_lineal + sum_cuadrático, con
      const      = pares (i<k) de rutas fijas,
      lineal a,b = sum_{k fija} F[S[a],k] * D[T[b], perm[k]],
      cuadr.     = F[S[a],S[a']] * D[T[b],T[b']]   (a<a', b!=b').
    """
    N, m = len(F), len(S)
    T = [perm[i] for i in S]
    fixed = [k for k in range(N) if k not in S]
    const = sum(int(F[i][k] * D[perm[i]][perm[k]])
                for a, i in enumerate(fixed) for k in fixed[a + 1:])
    qp = QuadraticProgram(f"sub_{'_'.join(map(str, S))}")
    names = [[f"x_{a}_{b}" for b in range(m)] for a in range(m)]
    for row in names:
        for name in row:
            qp.binary_var(name)
    linear = {names[a][b]: int(sum(F[S[a]][k] * D[T[b]][perm[k]] for k in fixed))
              for a in range(m) for b in range(m)}
    quad = {(names[a][b], names[a2][b2]): int(F[S[a]][S[a2]] * D[T[b]][T[b2]])
            for a in range(m) for a2 in range(a + 1, m)
            for b in range(m) for b2 in range(m) if b != b2 and F[S[a]][S[a2]]}
    qp.minimize(constant=const, linear=linear, quadratic=quad)
    for a in range(m):
        qp.linear_constraint({names[a][b]: 1 for b in range(m)}, "==", 1, f"ruta_{a}")
    for b in range(m):
        qp.linear_constraint({names[a][b]: 1 for a in range(m)}, "==", 1, f"estacion_{b}")
    return qp, T


def qp_penalty_bound(qp):
    """Delta = máximo ahorro al apagar una variable: lineal[v] + sum_u cuadr.[v,u].

    Todos los coeficientes son >= 0, así que es la misma regla que penalty_bound.
    """
    lin = qp.objective.linear.to_array()
    quad = qp.objective.quadratic.to_array(symmetric=False)
    return float(np.max(lin + quad.sum(axis=0) + quad.sum(axis=1)))


def repair(x, m):
    """Reparación clásica: permutación con más coincidencias con los bits de x.

    No usa el costo (para no esconder un solver exacto dentro de la reparación);
    los empates se resuelven por orden lexicográfico.
    """
    X = np.asarray(x).reshape(m, m)
    best = max(itertools.permutations(range(m)), key=lambda s: X[np.arange(m), s].sum())
    return list(best)


def solve_subproblem_qaoa(F, D, perm, S, reps=1, lam_factor=1.0, seed=SEED, shots=SHOTS,
                          mode="best"):
    """Resuelve el subproblema con QAOA y devuelve la nueva permutación completa.

    mode="best": postproceso clásico que elige la muestra factible de menor costo;
                 si no hay ninguna, repara la muestra más frecuente.
    mode="top":  usa solo la muestra más frecuente (salida "cruda" de QAOA);
                 si no es factible, la repara. Se informa antes/después.
    """
    m = len(S)
    qp, T = build_subproblem(F, D, perm, S)
    lam = lam_factor * qp_penalty_bound(qp)
    qubo, op, _ = to_qubo_ising(qp, lam)
    q = run_qaoa(op, reps, seed=seed, shots=shots)

    def full_perm(sigma):
        p = list(perm)
        for a, b in enumerate(sigma):
            p[S[a]] = T[b]
        return p

    n_feas, best = 0, None
    for bitstring, c in q["counts"].items():
        x = bits_from_bitstring(bitstring)
        sigma = decode(x, m)
        if sigma is None:
            continue
        n_feas += c
        cost = qap_cost(full_perm(sigma), F, D)
        assert np.isclose(cost, qp.objective.evaluate(x))
        if best is None or cost < best[0]:
            best = (cost, full_perm(sigma))

    top = max(q["counts"], key=q["counts"].get)
    x_top = bits_from_bitstring(top)
    sigma_top = decode(x_top, m)
    out = {"lambda": lam, "n_qubits": m * m, "feasible_fraction": n_feas / shots,
           "time_s": q["time_optimize_s"] + q["time_sample_s"],
           "top_probability": q["counts"][top] / shots,
           "top_feasible": sigma_top is not None,
           "top_qubo_energy": qubo.objective.evaluate(x_top),
           "best_feasible_cost": None if best is None else best[0]}
    if mode == "top" and sigma_top is not None:
        p = full_perm(sigma_top)
        out.update(perm=p, cost=qap_cost(p, F, D), repaired=False)
    elif mode == "top" or best is None:
        # Reparación: antes = muestra no factible (solo tiene energía QUBO).
        p = full_perm(repair(x_top, m))
        out.update(perm=p, cost=qap_cost(p, F, D), repaired=True,
                   before_repair="no factible, E_qubo=%g" % out["top_qubo_energy"])
    else:
        out.update(cost=best[0], perm=best[1], repaired=False)
    return out


# -------------------------------------------------------------------- QAOA
def run_qaoa(op, reps, seed=SEED, shots=SHOTS, maxiter=MAXITER, restarts=RESTARTS):
    """Optimiza los ángulos con el Estimator y muestrea con el Sampler."""
    scale = float(np.max(np.abs(op.coeffs.real)))
    h = (op / scale).simplify()
    ansatz = transpile(QAOAAnsatz(h, reps=reps), basis_gates=["h", "rx", "rz", "rzz", "cx"],
                       optimization_level=1, seed_transpiler=seed)
    estimator = AerEstimator(options={"default_precision": 0.0,
                                      "backend_options": {"method": "statevector",
                                                          "seed_simulator": seed}})
    rng = np.random.default_rng(seed)
    n_evals = 0

    def energy(theta):
        nonlocal n_evals
        n_evals += 1
        return float(estimator.run([(ansatz, h, theta)]).result()[0].data.evs)

    t0 = time.perf_counter()
    best = None
    for _ in range(restarts):
        # Orden de parámetros de QAOAAnsatz: beta[0..p-1], gamma[0..p-1].
        x0 = np.concatenate([rng.uniform(0, np.pi / 2, reps), rng.uniform(0, np.pi, reps)])
        res = minimize(energy, x0, method="COBYLA", options={"maxiter": maxiter})
        if best is None or res.fun < best.fun:
            best = res
    t_opt = time.perf_counter() - t0

    t1 = time.perf_counter()
    circ = ansatz.assign_parameters(best.x)
    circ.measure_all()
    sampler = AerSampler(default_shots=shots, seed=seed)
    counts = sampler.run([circ]).result()[0].data.meas.get_counts()
    t_sample = time.perf_counter() - t1

    return {
        "theta": best.x.tolist(),
        "expected_energy_normalized": float(best.fun),
        "hamiltonian_scale": scale,
        "n_energy_evals": n_evals,
        "circuit_depth": ansatz.depth(),
        "time_optimize_s": round(t_opt, 3),
        "time_sample_s": round(t_sample, 3),
        "counts": counts,
    }


def check_ising_consistency(qubo, op, offset, n_checks=20, seed=SEED):
    """Comprueba E_qubo(x) == <z|H_ising|z> + offset en bitstrings aleatorios."""
    rng = np.random.default_rng(seed)
    n = qubo.get_num_vars()
    for _ in range(n_checks):
        x = rng.integers(0, 2, n)
        z = 1 - 2 * x  # x = (1 - z)/2
        e_ising = offset
        for pauli, c in zip(op.paulis, op.coeffs.real):
            mask = pauli.z  # qubit q tiene Z
            e_ising += c * np.prod(z[mask])
        assert np.isclose(e_ising, qubo.objective.evaluate(x)), "Ising != QUBO"


# -------------------------------------------------------------- experimento
def run_grid(sizes=(3, 4), lambda_factors=LAMBDA_FACTORS, reps_list=REPS, seed=SEED):
    F_full, D_full, _ = load_qap()
    records = []
    for N in sizes:
        F, D = subinstance(F_full, D_full, N)
        qp = build_quadratic_program(F, D)
        delta = penalty_bound(F, D)
        print(f"\n=== N={N}: {N * N} qubits, Delta={delta}, óptimo={CONTROLS[N]} ===")
        for factor in lambda_factors:
            lam = factor * delta
            qubo, op, offset = to_qubo_ising(qp, lam)
            check_ising_consistency(qubo, op, offset)
            x_gs, e_gs = exact_qubo_ground(qubo)
            perm_gs = decode(x_gs, N)
            gs_cost = qap_cost(perm_gs, F, D) if perm_gs else None
            print(f"  lambda={lam:g} ({factor}*Delta): {len(op)} términos Pauli; "
                  f"mínimo exacto QUBO E={e_gs:g} -> "
                  f"{'factible, costo ' + str(gs_cost) if perm_gs else 'NO factible'}")
            for reps in reps_list:
                t0 = time.perf_counter()
                q = run_qaoa(op, reps, seed=seed)
                m = evaluate_samples(q["counts"], qubo, F, D)
                total = time.perf_counter() - t0
                rec = {
                    "N": N, "n_qubits": N * N, "lambda": lam, "lambda_factor": factor,
                    "delta": delta, "reps": reps, "seed": seed,
                    "n_pauli_terms": len(op), "ising_offset": float(offset),
                    "qubo_ground_energy": e_gs, "qubo_ground_feasible": perm_gs is not None,
                    "qubo_ground_cost": gs_cost, "time_total_s": round(total, 3),
                    **{k: v for k, v in q.items() if k != "counts"}, **m,
                }
                records.append(rec)
                print(f"    p={reps}: factibles={m['feasible_fraction']:.1%}  "
                      f"óptimas={m['optimal_fraction']:.1%}  "
                      f"mejor costo factible={m['best_feasible_cost']}  "
                      f"E_qubo min/media={m['min_qubo_energy']:g}/{m['mean_qubo_energy']:.1f}  "
                      f"tiempo={total:.1f}s ({q['n_energy_evals']} evals)")
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "qaoa_grid.json"
    out.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado en {out}")
    return records


if __name__ == "__main__":
    import sys
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    sys.stdout.reconfigure(encoding="utf-8")
    run_grid()
