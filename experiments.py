"""Experimentos y tabla comparativa clásico vs Qiskit.

Paso 4: para cada configuración QAOA (N, lambda, p) se compara contra
  * aleatorio_bits: el mismo número de bitstrings uniformes de N^2 bits
    (mismo espacio que QAOA, sin circuito) -> mide lo que aporta QAOA.
  * aleatorio_perm: el mismo número de permutaciones aleatorias (siempre
    factibles) -> búsqueda aleatoria clásica, sin la codificación QUBO.
  * fuerza bruta (classical.py) -> referencia exacta.
Todas las filas usan evaluate_samples, así la energía QUBO y el costo factible
se calculan igual. La energía QUBO depende de lambda; el costo factible no.
"""
from __future__ import annotations

import csv
import itertools
import json
import math
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

import classical
import qiskit_qaoa
from data import CONTROLS, OPTIMUM, is_permutation, load_qap, qap_cost, subinstance
from qiskit_qaoa import SEED, SHOTS, build_quadratic_program, evaluate_samples, to_qubo_ising

RESULTS_DIR = Path(__file__).parent / "results"


def gap(cost, N):
    """gap(%) respecto del óptimo de control; None si no hay solución factible."""
    return None if cost is None else round(100 * (cost - CONTROLS[N]) / CONTROLS[N], 2)


def random_bitstring_counts(n_bits, shots, rng):
    """Counts con el mismo formato que Qiskit (little-endian)."""
    X = rng.integers(0, 2, size=(shots, n_bits))
    return Counter("".join(map(str, row[::-1])) for row in X)


def random_permutation_counts(N, shots, rng):
    """Permutaciones aleatorias codificadas como bitstrings one-hot."""
    counts = Counter()
    for _ in range(shots):
        x = np.zeros(N * N, dtype=int)
        x[np.arange(N) * N + rng.permutation(N)] = 1
        counts["".join(map(str, x[::-1]))] += 1
    return counts


def load_or_run_qaoa(rerun=False):
    path = RESULTS_DIR / "qaoa_grid.json"
    if rerun or not path.exists():
        return qiskit_qaoa.run_grid()
    return json.loads(path.read_text(encoding="utf-8"))


def load_or_run_classical(rerun=False):
    path = RESULTS_DIR / "classical_bruteforce.json"
    if rerun or not path.exists():
        return classical.main()
    return json.loads(path.read_text(encoding="utf-8"))


def row(N, method, metrics, time_s, lam="", reps="", note=""):
    return {
        "N": N, "metodo": method, "lambda": lam, "p": reps,
        "muestras": metrics.get("shots", ""),
        "pct_factibles": round(100 * metrics["feasible_fraction"], 2),
        "pct_optimas": round(100 * metrics["optimal_fraction"], 2),
        "mejor_costo_factible": metrics["best_feasible_cost"],
        "gap_pct": gap(metrics["best_feasible_cost"], N),
        "E_qubo_min": metrics.get("min_qubo_energy", ""),
        "E_qubo_media": round(metrics["mean_qubo_energy"], 1) if "mean_qubo_energy" in metrics else "",
        "tiempo_s": round(time_s, 3),
        "nota": note,
    }


def step4_comparison(rerun=False, seed=SEED):
    F_full, D_full, _ = load_qap()
    qaoa = load_or_run_qaoa(rerun)
    brute = {r["N"]: r for r in load_or_run_classical(rerun)}
    rng = np.random.default_rng(seed)
    rows = []

    for N in sorted({r["N"] for r in qaoa}):
        F, D = subinstance(F_full, D_full, N)
        qp = build_quadratic_program(F, D)
        b = brute[N]
        rows.append(row(N, "clasico_fuerza_bruta",
                        {"shots": b["n_evaluated"], "feasible_fraction": 1.0,
                         "optimal_fraction": b["n_optimal_perms"] / b["n_evaluated"],
                         "best_feasible_cost": b["cost"]},
                        b["time_s"], note="exacto, enumera N! permutaciones"))

        # Probabilidades teóricas del muestreo uniforme de bitstrings.
        p_feas = math.factorial(N) / 2 ** (N * N)
        p_opt = brute[N]["n_optimal_perms"] / 2 ** (N * N)

        for lam in sorted({r["lambda"] for r in qaoa if r["N"] == N}):
            qubo, _, _ = to_qubo_ising(qp, lam)
            t0 = time.perf_counter()
            m_bits = evaluate_samples(random_bitstring_counts(N * N, SHOTS, rng), qubo, F, D)
            t_bits = time.perf_counter() - t0
            rows.append(row(N, "aleatorio_bits", m_bits, t_bits, lam=lam,
                            note=f"teórico: {100 * p_feas:.3g}% fact., {100 * p_opt:.3g}% ópt."))
            for r in (r for r in qaoa if r["N"] == N and r["lambda"] == lam):
                rows.append(row(N, "qiskit_qaoa", r, r["time_total_s"], lam=lam, reps=r["reps"],
                                note=f"{r['lambda_factor']}*Delta; mínimo QUBO "
                                     f"{'factible' if r['qubo_ground_feasible'] else 'NO factible'}"))

        qubo, _, _ = to_qubo_ising(qp, max(r["lambda"] for r in qaoa if r["N"] == N))
        t0 = time.perf_counter()
        m_perm = evaluate_samples(random_permutation_counts(N, SHOTS, rng), qubo, F, D)
        rows.append(row(N, "aleatorio_perm", m_perm, time.perf_counter() - t0,
                        note="búsqueda aleatoria clásica; E_qubo = costo (sin penalización)"))
    return rows


# ------------------------------------------------------------- paso 5: N=10
def hybrid_search(F, D, m, subsolver, seed, max_subproblems):
    """Descomposición en subproblemas de m rutas libres + pulido final con swaps.

    1. Permutación inicial aleatoria (semilla); no usa el óptimo conocido.
    2. Hasta max_subproblems subconjuntos S aleatorios: el subsolver propone una
       reasignación de S entre sus estaciones; se acepta si baja el costo total.
    3. Pulido clásico con swaps. Se informa el costo antes y después.
    """
    N = len(F)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N).tolist()
    c_init = qap_cost(perm, F, D)
    subsets = list(itertools.combinations(range(N), m))
    order = rng.permutation(len(subsets))[:max_subproblems]

    t0 = time.perf_counter()
    cost, n_acc, n_rep, n_rep_acc, feas, tops, log = c_init, 0, 0, 0, [], [], []
    for idx in order:
        S = list(subsets[idx])
        res = subsolver(F, D, perm, S)
        accepted = res["cost"] < cost
        if accepted:
            perm, cost, n_acc = res["perm"], res["cost"], n_acc + 1
        n_rep += res["repaired"]
        n_rep_acc += res["repaired"] and accepted
        if "feasible_fraction" in res:
            feas.append(res["feasible_fraction"])
        if "top_feasible" in res:
            tops.append(res["top_feasible"])
        log.append({"S": S, "cost_proposal": res["cost"], "accepted": accepted,
                    "repaired": res["repaired"], "before_repair": res.get("before_repair"),
                    "cost_after": cost, "feasible_fraction": res.get("feasible_fraction"),
                    "top_feasible": res.get("top_feasible"),
                    "best_feasible_cost": res.get("best_feasible_cost")})
    t_sub = time.perf_counter() - t0

    t1 = time.perf_counter()
    p_final, c_final, n_swaps = classical.swap_local_search(perm, F, D)
    t_polish = time.perf_counter() - t1
    assert is_permutation(p_final, N)

    return {
        "seed": seed, "m": m, "subproblems": len(order),
        "C_inicial": c_init, "C_tras_subproblemas": cost,
        "gap_tras_subproblemas": round(100 * (cost - OPTIMUM) / OPTIMUM, 2),
        "C_final_tras_swaps": c_final,
        "gap_final": round(100 * (c_final - OPTIMUM) / OPTIMUM, 2),
        "aceptados": n_acc, "reparados": n_rep, "reparados_aceptados": n_rep_acc,
        "swaps_pulido": n_swaps,
        "pct_factibles_medio": round(100 * float(np.mean(feas)), 2) if feas else None,
        "pct_top_factible": round(100 * float(np.mean(tops)), 1) if tops else None,
        "tiempo_subproblemas_s": round(t_sub, 2), "tiempo_pulido_s": round(t_polish, 3),
        "perm_final": p_final, "log": log,
    }


N10_COLS = ["metodo", "seed", "m", "subproblems", "C_inicial", "C_tras_subproblemas",
            "gap_tras_subproblemas", "C_final_tras_swaps", "gap_final", "aceptados",
            "reparados", "reparados_aceptados", "pct_factibles_medio", "pct_top_factible",
            "tiempo_subproblemas_s", "tiempo_pulido_s"]


def save_runs(runs, name, cols):
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"{name}.json").write_text(json.dumps(runs, indent=2, ensure_ascii=False),
                                             encoding="utf-8")
    return [{c: r.get(c, "") for c in cols} for r in runs]


def print_run(r):
    print(f"seed={r['seed']} {r['metodo']} m={r['m']}: {r['C_inicial']} -> subproblemas "
          f"{r['C_tras_subproblemas']} (gap {r['gap_tras_subproblemas']}%) -> "
          f"swaps {r['C_final_tras_swaps']} (gap {r['gap_final']}%)  "
          f"aceptados={r['aceptados']} reparados={r['reparados']} "
          f"factibles medio={r['pct_factibles_medio']}% top factible={r['pct_top_factible']}%  "
          f"t={r['tiempo_subproblemas_s']}s", flush=True)


def step5a_decomposition(seeds=(0, 1, 2), budgets=((3, 30), (4, 20)), reps=1):
    """5a: descomposición desde inicio aleatorio; QAOA (mejor muestra factible) vs exacto."""
    F, D, _ = load_qap()
    runs = []

    def qaoa_solver(F_, D_, perm, S):
        return qiskit_qaoa.solve_subproblem_qaoa(F_, D_, perm, S, reps=reps)

    for seed in seeds:
        # Referencia clásica: búsqueda local con swaps desde la misma inicial.
        p0 = np.random.default_rng(seed).permutation(10).tolist()
        t0 = time.perf_counter()
        _, c_sw, _ = classical.swap_local_search(p0, F, D)
        runs.append({"metodo": "clasico_swaps", "seed": seed, "m": "", "subproblems": "",
                     "C_inicial": qap_cost(p0, F, D), "C_tras_subproblemas": "",
                     "gap_tras_subproblemas": "", "C_final_tras_swaps": c_sw,
                     "gap_final": round(100 * (c_sw - OPTIMUM) / OPTIMUM, 2),
                     "tiempo_subproblemas_s": "", "tiempo_pulido_s": round(time.perf_counter() - t0, 3)})
        print(f"seed={seed} clásico swaps: {runs[-1]['C_inicial']} -> {c_sw}")
        for m, budget in budgets:
            for name, solver in (("hibrido_exacto", classical.solve_subproblem_exact),
                                 ("hibrido_qaoa", qaoa_solver)):
                r = hybrid_search(F, D, m, solver, seed, budget)
                r["metodo"] = name
                runs.append(r)
                print_run(r)
    return save_runs(runs, "n10_runs", N10_COLS)


def step5b_top_sample(seeds=(0, 1, 2), m=3, budget=30, reps=1):
    """5b: QAOA sin post-selección: propuesta = muestra más frecuente (reparada si hace falta)."""
    F, D, _ = load_qap()
    runs = []

    def qaoa_top(F_, D_, perm, S):
        return qiskit_qaoa.solve_subproblem_qaoa(F_, D_, perm, S, reps=reps, mode="top")

    for seed in seeds:
        r = hybrid_search(F, D, m, qaoa_top, seed, budget)
        r["metodo"] = "hibrido_qaoa_top"
        runs.append(r)
        print_run(r)
    return save_runs(runs, "n10_top_runs", N10_COLS)


def hybrid_ils(F, D, seed, max_iters, subsolver=None, m=3, n_sub=2, kick_k=4):
    """5c: búsqueda local iterada (ILS). Presupuesto fijo de iteraciones.

    Cada descenso = swaps hasta óptimo local + n_sub movimientos de subproblema
    (m rutas al azar) resueltos por `subsolver`; si alguno mejora, se vuelve a
    aplicar swaps. Entre descensos, perturbación kick (k-ciclo) sobre la mejor.
    Los subproblemas idénticos se resuelven una sola vez (caché).
    El óptimo conocido NO se usa para parar: solo para calcular el gap.
    """
    N = len(F)
    rng = np.random.default_rng(seed)
    # Generador aparte para los subconjuntos: así inicio y kicks son idénticos en
    # las tres variantes y la diferencia solo puede venir de los subproblemas.
    rng_sub = np.random.default_rng(seed + 10_000)
    cache, sub_hits = {}, 0

    def descent(p):
        nonlocal sub_hits
        p, c, _ = classical.swap_local_search(p, F, D)
        if subsolver is None:
            return p, c
        for _ in range(n_sub):
            S = sorted(rng_sub.choice(N, m, replace=False).tolist())
            key = (tuple(S), tuple(p))
            if key not in cache:
                cache[key] = subsolver(F, D, p, S)
            if cache[key]["cost"] < c:
                sub_hits += 1
                p, c, _ = classical.swap_local_search(cache[key]["perm"], F, D)
        return p, c

    t0 = time.perf_counter()
    p0 = rng.permutation(N).tolist()
    p, c = descent(p0)
    best_c, best_p, best_it, c_first = c, p, 0, c
    for it in range(1, max_iters + 1):
        p, c = descent(classical.kick(best_p, rng, kick_k))
        if c < best_c:
            best_it = it
        if c <= best_c:
            best_c, best_p = c, p
    assert is_permutation(best_p, N)
    return {
        "seed": seed, "iteraciones": max_iters, "C_inicial": qap_cost(p0, F, D),
        "C_primer_descenso": c_first, "C_final": best_c,
        "gap_final": round(100 * (best_c - OPTIMUM) / OPTIMUM, 2),
        "iter_mejor": best_it, "subproblemas_unicos": len(cache),
        "mejoras_por_subproblema": sub_hits,
        "tiempo_s": round(time.perf_counter() - t0, 2), "perm_final": best_p,
    }


def step5c_ils(seeds=(0, 1, 2), max_iters=60, m=3, n_sub=2, reps=1):
    """5c: ablación ILS: sin subproblemas / subsolver exacto / subsolver QAOA."""
    F, D, _ = load_qap()

    def qaoa_solver(F_, D_, perm, S):
        return qiskit_qaoa.solve_subproblem_qaoa(F_, D_, perm, S, reps=reps)

    runs = []
    for seed in seeds:
        for name, solver in (("ils_swaps", None), ("ils_hibrido_exacto", classical.solve_subproblem_exact),
                             ("ils_hibrido_qaoa", qaoa_solver)):
            r = hybrid_ils(F, D, seed, max_iters, solver, m=m, n_sub=n_sub)
            r["metodo"] = name
            runs.append(r)
            print(f"seed={seed} {name}: {r['C_inicial']} -> {r['C_primer_descenso']} -> "
                  f"{r['C_final']} (gap {r['gap_final']}%) iter_mejor={r['iter_mejor']} "
                  f"subproblemas={r['subproblemas_unicos']} mejoras_sub={r['mejoras_por_subproblema']} "
                  f"t={r['tiempo_s']}s", flush=True)
    cols = ["metodo", "seed", "iteraciones", "C_inicial", "C_primer_descenso", "C_final",
            "gap_final", "iter_mejor", "subproblemas_unicos", "mejoras_por_subproblema", "tiempo_s"]
    return save_runs(runs, "n10_ils_runs", cols)


def step5d_classical_ils(seeds=range(10), max_iters=300):
    """5d: ILS solo clásica (swaps + kicks) con más iteraciones: referencia clásica fuerte."""
    F, D, _ = load_qap()
    runs = []
    for seed in seeds:
        r = hybrid_ils(F, D, seed, max_iters)
        r["metodo"] = "ils_swaps"
        runs.append(r)
        print(f"seed={seed}: {r['C_inicial']} -> {r['C_final']} (gap {r['gap_final']}%) "
              f"iter_mejor={r['iter_mejor']} t={r['tiempo_s']}s", flush=True)
    cols = ["metodo", "seed", "iteraciones", "C_inicial", "C_primer_descenso", "C_final",
            "gap_final", "iter_mejor", "tiempo_s"]
    return save_runs(runs, "n10_ils_long_runs", cols)


# ------------------------------------------------------ paso 6: reproducibilidad
def save_environment():
    """Guarda versiones, plataforma, semillas y parámetros en results/environment.json."""
    import platform
    import subprocess
    from importlib.metadata import version

    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                            capture_output=True, text=True).stdout
    (Path(__file__).parent / "requirements.txt").write_text(freeze, encoding="utf-8")
    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {p: version(p) for p in ("numpy", "scipy", "qiskit", "qiskit-aer",
                                             "qiskit-optimization", "qiskit-algorithms",
                                             "matplotlib")},
        "parameters": {
            "seed_qaoa": SEED, "shots": SHOTS, "cobyla_maxiter": qiskit_qaoa.MAXITER,
            "restarts": qiskit_qaoa.RESTARTS, "lambda_factors_N3_N4": qiskit_qaoa.LAMBDA_FACTORS,
            "reps_N3_N4": qiskit_qaoa.REPS, "seed_random_baselines": SEED,
            "n10_seeds": [0, 1, 2], "n10_budgets_m_subproblemas": [[3, 30], [4, 20]],
            "n10_qaoa_reps": 1, "n10_lambda": "1.0 * Delta_sub",
            "n10b_mode": "top (muestra más frecuente), m=3, 30 subproblemas",
            "n10c_ils": {"iters": 60, "m": 3, "n_sub": 2, "kick_k": 4,
                         "seed_subsets": "seed + 10000"},
            "n10d_ils": {"iters": 300, "kick_k": 4, "seeds": list(range(10))},
            "simulator": "qiskit_aer statevector (EstimatorV2 precision=0, SamplerV2)",
            "transpile_basis": ["h", "rx", "rz", "rzz", "cx"],
        },
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    print(f"Guardado {RESULTS_DIR / 'environment.json'} y requirements.txt")
    return env


def write_table(rows, name):
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / f"{name}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join("" if r[c] is None else str(r[c]) for c in cols) + " |" for r in rows]
    (RESULTS_DIR / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nGuardado en {RESULTS_DIR / name}.csv / .md")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    sys.stdout.reconfigure(encoding="utf-8")
    # Uso:
    #   python experiments.py            -> tabla N=3/4 (reutiliza resultados guardados)
    #   python experiments.py --rerun    -> recalcula fuerza bruta y QAOA N=3/4
    #   python experiments.py --n10a     -> 5a descomposición N=10 (~14 min)
    #   python experiments.py --n10b     -> 5b QAOA sin post-selección N=10 (~3 min)
    #   python experiments.py --n10c     -> 5c ILS con ablación N=10 (~25 min)
    #   python experiments.py --n10d     -> 5d ILS clásica larga, 10 semillas (~10 s)
    #   python experiments.py --all      -> todo lo anterior desde cero + entorno
    args = set(sys.argv[1:])
    run_all = "--all" in args
    n10 = {"--n10a", "--n10b", "--n10c", "--n10d"} & args
    if run_all or not n10:
        write_table(step4_comparison(rerun=run_all or "--rerun" in args), "comparison_N3_N4")
    if run_all or "--n10a" in args:
        write_table(step5a_decomposition(), "n10_hybrid")
    if run_all or "--n10b" in args:
        write_table(step5b_top_sample(), "n10_top")
    if run_all or "--n10c" in args:
        write_table(step5c_ils(), "n10_ils")
    if run_all or "--n10d" in args:
        write_table(step5d_classical_ils(), "n10_ils_long")
    save_environment()
