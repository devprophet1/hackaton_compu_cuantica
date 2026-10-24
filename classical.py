"""Baseline clásico: fuerza bruta con itertools.permutations.

Recorre las N! permutaciones una a una (generador), sin guardarlas en memoria.
Para que el bucle en Python puro sea rápido, el costo se evalúa solo sobre
los pares (i<k) con flujo F[i,k] != 0; los pares con flujo 0 no aportan.
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

from data import CONTROLS, load_qap, qap_cost, subinstance

RESULTS_DIR = Path(__file__).parent / "results"


def brute_force(F, D):
    """Devuelve el óptimo exacto y estadísticas de la enumeración."""
    N = len(F)
    pairs = [(i, k, int(F[i][k])) for i in range(N) for k in range(i + 1, N) if F[i][k]]
    Dl = D.tolist()

    t0 = time.perf_counter()
    best, best_perm, n_opt, n_eval = None, None, 0, 0
    for p in itertools.permutations(range(N)):
        n_eval += 1
        c = 0
        for i, k, f in pairs:
            c += f * Dl[p[i]][p[k]]
        if best is None or c < best:
            best, best_perm, n_opt = c, p, 1
        elif c == best:
            n_opt += 1
    elapsed = time.perf_counter() - t0

    assert qap_cost(best_perm, F, D) == best
    return {
        "N": N,
        "method": "brute_force_itertools",
        "perm": list(best_perm),
        "cost": best,
        "n_optimal_perms": n_opt,
        "n_evaluated": n_eval,
        "time_s": round(elapsed, 4),
    }


def swap_local_search(perm, F, D):
    """Búsqueda local con swaps (mejor mejora) hasta un óptimo local.

    Vecindario: intercambiar las estaciones de dos rutas (N(N-1)/2 vecinos).
    """
    p = list(perm)
    cost = qap_cost(p, F, D)
    n_moves = 0
    while True:
        best_delta, best_move = 0, None
        for i, k in itertools.combinations(range(len(p)), 2):
            p[i], p[k] = p[k], p[i]
            delta = qap_cost(p, F, D) - cost
            p[i], p[k] = p[k], p[i]
            if delta < best_delta:
                best_delta, best_move = delta, (i, k)
        if best_move is None:
            return p, cost, n_moves
        i, k = best_move
        p[i], p[k] = p[k], p[i]
        cost += best_delta
        n_moves += 1


def solve_subproblem_exact(F, D, perm, S):
    """Reasigna las rutas S entre sus propias estaciones probando las m! opciones."""
    T = [perm[i] for i in S]
    best_cost, best_perm = None, None
    for sigma in itertools.permutations(T):
        p = list(perm)
        for i, t in zip(S, sigma):
            p[i] = t
        c = qap_cost(p, F, D)
        if best_cost is None or c < best_cost:
            best_cost, best_perm = c, p
    return {"perm": best_perm, "cost": best_cost, "repaired": False}


def kick(perm, rng, k=4):
    """Perturbación para ILS: rota las estaciones de k rutas al azar (un k-ciclo)."""
    p = list(perm)
    idx = rng.choice(len(p), k, replace=False)
    vals = [p[i] for i in idx]
    for i, v in zip(idx, vals[1:] + vals[:1]):
        p[i] = v
    return p


def format_assignment(perm):
    """'ruta 1 -> estación 3, ...' en base 1 para lectura humana."""
    return ", ".join(f"R{i + 1}->E{j + 1}" for i, j in enumerate(perm))


def main(sizes=(3, 4, 10)):
    F, D, _ = load_qap()
    results = []
    for N in sizes:
        r = brute_force(*subinstance(F, D, N))
        r["expected"] = CONTROLS.get(N)
        results.append(r)
        ok = "OK" if r["cost"] == r["expected"] else "FALLO"
        print(f"[{ok}] N={N:2d}  costo={r['cost']:4d}  evaluadas={r['n_evaluated']:>9,}  "
              f"tiempo={r['time_s']:.3f}s  óptimos empatados={r['n_optimal_perms']}")
        print(f"        asignación: {format_assignment(r['perm'])}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "classical_bruteforce.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Guardado en {out}")
    return results


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    main()
