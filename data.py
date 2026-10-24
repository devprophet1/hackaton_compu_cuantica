"""Lectura de qap10.mps.gz (MIPLIB) y función de costo del QAP.

Cómo está construido el MPS (linealización del QAP):
  * X001..X100  -> x_ij binaria, indice = 10*i + j + 1 (ruta i, estación j, base 0).
                  Filas NX01..NX10: sum_j x_ij = 1 ; NX11..NX20: sum_i x_ij = 1.
  * Y????A????  -> y que sustituye al producto x_ij * x_kl, con i<k y j!=l
                  (45 pares de rutas x 90 pares de estaciones = 4050 variables).
                  Su coeficiente en el objetivo (fila NOBJ) es F_ik * D_jl.
  * Filas C..A.. -> restricciones de enlace; cada una contiene exactamente una X
                  con coeficiente -1. Las dos X distintas que comparten fila
                  con una Y nos dicen qué producto x_ij * x_kl representa.

Recuperación de F y D:
  Para un par (i,k) fijo, la matriz de coeficientes sobre (j,l) es F_ik * D,
  así que todas esas matrices son múltiplos de una misma D (rango 1).
  F y D solo están definidas salvo escala (F*c y D/c dan el mismo costo).
  Convención elegida: D con enteros de mcd 1 (resulta ser la distancia
  Manhattan en una cuadrícula) y F = coeficiente / D.
"""
from __future__ import annotations

import gzip
import itertools
from collections import defaultdict
from functools import reduce
from math import gcd
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data"
MPS_PATH = DATA_DIR / "qap10.mps.gz"
OPTIMUM = 340
CONTROLS = {3: 24, 4: 32, 10: OPTIMUM}


def parse_mps(path=MPS_PATH):
    """Devuelve {columna: {fila: valor}} de la sección COLUMNS."""
    cols: dict[str, dict[str, float]] = {}
    section = None
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if not line.startswith(" "):
                section = line.split()[0]
                continue
            if section != "COLUMNS" or "MARKER" in line:
                continue
            tok = line.split()
            entries = cols.setdefault(tok[0], {})
            for row, val in zip(tok[1::2], tok[2::2]):
                entries[row] = float(val)
    return cols


def linearized_coefficients(cols, n=10):
    """Devuelve {(i, j, k, l): coeficiente de y = x_ij * x_kl} con i<k, j!=l."""
    x_of_row = {}
    for name, entries in cols.items():
        if name.startswith("X"):
            for row, val in entries.items():
                if row.startswith("C"):
                    assert val == -1.0 and row not in x_of_row
                    x_of_row[row] = int(name[1:]) - 1

    coef = {}
    for name, entries in cols.items():
        if not name.startswith("Y"):
            continue
        xs = sorted({x_of_row[r] for r in entries if r.startswith("C")})
        assert len(xs) == 2, f"{name} no enlaza exactamente dos x: {xs}"
        (i, j), (k, l) = divmod(xs[0], n), divmod(xs[1], n)
        assert i < k and j != l
        coef[(i, j, k, l)] = entries.get("NOBJ", 0.0)
    assert len(coef) == (n * (n - 1) // 2) * (n * (n - 1)), "faltan variables y"
    return coef


def recover_F_D(coef, n=10):
    """Factoriza coef[(i,j,k,l)] = F_ik * D_jl. Devuelve F, D simétricas (int)."""
    blocks = {}
    for (i, j, k, l), c in coef.items():
        blocks.setdefault((i, k), np.zeros((n, n)))[j, l] = c

    # D: tomamos cualquier bloque no nulo y lo normalizamos a enteros con mcd 1.
    ref = next(b for b in blocks.values() if b.any())
    g = reduce(gcd, ref.astype(int).ravel().tolist())
    D = (ref / g).astype(int)

    F = np.zeros((n, n), dtype=int)
    mask = D != 0
    for (i, k), b in blocks.items():
        ratios = b[mask] / D[mask]
        assert np.allclose(ratios, ratios[0]), f"bloque ({i},{k}) no es múltiplo de D"
        assert float(ratios[0]).is_integer()
        F[i, k] = F[k, i] = int(ratios[0])

    # Comprobación de la factorización completa.
    for (i, j, k, l), c in coef.items():
        assert F[i, k] * D[j, l] == c
    assert (D == D.T).all() and (np.diag(D) == 0).all()
    return F, D


def load_qap(path=MPS_PATH, n=10):
    """Carga el MPS y devuelve (F, D, coef) para la instancia completa."""
    coef = linearized_coefficients(parse_mps(path), n)
    F, D = recover_F_D(coef, n)
    return F, D, coef


def subinstance(F, D, N):
    """Primeras N filas/columnas (rutas 0..N-1 y estaciones 0..N-1)."""
    return F[:N, :N], D[:N, :N]


def qap_cost(perm, F, D):
    """C(pi) = sum_{i<k} F[i,k] * D[pi(i), pi(k)]  (cada par una sola vez).

    perm[i] = estación asignada a la ruta i.
    Como F es simétrica con diagonal 0, equivale a la mitad de la suma completa.
    """
    p = np.asarray(perm)
    return int((F * D[np.ix_(p, p)]).sum()) // 2


def mps_objective(perm, coef):
    """Objetivo calculado directamente con los coeficientes del MPS (control)."""
    n = len(perm)
    return sum(coef[(i, perm[i], k, perm[k])] for i in range(n) for k in range(i + 1, n))


def is_permutation(perm, N):
    return sorted(perm) == list(range(N))


def _exhaustive_min(F, D):
    """Mínimo exacto enumerando todas las permutaciones (solo para validar).

    Para N grande vectoriza: fija un prefijo con itertools y evalúa todas
    las permutaciones del resto de golpe con numpy.
    """
    N = len(F)
    iu, ku = np.triu_indices(N, 1)
    w = F[iu, ku]
    tail = min(N, 7)
    head = N - tail
    best, best_p = None, None
    for prefix in itertools.permutations(range(N), head):
        rest = [s for s in range(N) if s not in prefix]
        P = np.array([prefix + t for t in itertools.permutations(rest)])
        costs = D[P[:, iu], P[:, ku]] @ w
        idx = int(costs.argmin())
        if best is None or costs[idx] < best:
            best, best_p = int(costs[idx]), P[idx].tolist()
    return best, best_p


def validate(full=True, seed=0):
    F, D, coef = load_qap()
    print("F (flujo entre rutas):\n", F)
    print("D (distancia entre estaciones):\n", D)

    # 1) La función de costo coincide con el objetivo linealizado del MPS.
    rng = np.random.default_rng(seed)
    for _ in range(200):
        p = rng.permutation(10).tolist()
        assert qap_cost(p, F, D) == mps_objective(p, coef)
    print("OK  qap_cost == objetivo del MPS en 200 permutaciones aleatorias")

    # 2) Óptimos de control.
    sizes = [3, 4, 10] if full else [3, 4]
    for N in sizes:
        f, d = subinstance(F, D, N)
        best, p = _exhaustive_min(f, d)
        status = "OK " if best == CONTROLS[N] else "FALLO"
        print(f"{status} N={N}: óptimo={best} (esperado {CONTROLS[N]}), perm={p}")
        assert best == CONTROLS[N]


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    validate(full="--quick" not in sys.argv)
