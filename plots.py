"""Figuras (matplotlib) para el README y el notebook. Se guardan en figures/.

Reutiliza los resultados de results/. Lo único que recalcula son las muestras
QAOA de los histogramas (N=3 y N=4, lambda=Delta, p=2), que son deterministas
con la semilla fija y se guardan en results/qaoa_counts_N*.json.

Paleta validada (scripts/validate_palette.js del skill de visualización):
azul #2a78d6 y naranja #eb6834 como categorías; gris neutro para lo que no
es factible / referencia.
"""
from __future__ import annotations

import json
import warnings
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import qiskit_qaoa as qq  # noqa: E402
from data import CONTROLS, OPTIMUM, load_qap, qap_cost, subinstance  # noqa: E402

ROOT = Path(__file__).parent
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"

BLUE, ORANGE = "#2a78d6", "#eb6834"
GRAY, GRID, AXIS = "#c3c2b7", "#e1e0d9", "#c3c2b7"
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
SEQ_BLUES = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.axisbelow": True, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "legend.frameon": False, "legend.fontsize": 9, "figure.dpi": 110, "savefig.dpi": 150,
})


def _save(fig, name):
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    return path


def _bars(ax, x, h, color, width, **kw):
    """Barras finas con 2px de separación del fondo (sin bordes oscuros)."""
    return ax.bar(x, h, width=width, color=color, edgecolor=SURFACE, linewidth=1.5, **kw)


# ------------------------------------------------------------ datos de QAOA
def qaoa_counts(N, lam_factor=1.0, reps=2):
    """Muestras de QAOA para la subinstancia N (cacheadas en results/)."""
    path = RESULTS_DIR / f"qaoa_counts_N{N}_lam{lam_factor:g}_p{reps}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    F, D, _ = load_qap()
    F, D = subinstance(F, D, N)
    qp = qq.build_quadratic_program(F, D)
    lam = lam_factor * qq.penalty_bound(F, D)
    _, op, _ = qq.to_qubo_ising(qp, lam)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        counts = qq.run_qaoa(op, reps)["counts"]
    data = {"N": N, "lambda": lam, "reps": reps, "seed": qq.SEED, "counts": counts}
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return data


def _classify(bitstring, N, F, D):
    perm = qq.decode(qq.bits_from_bitstring(bitstring), N)
    if perm is None:
        return "no factible", None
    cost = qap_cost(perm, F, D)
    return ("óptimo" if cost == CONTROLS[N] else "factible"), (perm, cost)


# ------------------------------------------------------------------ figuras
def fig_histogram(N, top=20):
    """Izq.: las `top` bitstrings más probables. Der.: probabilidad de cada asignación válida."""
    import itertools
    import math

    d = qaoa_counts(N)
    counts, shots = d["counts"], sum(d["counts"].values())
    F, D, _ = load_qap()
    F, D = subinstance(F, D, N)
    items = sorted(counts.items(), key=lambda kv: -kv[1])[:top]
    colors = {"óptimo": BLUE, "factible": ORANGE, "no factible": GRAY}

    labels, probs, cols = [], [], []
    for b, c in items:
        kind, info = _classify(b, N, F, D)
        x = qq.bits_from_bitstring(b)
        rows = "|".join("".join(map(str, x[i * N:(i + 1) * N])) for i in range(N))
        labels.append(rows + (f"  C={info[1]}" if info else ""))
        probs.append(100 * c / shots)
        cols.append(colors[kind])

    # Probabilidad de cada una de las N! asignaciones válidas, ordenadas por costo.
    perms = []
    for p in itertools.permutations(range(N)):
        x = np.zeros(N * N, dtype=int)
        x[np.arange(N) * N + np.array(p)] = 1
        perms.append((qap_cost(p, F, D), 100 * counts.get("".join(map(str, x[::-1])), 0) / shots))
    perms.sort()
    feas = sum(pr for _, pr in perms)
    p_rand = 100 * math.factorial(N) / 2 ** (N * N)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.6),
                                  gridspec_kw={"width_ratios": [2.2, 1]})
    _bars(ax, range(len(items)), probs, cols, 0.8)
    ax.set_xticks(range(len(items)), labels, rotation=90, family="monospace", fontsize=8)
    ax.set_ylabel("probabilidad (% de 2048 muestras)")
    ax.set_xlabel("bitstring: filas = rutas, columnas = estaciones (x_ij); C = costo si es factible",
                  fontsize=8, color=MUTED)
    ax.grid(axis="x", visible=False)
    ax.set_title(f"{top} bitstrings más probables")
    for k, c in colors.items():
        ax.bar(0, 0, color=c, label=k)
    ax.legend(loc="upper right", title="tipo de muestra", title_fontsize=9)

    _bars(ax2, range(len(perms)), [pr for _, pr in perms],
          [BLUE if c == CONTROLS[N] else ORANGE for c, _ in perms], 0.8)
    ax2.set_xticks(range(len(perms)), [c for c, _ in perms], rotation=90, fontsize=8)
    ax2.set_xlabel("costo C de la asignación (ordenado)", fontsize=8, color=MUTED)
    ax2.grid(axis="x", visible=False)
    ax2.set_title(f"Las {len(perms)} asignaciones válidas")
    ax2.text(0.98, 0.97, f"factibles: {feas:.1f}%\nazar: {p_rand:.3g}%", transform=ax2.transAxes,
             ha="right", va="top", fontsize=9, color=INK2)
    fig.suptitle(f"N={N} ({N * N} qubits) · QAOA p={d['reps']}, λ=Δ={d['lambda']:g}, "
                 f"2048 muestras · óptimo de control C={CONTROLS[N]}",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, f"hist_qaoa_N{N}.png")


def fig_feasibility():
    """% de muestras factibles por lambda y p, contra el azar (N=3 y N=4)."""
    grid = json.loads((RESULTS_DIR / "qaoa_grid.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, N in zip(axes, (3, 4)):
        rows = [r for r in grid if r["N"] == N]
        factors = sorted({r["lambda_factor"] for r in rows})
        x = np.arange(len(factors))
        for off, reps, color in ((-0.18, 1, BLUE), (0.18, 2, ORANGE)):
            vals = [100 * next(r["feasible_fraction"] for r in rows
                               if r["lambda_factor"] == f and r["reps"] == reps) for f in factors]
            _bars(ax, x + off, vals, color, 0.34, label=f"QAOA p={reps}")
            i = int(np.argmax(vals))
            ax.annotate(f"{vals[i]:.1f}%", (x[i] + off, vals[i]), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8, color=INK2)
        p_rand = 100 * __import__("math").factorial(N) / 2 ** (N * N)
        ax.axhline(p_rand, color=INK2, linewidth=1)
        ax.text(x[-1] + 0.45, p_rand, f" azar\n {p_rand:.3g}%", va="center", fontsize=8, color=INK2)
        delta = rows[0]["delta"]
        ax.set_xticks(x, [f"{f:g}·Δ\n(λ={f * delta:g})" for f in factors])
        ax.set_title(f"N={N} ({N * N} qubits)")
        ax.set_ylabel("muestras factibles (%)")
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="upper left")
    fig.suptitle("Factibilidad de QAOA según la penalización λ y la profundidad p",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, "factibilidad_lambda_p.png")


def fig_energy(N=3):
    """Distribución de energía QUBO: QAOA vs bitstrings aleatorios (mismo λ y tamaño)."""
    d = qaoa_counts(N)
    F, D, _ = load_qap()
    F, D = subinstance(F, D, N)
    qubo, _, _ = qq.to_qubo_ising(qq.build_quadratic_program(F, D), d["lambda"])
    n, const, lin, quad = qq.qubo_arrays(qubo)

    def energies(counts):
        e, w = [], []
        for b, c in counts.items():
            x = np.array(qq.bits_from_bitstring(b))
            e.append(const + x @ lin + x @ quad @ x)
            w.append(c)
        return np.array(e), np.array(w)

    shots = sum(d["counts"].values())
    rng = np.random.default_rng(qq.SEED)
    rand = Counter("".join(map(str, r[::-1])) for r in rng.integers(0, 2, (shots, n)))
    e_q, w_q = energies(d["counts"])
    e_r, w_r = energies(rand)
    bins = np.arange(0, max(e_q.max(), e_r.max()) + 40, 40)

    fig, ax = plt.subplots(figsize=(9, 3.8))
    for e, w, color, label in ((e_r, w_r, ORANGE, "bitstrings aleatorios"),
                               (e_q, w_q, BLUE, f"QAOA p={d['reps']}")):
        h, _ = np.histogram(e, bins=bins, weights=100 * w / shots)
        ax.stairs(h, bins, color=color, linewidth=2, label=label)
    top = ax.get_ylim()[1] * 1.15
    ax.set_ylim(0, top)
    ax.axvline(CONTROLS[N], color=INK, linewidth=1)
    ax.text(CONTROLS[N], top * 0.98, f"  óptimo factible = {CONTROLS[N]}",
            fontsize=8, color=INK, va="top")
    ax.set_xlabel(f"energía QUBO (costo + penalización, λ={d['lambda']:g})")
    ax.set_ylabel("% de muestras")
    ax.set_title(f"N={N}: QAOA desplaza las muestras hacia energías bajas "
                 f"(media {np.average(e_q, weights=w_q):.0f} frente a {np.average(e_r, weights=w_r):.0f})")
    ax.legend(loc="upper right")
    return _save(fig, f"energia_qubo_N{N}.png")


def _n10_summary():
    """Mejor gap y tiempo de cada método N=10, leído de results/."""
    load = lambda name: json.loads((RESULTS_DIR / name).read_text(encoding="utf-8"))
    brute = next(r for r in load("classical_bruteforce.json") if r["N"] == 10)
    a, b = load("n10_runs.json"), load("n10_top_runs.json")
    c, dd = load("n10_ils_runs.json"), load("n10_ils_long_runs.json")

    def best(runs, gap_key, time_fn):
        r = min(runs, key=lambda r: (r[gap_key], time_fn(r)))
        return r[gap_key], time_fn(r)

    t_sub = lambda r: r["tiempo_subproblemas_s"] + r["tiempo_pulido_s"]
    rows = [
        ("Fuerza bruta (exacto)", 100 * (brute["cost"] - OPTIMUM) / OPTIMUM, brute["time_s"], False),
        ("ILS clásica, 300 iter.", *best(dd, "gap_final", lambda r: r["tiempo_s"]), False),
        ("Swaps, un descenso", *best([r for r in a if r["metodo"] == "clasico_swaps"],
                                     "gap_final", lambda r: r["tiempo_pulido_s"]), False),
        ("ILS + subproblemas QAOA (5c)",
         *best([r for r in c if r["metodo"] == "ils_hibrido_qaoa"], "gap_final", lambda r: r["tiempo_s"]), True),
        ("Descomposición QAOA (5a)",
         *best([r for r in a if r["metodo"] == "hibrido_qaoa"], "gap_final", t_sub), True),
        ("QAOA muestra más frecuente (5b)", *best(b, "gap_final", t_sub), True),
    ]
    return rows


def fig_n10():
    """N=10: mejor gap y tiempo por método (dos paneles, misma lista de métodos)."""
    rows = _n10_summary()[::-1]
    names = [r[0] for r in rows]
    colors = [ORANGE if r[3] else BLUE for r in rows]
    y = np.arange(len(rows))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True,
                                 gridspec_kw={"width_ratios": [1, 1.1]})
    a1.barh(y, [r[1] for r in rows], color=colors, height=0.6, edgecolor=SURFACE, linewidth=1.5)
    for yi, r in zip(y, rows):
        a1.text(r[1] + 0.15, yi, f"{r[1]:.1f}%", va="center", fontsize=8, color=INK2)
    a1.set_yticks(y, names)
    a1.set_xlabel("mejor gap respecto de 340 (%)")
    a1.set_title("Calidad: gap = 100·(C − 340)/340")
    a1.grid(axis="y", visible=False)
    a1.set_xlim(0, max(r[1] for r in rows) * 1.25)

    a2.barh(y, [r[2] for r in rows], color=colors, height=0.6, edgecolor=SURFACE, linewidth=1.5)
    a2.set_xscale("log")
    for yi, r in zip(y, rows):
        a2.text(r[2] * 1.3, yi, f"{r[2]:.3g} s", va="center", fontsize=8, color=INK2)
    a2.set_xlabel("tiempo de esa corrida (s, escala log)")
    a2.set_title("Tiempo")
    a2.grid(axis="y", visible=False)
    a2.set_xlim(1e-3, 5e3)
    a1.bar(0, 0, color=BLUE, label="clásico")
    a1.bar(0, 0, color=ORANGE, label="usa Qiskit (simulado)")
    fig.suptitle("Instancia completa N=10 (óptimo conocido 340, solo soluciones factibles)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.legend(loc="upper right", ncol=2, bbox_to_anchor=(0.99, 1.0))
    return _save(fig, "n10_comparacion.png")


def fig_matrices():
    """Matrices F (flujo entre rutas) y D (distancia entre estaciones) recuperadas del MPS."""
    F, D, _ = load_qap()
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seqblue", SEQ_BLUES)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, M, title, lab in ((axes[0], F, "F: flujo / demanda compartida entre rutas", "R"),
                              (axes[1], D, "D: distancia entre estaciones (Manhattan)", "E")):
        im = ax.imshow(M, cmap=cmap, vmin=0)
        ax.grid(False)
        for i in range(10):
            for j in range(10):
                v = M[i, j]
                ax.text(j, i, v if v else "·", ha="center", va="center", fontsize=7,
                        color="#ffffff" if v > 0.6 * M.max() else INK2)
        ticks = [f"{lab}{k + 1}" for k in range(10)]
        ax.set_xticks(range(10), ticks, fontsize=7)
        ax.set_yticks(range(10), ticks, fontsize=7)
        ax.set_title(title, fontsize=10)
        for s in ax.spines.values():
            s.set_visible(False)
        fig.colorbar(im, ax=ax, shrink=0.75).outline.set_visible(False)
    fig.suptitle("Datos recuperados de qap10.mps.gz (convención: D entera con mcd 1)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, "matrices_F_D.png")


# Posiciones de las estaciones: D es la distancia Manhattan en esta cuadrícula.
STATION_XY = [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1), (0, 2), (1, 2)]


def fig_layout():
    """Esquema ilustrativo: asignación aleatoria vs óptima sobre la cuadrícula de estaciones."""
    F, D, _ = load_qap()
    for a in range(10):
        for b in range(10):
            (xa, ya), (xb, yb) = STATION_XY[a], STATION_XY[b]
            assert abs(xa - xb) + abs(ya - yb) == D[a, b]
    brute = next(r for r in json.loads((RESULTS_DIR / "classical_bruteforce.json")
                                       .read_text(encoding="utf-8")) if r["N"] == 10)
    p_rand = np.random.default_rng(0).permutation(10).tolist()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    for ax, perm, title in ((axes[0], p_rand, "Asignación aleatoria (seed 0)"),
                            (axes[1], brute["perm"], "Asignación óptima")):
        fmax = F.max()
        for i in range(10):
            for k in range(i + 1, 10):
                if F[i, k]:
                    (x1, y1), (x2, y2) = STATION_XY[perm[i]], STATION_XY[perm[k]]
                    ax.plot([x1, x2], [y1, y2], color=BLUE, alpha=0.25 + 0.6 * F[i, k] / fmax,
                            linewidth=0.6 + 3.4 * F[i, k] / fmax, solid_capstyle="round", zorder=1)
        for s, (x, y) in enumerate(STATION_XY):
            ax.scatter(x, y, s=520, color=SURFACE, edgecolor=INK2, linewidth=1, zorder=2)
            route = perm.index(s)
            ax.text(x, y, f"R{route + 1}", ha="center", va="center", fontsize=9,
                    fontweight="bold", zorder=3)
            ax.text(x - 0.2, y - 0.27, f"E{s + 1}", ha="right", va="center", fontsize=7,
                    color=INK2, zorder=3,
                    bbox={"fc": SURFACE, "ec": "none", "pad": 0.6, "alpha": 0.9})
        ax.set_title(f"{title} · C = {qap_cost(perm, F, D)}")
        ax.set_xlim(-0.6, 3.6)
        ax.set_ylim(2.6, -0.6)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.suptitle("Track B (ilustrativo): rutas R asignadas a estaciones E; "
                 "grosor de línea = flujo F entre rutas", x=0.01, ha="left", fontweight="bold")
    fig.text(0.01, 0.0, "Las posiciones reproducen exactamente D (distancia Manhattan). "
             "Unidades abstractas: no representa una ciudad real.", fontsize=8, color=MUTED)
    fig.tight_layout()
    return _save(fig, "esquema_asignacion.png")


def fig_circuit():
    """Circuito QAOA (N=3, p=1, λ=Δ): capa de costo (Rz, Rzz) + capa mezcladora (Rx)."""
    from qiskit import transpile
    from qiskit.circuit.library import QAOAAnsatz

    F, D, _ = load_qap()
    F, D = subinstance(F, D, 3)
    _, op, _ = qq.to_qubo_ising(qq.build_quadratic_program(F, D), qq.penalty_bound(F, D))
    h = (op / np.max(np.abs(op.coeffs.real))).simplify()
    circ = transpile(QAOAAnsatz(h, reps=1), basis_gates=["h", "rx", "rz", "rzz"],
                     optimization_level=1, seed_transpiler=qq.SEED)
    # Ángulos optimizados de la grilla (N=3, λ=Δ, p=1), redondeados para que se lean.
    grid = json.loads((RESULTS_DIR / "qaoa_grid.json").read_text(encoding="utf-8"))
    theta = next(r["theta"] for r in grid if r["N"] == 3 and r["lambda_factor"] == 1.0
                 and r["reps"] == 1)
    circ = circ.assign_parameters(np.round(theta, 2))
    fig = circ.draw("mpl", fold=-1, idle_wires=False, scale=0.7,
                    style={"backgroundcolor": SURFACE})
    fig.suptitle(f"Circuito QAOA para N=3 (9 qubits = x_ij, p=1, ángulos optimizados β={theta[0]:.2f}, "
                 f"γ={theta[1]:.2f}): H → capa de costo (Rz, Rzz) → capa mezcladora (Rx). "
                 f"Profundidad {circ.depth()}, {sum(circ.count_ops().values())} puertas",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    return _save(fig, "circuito_qaoa_N3.png")


def make_all():
    paths = [fig_matrices(), fig_layout(), fig_circuit(), fig_histogram(3), fig_histogram(4),
             fig_feasibility(), fig_energy(3), fig_n10()]
    for p in paths:
        print("Guardado", p)
    return paths


if __name__ == "__main__":
    make_all()
