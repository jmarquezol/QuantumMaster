"""
Benasque Quantum Hackathon - ALGORITHMIC SHOWDOWN
=================================================
Classical Heuristic (NN + 2-opt) vs Simulated Annealing vs DMRG
on the exact same 4-node, 16-qubit TSP subproblem.

Nodes  : Benasque (3), Cerler (4), Ancils (5), Banos de Benasque (8)
Qubits : 4 nodes x 4 positions = 16
QUBO   : standard TSP (each node exactly once, each position exactly once)

pip install numpy dimod neal physics-tenpy
"""

import itertools
import time
import numpy as np
import dimod
import warnings, logging
warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)

from tenpy.networks.site import SpinHalfSite
from tenpy.networks.mpo import MPO
from tenpy.networks.mps import MPS
from tenpy.models.model import MPOModel
from tenpy.models.lattice import Chain
import tenpy.linalg.np_conserved as npc
from tenpy.algorithms import dmrg

# ─────────────────────────────────────────────────────────────
# PROBLEM DATA (identical to the QAOA script)
# ─────────────────────────────────────────────────────────────

# Same 4-node problem as the QAOA script.
# t=0 edges = road/car connections — physically valid.
# Tours are NOT all zero: optimal=2.17h, worst=2.92h, gap=0.75h.
NODES     = {3: "Benasque", 4: "Cerler", 5: "Ancils", 8: "Banos de Benasque"}
NODE_LIST = [3, 4, 5, 8]
N         = len(NODE_LIST)        # 4 nodes
P         = N                     # 4 route positions
NUM_QUBITS = N * P                # 16 qubits

RAW_EDGES = {
    (3, 4): 1.4166666666666667,
    (3, 5): 0.75,
    (3, 8): 0.0,
    (4, 5): 1.5,
    (4, 8): 0.0,
    (5, 8): 0.0,
}

DIST = {}
for (i, j), c in RAW_EDGES.items():
    DIST[(i, j)] = float(c)
    DIST[(j, i)] = float(c)
for i in NODE_LIST:
    DIST[(i, i)] = 0.0

# QUBO hyperparameters (same as QAOA script)
A = 10.0   # penalty for constraint violations
B = 1.0    # route cost weight

I2   = np.eye(2)
N_op = np.diag([0., 1.])


# ─────────────────────────────────────────────────────────────
# QUBO — standard TSP formulation
#
# x[i, p] = 1 if node i is visited at position p
# var_index(i, p) = i * P + p
#
# H = A * sum_i (sum_p x[i,p] - 1)^2   <- each node exactly once
#   + A * sum_p (sum_i x[i,p] - 1)^2   <- each position exactly once
#   + B * sum_p sum_{i,j} d(i,j) * x[i,p] * x[j,(p+1)%P]
# ─────────────────────────────────────────────────────────────

def var_index(i_idx, pos):
    return i_idx * P + pos


def build_qubo():
    Q = {}

    def add(a, b, val):
        key = (min(a, b), max(a, b))
        Q[key] = Q.get(key, 0.0) + val

    # Constraint 1: each node appears exactly once
    for i in range(N):
        vars_i = [var_index(i, p) for p in range(P)]
        for a in vars_i:
            add(a, a, -A)
        for ia in range(len(vars_i)):
            for ib in range(ia + 1, len(vars_i)):
                add(vars_i[ia], vars_i[ib], 2.0 * A)

    # Constraint 2: each position contains exactly one node
    for p in range(P):
        vars_p = [var_index(i, p) for i in range(N)]
        for a in vars_p:
            add(a, a, -A)
        for ia in range(len(vars_p)):
            for ib in range(ia + 1, len(vars_p)):
                add(vars_p[ia], vars_p[ib], 2.0 * A)

    # Cost: cyclic tour
    for p in range(P):
        p_next = (p + 1) % P
        for i_idx, i in enumerate(NODE_LIST):
            for j_idx, j in enumerate(NODE_LIST):
                coeff = B * DIST[(i, j)]
                add(var_index(i_idx, p), var_index(j_idx, p_next), coeff)

    return Q


# ─────────────────────────────────────────────────────────────
# DECODE & EVALUATE
# ─────────────────────────────────────────────────────────────

def decode_route(bits):
    """Decode a bitstring to a route. Returns None if invalid."""
    route    = []
    used     = set()
    for p in range(P):
        chosen = [i for i in range(N) if bits[var_index(i, p)] > 0.5]
        if len(chosen) != 1:
            return None
        i_idx = chosen[0]
        if i_idx in used:
            return None
        used.add(i_idx)
        route.append(NODE_LIST[i_idx])
    return route if len(used) == N else None


def tour_cost(route):
    return sum(DIST[(route[k], route[(k + 1) % N])] for k in range(N))


def route_str(route):
    if route is None:
        return "Invalid"
    return " -> ".join(NODES[n] for n in route) + f" -> {NODES[route[0]]}"


def qubo_energy(Q, bits):
    e = 0.0
    for (a, b), c in Q.items():
        if a == b: e += c * bits[a]
        else:      e += c * bits[a] * bits[b]
    return e


# ─────────────────────────────────────────────────────────────
# BRUTE FORCE
# ─────────────────────────────────────────────────────────────

def brute_force(Q):
    best_route = None
    best_cost  = float('inf')
    best_e     = float('inf')

    for perm in itertools.permutations(range(N)):
        route = [NODE_LIST[i] for i in perm]
        cost  = tour_cost(route)

        # Encode as bitstring
        bits = [0] * NUM_QUBITS
        for pos, i_idx in enumerate(perm):
            bits[var_index(i_idx, pos)] = 1

        e = qubo_energy(Q, bits)

        if cost < best_cost:
            best_cost  = cost
            best_route = route
            best_e     = e

    return best_route, best_cost, best_e


# ─────────────────────────────────────────────────────────────
# MPO / DMRG
# ─────────────────────────────────────────────────────────────

def build_mpo(Q, total_qubits):
    linear    = {a: c for (a, b), c in Q.items() if a == b}
    quadratic = {(a, b): c for (a, b), c in Q.items() if a != b}

    starts_at = {q: [] for q in range(total_qubits)}
    ends_at   = {q: [] for q in range(total_qubits)}
    for (a, b), c in quadratic.items():
        starts_at[a].append((b, c))
        ends_at[b].append((a, c))

    mpo_np = []; open_terms = []
    for q in range(total_qubits):
        chi_left  = len(open_terms) + 2
        new_open  = list(open_terms)
        for (b, c) in starts_at[q]: new_open.append((q, b))
        for (a, c) in ends_at[q]:
            if (a, q) in new_open: new_open.remove((a, q))

        chi_right = len(new_open) + 2
        W = np.zeros((chi_left, chi_right, 2, 2))

        W[0, 0] += I2
        if q in linear: W[0, 1] += linear[q] * N_op
        for (b, c) in starts_at[q]:
            W[0, new_open.index((q, b)) + 2] += c * N_op
        for k, (a, b) in enumerate(open_terms):
            alpha = k + 2
            if b == q:
                W[alpha, 1] += N_op
            elif (a, b) in new_open:
                W[alpha, new_open.index((a, b)) + 2] += I2
        W[1, 1] += I2

        mpo_np.append(W)
        open_terms = new_open
    return mpo_np


def run_dmrg(Q, total_qubits, chi_max=32):
    mpo_np = build_mpo(Q, total_qubits)
    site   = SpinHalfSite(conserve='None')
    sites  = [site] * total_qubits

    Ws = []
    for q, W_np in enumerate(mpo_np):
        chi_L, chi_R, _, _ = W_np.shape
        Ws.append(npc.Array.from_ndarray(
            W_np,
            [npc.LegCharge.from_trivial(chi_L),
             npc.LegCharge.from_trivial(chi_R),
             site.leg, site.leg.conj()],
            labels=['wL', 'wR', 'p', 'p*']
        ))

    mpo_tenpy = MPO(sites, Ws, bc='finite', IdL=0, IdR=1)

    class QuboModel(MPOModel):
        def __init__(self):
            lat = Chain(len(sites), sites[0], bc='open', bc_MPS='finite')
            self.lat = lat; self.H_MPO = mpo_tenpy
            MPOModel.__init__(self, lat, mpo_tenpy)

    model   = QuboModel()
    initial = ['down'] * total_qubits   # |0...0> best start for TSP QUBO
    psi     = MPS.from_product_state(sites, initial, bc='finite')

    params = {
        'trunc_params': {'chi_max': chi_max, 'svd_min': 1e-12},
        'N_sweeps_check': 1,
        'min_sweeps': 6,
        'max_sweeps': 30,
        'mixer': 'DensityMatrixMixer',
        'mixer_params': {'amplitude': 1e-3, 'decay': 1.5, 'disable_after': 15},
    }
    eng  = dmrg.TwoSiteDMRGEngine(psi, model, params)
    E, psi = eng.run()

    # Read out probabilities and threshold
    probs = np.array([
        float((1 - psi.expectation_value_term([('Sigmaz', q)])) / 2)
        for q in range(psi.L)
    ])
    bits = (probs > 0.5).astype(int).tolist()
    return E, bits


# ─────────────────────────────────────────────────────────────
# CLASSICAL HEURISTICS
# ─────────────────────────────────────────────────────────────

def nearest_neighbor_tsp(start_idx=0):
    """Greedy nearest-neighbour tour starting from NODE_LIST[start_idx]."""
    unvisited = list(range(N))
    current   = start_idx
    route_idx = [current]
    unvisited.remove(current)

    while unvisited:
        nearest = min(unvisited, key=lambda j: DIST[(NODE_LIST[current], NODE_LIST[j])])
        route_idx.append(nearest)
        unvisited.remove(nearest)
        current = nearest

    return [NODE_LIST[i] for i in route_idx]


def two_opt_tsp(route):
    """2-opt improvement on a TSP tour."""
    best = list(route)

    def cost(r):
        return sum(DIST[(r[k], r[(k+1) % N])] for k in range(N))

    improved = True
    while improved:
        improved = False
        for i in range(N - 1):
            for j in range(i + 2, N):
                if j == N - 1 and i == 0:
                    continue
                new = best[:i] + best[i:j+1][::-1] + best[j+1:]
                if cost(new) < cost(best) - 1e-9:
                    best = new; improved = True
    return best


def run_heuristic():
    t0 = time.time()
    # Try all starting nodes and keep the best
    best_route = None
    best_cost  = float('inf')
    for start in range(N):
        r = nearest_neighbor_tsp(start)
        r = two_opt_tsp(r)
        c = tour_cost(r)
        if c < best_cost:
            best_cost = c; best_route = r
    t_h = time.time() - t0
    return best_route, best_cost, t_h


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print(" ALGORITHMIC SHOWDOWN — 4-node Benasque TSP (16 qubits)")
    print("=" * 65)

    print(f"\nNodes : {[NODES[n] for n in NODE_LIST]}")
    print(f"Qubits: {N} nodes x {P} positions = {NUM_QUBITS}")
    print(f"Search space: 2^{NUM_QUBITS} = {2**NUM_QUBITS:,}")
    print(f"Valid TSP tours: {N-1}! = {int(__import__('math').factorial(N-1))}")

    print(f"\nDistance matrix (hours):")
    header = f"{'':12}" + "".join(f"{NODES[j]:>15}" for j in NODE_LIST)
    print(f"  {header}")
    for i in NODE_LIST:
        row = f"{NODES[i]:12}" + "".join(f"{DIST[(i,j)]:>15.3f}" for j in NODE_LIST)
        print(f"  {row}")

    # Build QUBO
    Q = build_qubo()
    print(f"\nQUBO: {len(Q)} terms  (A={A}, B={B})")

    # ── BRUTE FORCE ───────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f" BRUTE FORCE (exact)")
    print(f"{'─'*65}")
    t0 = time.time()
    bf_route, bf_cost, bf_e = brute_force(Q)
    t_bf = time.time() - t0
    print(f"  Route      : {route_str(bf_route)}")
    print(f"  Tour cost  : {bf_cost:.4f}h")
    print(f"  QUBO energy: {bf_e:.4f}")
    print(f"  Wall time  : {t_bf*1000:.1f}ms")

    # ── HEURISTIC ─────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f" CLASSICAL HEURISTIC (Nearest Neighbour + 2-opt)")
    print(f"{'─'*65}")
    h_route, h_cost, t_h = run_heuristic()
    # Encode to get QUBO energy
    h_bits = [0] * NUM_QUBITS
    for pos, node in enumerate(h_route):
        h_bits[var_index(NODE_LIST.index(node), pos)] = 1
    h_e = qubo_energy(Q, h_bits)
    print(f"  Route      : {route_str(h_route)}")
    print(f"  Tour cost  : {h_cost:.4f}h")
    print(f"  QUBO energy: {h_e:.4f}")
    print(f"  Wall time  : {t_h*1000:.1f}ms")

    # ── SIMULATED ANNEALING ───────────────────────────────────
    print(f"\n{'─'*65}")
    print(f" SIMULATED ANNEALING ({NUM_QUBITS} qubits)")
    print(f"{'─'*65}")
    t0  = time.time()
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    try:
        import neal; sampler = neal.SimulatedAnnealingSampler()
    except ImportError:
        sampler = dimod.SimulatedAnnealingSampler()

    sampleset = sampler.sample(bqm, num_reads=2000, num_sweeps=2000)
    t_sa = time.time() - t0

    best_sa   = sampleset.first
    bits_sa   = [int(best_sa.sample.get(q, 0)) for q in range(NUM_QUBITS)]
    route_sa  = decode_route(bits_sa)
    cost_sa   = tour_cost(route_sa) if route_sa else float('inf')
    unique_e  = len(set(float(s.energy) for s in sampleset.record))

    print(f"  Unique energy levels sampled: {unique_e}")
    print(f"  Best QUBO energy found      : {best_sa.energy:.4f}")
    print(f"  Route      : {route_str(route_sa)}")
    print(f"  Tour cost  : {cost_sa:.4f}h" if route_sa else "  Route: Invalid")
    print(f"  Wall time  : {t_sa:.2f}s")

    # ── DMRG ─────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f" DMRG TENSOR NETWORK ({NUM_QUBITS} qubits)")
    print(f"{'─'*65}")
    mpo_tmp   = build_mpo(Q, NUM_QUBITS)
    bond_dims = [W.shape[0] for W in mpo_tmp] + [mpo_tmp[-1].shape[1]]
    mem_mb    = sum(W.size for W in mpo_tmp) * 8 / 1e6
    print(f"  MPO bond dim max : {max(bond_dims)}")
    print(f"  MPO memory       : {mem_mb:.2f} MB")

    t0 = time.time()
    E_dmrg, bits_dmrg = run_dmrg(Q, NUM_QUBITS, chi_max=32)
    t_dmrg = time.time() - t0

    route_dmrg = decode_route(bits_dmrg)
    cost_dmrg  = tour_cost(route_dmrg) if route_dmrg else float('inf')
    e_dmrg_q   = qubo_energy(Q, bits_dmrg)

    print(f"  Converged QUBO energy: {E_dmrg:.4f}")
    print(f"  Route      : {route_str(route_dmrg)}")
    print(f"  Tour cost  : {cost_dmrg:.4f}h" if route_dmrg else "  Route: Invalid")
    print(f"  Wall time  : {t_dmrg:.2f}s")

    # ── SUMMARY ───────────────────────────────────────────────
    w = 14
    print(f"\n{'='*65}")
    print(f" EXECUTIVE SUMMARY")
    print(f"{'='*65}")
    print(f"\n{'':22} {'Brute force':>{w}} {'Heuristic':>{w}} {'SA':>{w}} {'DMRG':>{w}}")
    print(f"{'─'*65}")
    print(f"{'Search space':22} {'N!':>{w}} {'O(N²)':>{w}} {'2^'+str(NUM_QUBITS):>{w}} {'2^'+str(NUM_QUBITS):>{w}}")

    def fmt_cost(c): return f"{c:.4f}h" if c < float('inf') else "N/A"
    print(f"{'Tour cost (h)':22} {fmt_cost(bf_cost):>{w}} {fmt_cost(h_cost):>{w}} {fmt_cost(cost_sa):>{w}} {fmt_cost(cost_dmrg):>{w}}")
    print(f"{'QUBO energy':22} {bf_e:>{w}.4f} {h_e:>{w}.4f} {best_sa.energy:>{w}.4f} {E_dmrg:>{w}.4f}")
    print(f"{'Valid route':22} {'True':>{w}} {'True':>{w}} {str(route_sa is not None):>{w}} {str(route_dmrg is not None):>{w}}")
    print(f"{'Wall time':22} {str(round(t_bf*1000,1))+'ms':>{w}} {str(round(t_h*1000,1))+'ms':>{w}} {str(round(t_sa,2))+'s':>{w}} {str(round(t_dmrg,2))+'s':>{w}}")

    print(f"\n CONCLUSION:")
    print(f"  {NUM_QUBITS} qubits — fully simulable on a laptop (2^{NUM_QUBITS} = {2**NUM_QUBITS:,} states).")
    print(f"  Brute force checks all {int(__import__('math').factorial(N-1))} valid tours in {t_bf*1000:.0f}ms.")
    print(f"  Heuristic finds {'optimal' if abs(h_cost-bf_cost)<1e-6 else 'near-optimal'} solution in {t_h*1000:.0f}ms.")
    print(f"  SA samples stochastically — {'found optimum' if route_sa and abs(cost_sa-bf_cost)<1e-6 else 'approximate'}.")
    print(f"  DMRG finds exact ground state via tensor compression — {'found optimum' if route_dmrg and abs(cost_dmrg-bf_cost)<1e-6 else 'approximate'}.")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()