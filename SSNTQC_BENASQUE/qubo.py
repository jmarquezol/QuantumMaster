"""
Benasque Quantum Hackathon - ALGORITHMIC SHOWDOWN
=================================================
Comparison Metric 1: Stochastic Landscape Exploration (Annealing) 
vs. Deterministic Variational Sweeping (Tensor Networks / DMRG).
"""

import numpy as np
import networkx as nx
import dimod
import neal
import time
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
# 1. SHARED DATA & GRAPH PREPARATION
# ─────────────────────────────────────────────────────────────

PLACES = {
    1:  ("Pico Aneto",           "Peak",         3404, "Snow",     "Snow"),
    3:  ("Benasque",             "Town",         1135, "Urban",    "Urban"),
    6:  ("Portillon de Benas",   "Landmark",     2444, "Mountain", "Trail"),
    10: ("Forau d'aiguallut",    "Landmark",     2020, "Mountain", "Trail"),
    11: ("Tres Cascadas",        "Landmark",     1900, "Mountain", "Trail"),
    13: ("Tuca Maladeta",        "Peak",         3312, "Snow",     "Mountain"),
    14: ("Cap Llauset",          "Refugio",      2425, "Snow",     "Mountain"),
    15: ("Ibon Creguena",        "Lake",         2632, "Snow",     "Mountain"),
    16: ("Batisielles",          "Lake",         2216, "Mountain", "Trail"),
    19: ("Tempestades",          "Peak",         3289, "Snow",     "Snow"),
    20: ("La Besurta",           "Resting area", 1860, "Trail",    "Trail"),
    21: ("La Renclusa",          "Refugio",      2160, "Snow",     "Trail"),
    22: ("Escaleta",             "Lake",         2630, "Snow",     "Mountain"),
    23: ("Mulleres",             "Peak",         3013, "Snow",     "Snow"),
    24: ("Salterillo",           "Lake",         2460, "Snow",     "Mountain"),
}

EDGES_RAW = {
    (1, 20): 4.83, (1, 24): 4.58, (3, 16): 3.17, (3, 20): 0.0, 
    (6, 10): 3.42, (10, 20): 1.00, (10, 21): 1.83, (13, 20): 4.33,
    (14, 19): 3.67, (14, 21): 5.83, (20, 21): 4.33, (20, 22): 3.08,
    (21, 24): 1.58, (22, 23): 1.25, (23, 24): 0.83, (3, 11): 1.17
}

SCENIC_VALUE = {"Peak": 4, "Lake": 3, "Landmark": 3, "Refugio": 2, "Resting area": 1, "Town": 0}
GEAR_ORDER = ["Urban", "Trail", "Mountain", "Snow"]

# --- SCENARIO TO PROVE OPTIMIZATION DIFFERENCES ---
SCENARIO = {
    "gear":   "Mountain",
    "season": "summer",
    "base":   3,       # Benasque
    "D_MAX":  2500,
    "P":      3,       # 5 steps
    "N":      5,       # 5 locations
}

REWARD = 2.0    # ganancia base por visitar cualquier nodo
ALPHA  = 0.4     # peso adicional por valor escenico
LAM    = 50.0    # penalizacion por violar C1/C2 (debe ser >> REWARD*P)
BIG    = 200.0   # penalizacion por arista inexistente (debe ser >> REWARD)

def get_shortest_paths(edges_raw):
    G = nx.Graph()
    for (i, j), t in edges_raw.items():
        G.add_edge(i, j, weight=t)
        G.add_edge(j, i, weight=t) 
    return dict(nx.floyd_warshall(G, weight='weight'))

def filter_candidates_fixed(s, shortest_paths):
    gear_idx = GEAR_ORDER.index(s["gear"])
    base = s["base"]
    valid_nodes = [i for i in PLACES if GEAR_ORDER.index(PLACES[i][4] if s["season"] == "summer" else PLACES[i][3]) <= gear_idx]
    reachable = [n for n in valid_nodes if n in shortest_paths[base]]
    candidates = [n for n in reachable if n != base]
    if "N" in s and s["N"] < len(candidates):
        candidates = sorted(candidates, key=lambda n: SCENIC_VALUE.get(PLACES[n][1], 0), reverse=True)[:s["N"]]
        candidates = sorted(candidates)
    true_edges = {}
    for i in candidates + [base]:
        for j in candidates + [base]:
            if i != j and j in shortest_paths[i]:
                true_edges[(i, j)] = shortest_paths[i][j]
    return candidates, true_edges

def build_qubo_terms(candidates, edges, base, P):
    N = len(candidates)
    TOTAL = N * P
    def qubit(ii, k): return ii + k * N
    terms = {}
    def add(a, b, val):
        if abs(val) < 1e-12: return
        key = (min(a, b), max(a, b))
        terms[key] = terms.get(key, 0.0) + val

    for k in range(P):
        for ii, i in enumerate(candidates):
            q_ik = qubit(ii, k)
            vi = SCENIC_VALUE.get(PLACES[i][1], 0)
            add(q_ik, q_ik, -(REWARD + ALPHA * vi))
            if k == 0: add(q_ik, q_ik, edges.get((base, i), BIG))
            if k == P - 1: add(q_ik, q_ik, edges.get((i, base), BIG))
            if k < P - 1:
                for jj, j in enumerate(candidates):
                    add(q_ik, qubit(jj, k + 1), edges.get((i, j), BIG))
            for jj in range(ii + 1, N): add(q_ik, qubit(jj, k), 2 * LAM)
            for kp in range(k + 1, P): add(q_ik, qubit(ii, kp), 2 * LAM)
    return terms, N, TOTAL

def calculate_route_time(route, base, edges):
    """Calculates the physical travel time of a given route."""
    if not route:
        return 0.0
    full_route = [base] + route + [base]
    total_time = 0.0
    for i in range(len(full_route) - 1):
        # We assume edges dict has the shortest paths pre-calculated
        total_time += edges.get((full_route[i], full_route[i+1]), 0.0)
    return total_time

# ─────────────────────────────────────────────────────────────
# 2. TENSOR NETWORK FUNCTIONS (MPO/DMRG)
# ─────────────────────────────────────────────────────────────

I2   = np.eye(2)
N_op = np.diag([0., 1.])

def build_mpo_from_terms(terms, n_qubits):
    linear    = {a: c for (a,b),c in terms.items() if a == b}
    quadratic = {(a,b): c for (a,b),c in terms.items() if a != b}
    starts_at = {q: [] for q in range(n_qubits)}
    ends_at   = {q: [] for q in range(n_qubits)}
    for (a,b), c in quadratic.items():
        starts_at[a].append((b, c))
        ends_at[b].append((a, c))

    mpo_np = []
    open_terms = []
    for q in range(n_qubits):
        chi_left = len(open_terms) + 2
        new_open = list(open_terms)
        for (b, c) in starts_at[q]: new_open.append((q, b))
        for (a, c) in ends_at[q]:
            if (a, q) in new_open: new_open.remove((a, q))

        chi_right = len(new_open) + 2
        W = np.zeros((chi_left, chi_right, 2, 2))
        W[0, 0] += I2
        if q in linear: W[0, 1] += linear[q] * N_op
        for (b, c) in starts_at[q]:
            slot = new_open.index((q, b)) + 2
            W[0, slot] += c * N_op
        for k, (a, b) in enumerate(open_terms):
            alpha = k + 2
            if b == q: W[alpha, 1] += N_op
            elif (a, b) in new_open:
                slot = new_open.index((a, b)) + 2
                W[alpha, slot] += I2
        W[1, 1] += I2
        mpo_np.append(W)
        open_terms = new_open
    return mpo_np

def mpo_np_to_tenpy(mpo_np, sites):
    Ws = []
    for q, W_np in enumerate(mpo_np):
        chi_L, chi_R, d, _ = W_np.shape
        W_npc = npc.Array.from_ndarray(
            W_np, [npc.LegCharge.from_trivial(chi_L), npc.LegCharge.from_trivial(chi_R), sites[q].leg, sites[q].leg.conj()],
            labels=['wL', 'wR', 'p', 'p*']
        )
        Ws.append(W_npc)
    return MPO(sites, Ws, bc='finite', IdL=0, IdR=1)

class QuboModel(MPOModel):
    def __init__(self, sites, mpo):
        lat = Chain(len(sites), sites[0], bc='open', bc_MPS='finite')
        self.lat = lat
        self.H_MPO = mpo
        MPOModel.__init__(self, lat, mpo)

# ─────────────────────────────────────────────────────────────
# 3. THE ALGORITHMIC SHOWDOWN
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 75)
    print(" OPTIMIZATION PARADIGMS: STOCHASTIC (ANNEALING) VS DETERMINISTIC (DMRG)")
    print("=" * 75)

    shortest_paths = get_shortest_paths(EDGES_RAW)
    candidates, true_edges = filter_candidates_fixed(SCENARIO, shortest_paths)
    base, P = SCENARIO["base"], SCENARIO["P"]
    
    terms, N, TOTAL_QUBITS = build_qubo_terms(candidates, true_edges, base, P)
    search_space = 2**TOTAL_QUBITS
    
    print(f"\n[PROBLEM COMPLEXITY]")
    print(f"Mapping {N} locations over {P} steps -> {TOTAL_QUBITS} Qubits.")
    print(f"Total possible route combinations to search: {search_space:,}\n")

    # ---------------------------------------------------------
    # METHOD A: SIMULATED ANNEALING (STOCHASTIC)
    # ---------------------------------------------------------
    print("── METHOD A: SIMULATED ANNEALING ──")
    print("Behavior: Probabilistic exploration. Employs thermal/quantum fluctuations")
    print("to jump out of local minima and sample the landscape globally.")
    
    start_time = time.time()
    bqm = dimod.BinaryQuadraticModel.from_qubo(terms)
    sampler = neal.SimulatedAnnealingSampler()
    
    # Run 1000 quantum/thermal experiments
    sampleset = sampler.sample(bqm, num_reads=1000)
    anneal_time = time.time() - start_time
    
    best_energy_sa = sampleset.first.energy
    
    # We count how many DIFFERENT answers the annealer found
    unique_energies = set([float(sample.energy) for sample in sampleset.record])
    
    route_anneal = []
    best_sample = sampleset.first.sample
    for step in range(P):
        for ii, node_id in enumerate(candidates):
            if best_sample[ii + step * N] == 1: route_anneal.append(node_id)
            
    # Calculate Travel Time for Method A
    time_anneal = calculate_route_time(route_anneal, base, true_edges)
            
    print(f"  > Unique Energy States Sampled: {len(unique_energies)}")
    print(f"  > Lowest Energy Found: {best_energy_sa:.4f}")

    # ---------------------------------------------------------
    # METHOD B: TENSOR NETWORK DMRG (DETERMINISTIC)
    # ---------------------------------------------------------
    print("\n── METHOD B: TENSOR NETWORK (DMRG) ──")
    print("Behavior: Variational optimization. Sweeps deterministically across the")
    print("tensor chain, mathematically guaranteeing an energy descent each pass.")
    
    start_time = time.time()
    mpo_np = build_mpo_from_terms(terms, TOTAL_QUBITS)
    
    site = SpinHalfSite(conserve='None')
    sites = [site] * TOTAL_QUBITS
    model = QuboModel(sites, mpo_np_to_tenpy(mpo_np, sites))
    psi = MPS.from_product_state(sites, (['up', 'down'] * (TOTAL_QUBITS // 2 + 1))[:TOTAL_QUBITS], bc='finite')
    
    params = {'trunc_params': {'chi_max': 20, 'svd_min': 1e-10}, 'N_sweeps_check': 2, 'min_sweeps': 2, 'max_sweeps': 6, 'mixer': 'DensityMatrixMixer', 'mixer_params': {'amplitude': 1e-4, 'decay': 1.5, 'disable_after': 3}}
    eng = dmrg.TwoSiteDMRGEngine(psi, model, params)
    
    # Capture the sweep to show the deterministic descent
    E_dmrg, psi = eng.run()
    dmrg_time = time.time() - start_time

    probs = np.zeros(TOTAL_QUBITS)
    for q in range(psi.L): probs[q] = float((1 - psi.expectation_value_term([('Sigmaz', q)])) / 2)
    
    route_dmrg = []
    for k in range(P):
        for ii in range(N):
            if probs[ii + k * N] > 0.3: route_dmrg.append(candidates[ii])

    # Calculate Travel Time for Method B
    time_dmrg = calculate_route_time(route_dmrg, base, true_edges)

    print(f"  > Final Converged Energy: {E_dmrg:.4f}")

    # ---------------------------------------------------------
    # EXECUTIVE SUMMARY FOR THE SLIDE
    # ---------------------------------------------------------
    print("\n" + "=" * 75)
    print(" EXECUTIVE ALGORITHMIC COMPARISON")
    print("=" * 75)
    
    route_a_str = " -> ".join([PLACES[base][0]] + [PLACES[n][0] for n in route_anneal] + [PLACES[base][0]]) if route_anneal else "Empty Route"
    route_b_str = " -> ".join([PLACES[base][0]] + [PLACES[n][0] for n in route_dmrg] + [PLACES[base][0]]) if route_dmrg else "Empty Route"

    print("\n1. THE STOCHASTIC SOLVER (Simulated Annealing)")
    print(f"   ► Method       : Fluctuation-based global landscape sampling")
    print(f"   ► Lowest Energy: {best_energy_sa:.4f} (Found in {anneal_time:.2f}s)")
    print(f"   ► Best Route   : {route_a_str}")
    print(f"   ► Travel Time  : {time_anneal:.2f} hours")

    print("\n2. THE DETERMINISTIC SOLVER (DMRG Tensor Network)")
    print(f"   ► Method       : Variational localized energy minimization")
    print(f"   ► Lowest Energy: {E_dmrg:.4f} (Found in {dmrg_time:.2f}s)")
    print(f"   ► Best Route   : {route_b_str}")
    print(f"   ► Travel Time  : {time_dmrg:.2f} hours")

if __name__ == "__main__":
    main()