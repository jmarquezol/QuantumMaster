# Long-time dynamics of an ANNNI-type chain via transverse contraction

Master's thesis code — **Joaquín G. Márquez Olguín**, supervised by **Stefano Carignano** (Barcelona Supercomputing Center).

---

## What this project is about

When a closed quantum system is driven out of equilibrium by a sudden quench, its
entanglement typically grows linearly in time. Representing the state as a Matrix
Product State (MPS) then forces an exponentially growing bond dimension — the
**entanglement barrier** that bottlenecks every standard time-evolution method
(TEBD, TDVP) at late times.

**Transverse contraction** sidesteps the barrier. Instead of evolving the state
forward in the Schrödinger picture, the full dynamics of the 1D chain is encoded as
a 2D tensor network in which *time plays the role of a second spatial dimension*.
Contracting that network along the original spatial direction turns time evolution
into the repeated application of a single, fixed **spatial transfer matrix** — an
eigenproblem solved with a tensor-network power method. The boundary objects are
*temporal* MPS, and the cost of the contraction is set by their **temporal
entanglement entropy**.

The key physical fact is that this cost is dictated by *criticality*, not
integrability. For a quench **to** a conformally invariant critical point, the
generalized temporal entropies grow only **logarithmically** — the transfer matrix
becomes effectively unitary (*emergent dual unitarity*), and universal CFT data such
as the central charge `c` can be read off the Loschmidt echo. Carignano & Tagliacozzo
(arXiv:2405.14706) showed this explicitly for the integrable critical Ising and Potts
chains.

**Central question of this thesis:** does emergent dual unitarity — and with it the
logarithmic scaling of the temporal entropies — *survive when integrability is
broken*? We test it on an **ANNNI-type (Alcaraz) model**: a self-dual spin chain that
extends the transverse-field Ising model with next-nearest-neighbour (NNN) couplings
of strength `p`,

```
H = -Σ_i [ σᶻ_i σᶻ_{i+1}  +  p σᶻ_i σᶻ_{i+2}  +  λ σˣ_i  +  p λ σˣ_i σˣ_{i+1} ].
```

The model is non-integrable for any `p>0` yet (confirmed here by DMRG) stays critical
and in the **Ising universality class** `c≈1/2` over a wide range of `p`. It is a
deliberate **stress-test**: its longer-range couplings are exactly what makes it
expensive for conventional methods, while the CFT prediction still has a well-defined
target. We quench from the polarized paramagnet `|Ψ₀⟩ = |X+⟩^N` to the critical point
`λ=1`, compute the Loschmidt echo by transverse contraction, and compare the
generalized temporal entropies against the Ising CFT prediction.

---

## Repository structure

```
README.md                     this file
CLAUDE.md                     technical context for AI agents (highly detailed)
carignano-tagliacozzo.md      primary physics reference (arXiv:2405.14706), in markdown
Project.toml / Manifest.toml  Julia environment

src/                          consolidated library (include "src/thesislib.jl")
  thesislib.jl                  entry point: loads packages + both files below
  models.jl                     Alcaraz (ANNNI) + tricritical Hamiltonians, OpSums,
                                exp-MPO wrappers, ITransverse.expH dispatch hooks
  transverse_tools.jl           build the temporal MPO, (block) power method,
                                generalized temporal entropies, TDVP cross-check,
                                crash-safe sweeps and plotting

Notebooks — the narrative, in order:
  1_introduction_model.ipynb     naive Trotter TEBD benchmarked vs TDVP ⟨Z⟩ (mismatch);
                                 TDVP Loschmidt rate + the entanglement barrier
  2_mpo_exponentiation.ipynb     the finite-state-machine exp-MPO: WI / WII / VD2 benchmarks
  3_temporal_entropies.ipynb     generalized temporal entropies; single-vector PM convergence
                                 failure at the gap closing → block PM recovers the dome
  3.5_block_pm_efficiency.ipynb  making the block power method cheaper — profile + maxdim/ramp/
                                 cutoff/kernel/warm-start/k benchmarks
  4_cft_ground_state.ipynb       DMRG: c(p) sweep, finite-size scaling, chord fit — c ≈ 1/2
  5_spectral_gap_degeneracy.ipynb  deflation → block PM; gap closes faster at larger p; DQPTs
  6_loschmidt_ising.ipynb        reproduce Carignano–Tagliacozzo (c=1/2); λ₀(T) circle plot
  7_loschmidt_alcaraz.ipynb      Alcaraz temporal c via block PM — entropy slope + leading
                                 eigenvalues (Eq.3/Eq.4/circle), p=0 calibration (outcome pending run)
  TN_assignment.ipynb            course exercise (kept, unrelated to the thesis)

results/
  imgs/                         figures used by the notebooks / thesis
  data/                         cached .jld2 (power-method sweeps, TDVP runs, rate benchmarks)

legacy/                         superseded notebooks, old helper scripts, and obsolete data
                                from closed investigations (kept for provenance; safe to purge)

ITensorExpMPOv2.jl/             exp-MPO package (fork — see credits below)
ITransverse_source/             read-only reference clone of ITransverse.jl (see version note)
```

---

## Running the code

The notebooks assume the project environment is active:

```bash
julia --project=.
```

```julia
include("src/thesislib.jl")   # loads ITensors, ITransverse, ITensorExpMPO + the library
```

From there the high-level drivers are available directly, e.g.

```julia
mp  = AlcarazParams(lambda=1.0, p=0.1)
res = compute_entropies(mp, 4.0; scheme=AlcarazVD2(), nbeta=4)   # Rényi-2 temporal entropy
mpo, scaffold = build_alcaraz_tmpo(4.0; p=0.1, nbeta=4)          # tMPO for the block power method
theta, L, R, info = block_transfer_eigs(mpo, scaffold; k=4)      # leading transfer-matrix eigenvalues
```

Heavy runs (block power method near the gap closing, long-T TDVP) cache to
`results/data/` via `crashsafe_sweep` / `tdvp_loschmidt_amplitude`, so an interrupted
sweep resumes where it stopped.

---

## Packages and credits

- **[ITransverse.jl](https://github.com/starsfordummies/ITransverse.jl)** — the transverse
  contraction library (power method, RTM truncation, temporal entropies), developed by
  **Stefano Carignano** (BSC). All transverse algorithms used here are from this package.

  > **Version note.** The notebooks run against the *installed* (older) version of
  > ITransverse; `ITransverse_source/` is a *newer* clone kept only as a read-only API
  > reference. Some signatures differ between the two (e.g. `tMPOParams` is keyword-only
  > with `bl=` for the initial state; the asymmetric power method is `powermethod_lr`).
  > When in doubt, the installed version is authoritative — `CLAUDE.md` §6,7,9,10 lists
  > the known discrepancies. Verify against `ITransverse_source/src/` before relying on
  > any signature.

- **ITensorExpMPO** (`ITensorExpMPOv2.jl/`) — builds the time-evolution operator
  `exp(-iHdt)` directly as an MPO from an `OpSum`, via a finite-state-machine + Euler-style
  builder. **This directory is a fork of [tipfom/ITensorExpMPO.jl](https://github.com/tipfom/ITensorExpMPO.jl);
  all of its code is upstream work by [@tipfom](https://github.com/tipfom).** The only original contribution for this thesis is the
  **VD2 second-order kernel** (`makeW(::Algorithm"VD2", …)` in `src/eulerbuilder.jl`),
  the Van Damme *et al.* (SciPost Phys. **17**, 135 (2024)) Appendix-A second-order MPO,
  added so the NNN Alcaraz model evolves with genuine 2nd-order accuracy.

- **[ITensors.jl / ITensorMPS.jl](https://github.com/ITensor/ITensors.jl)** — the
  underlying tensor-network library.

---

## Key references

- Carignano & Tagliacozzo, *Loschmidt echo, emerging dual unitarity and scaling of
  generalized temporal entropies after quenches to the critical point*, arXiv:2405.14706
  (the primary physics reference; included as `carignano-tagliacozzo.md`).
- Carignano, *The ITransverse.jl library for transverse tensor network contractions*,
  arXiv:2509.03699.
- Van Damme, Haegeman, McCulloch, Vanderstraeten, *Efficient higher-order matrix product
  operators for time evolution*, SciPost Phys. **17**, 135 (2024) — the VD2 construction.
- Zaletel, Mong, Karrasch, Moore, Pollmann, PRB **91**, 165112 (2015) — the WI/WII exp-MPO
  constructors.
- Alcaraz — the original ANNNI-type model and its Ising-class finite-size scaling.
