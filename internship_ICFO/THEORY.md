# THEORY — bounding the mixing time

This is the analytical side of the project: turning the numerical observation
"`ρ − 1` tracks the truncation error" into a derivation. See [README.md](README.md)
for the numerical results and how to run things.

---

## The goal

The mixing time of the IMH chain is set by `C = max_x π(x)/q(x)` (exactly, via Liu:
`λ₂ = 1 − 1/C`). The truncation gives us natural control of an **additive** error (a
TV-like distance between the exact and truncated boundary environment, call it `ε_D`).
But `C` is a **multiplicative**, worst-case quantity, and a small additive error does
*not* bound it. The whole problem is to bridge the two.

The bridge is the chain rule. Because the proposal samples the lattice **row by row**,

> `log C = max_x Σ_x δ_x`,  with  `δ_x = log[ π(row_x | rows above) / q(row_x | rows above) ]`.

`log C` is an **additive accumulation** of per-row terms. If each `δ_x` is bounded by
the truncation error at that row, then `log C ≤ Lx · (bound)` and we get a mixing-time
bound from truncation errors alone — no enumeration, no need to find the worst config.
So everything reduces to **bounding one per-row term `δ_x`**.

A structural simplification makes this one-sided: in the top-down sweep, the rows
*above* the current one are **pinned** — a classical product state, rank 1 — so
truncating them loses nothing. **All** the error comes from truncating the **traced
half-plane below**. We only ever truncate one object.

---

## The per-row ratio, exactly

Fix one row. Contracting it (pinned to a spin pattern `s`) against the environment
gives an unnormalized weight `a(s) = ⟨v(s)|β⟩` (exact) or `a_D(s) = ⟨v(s)|β_D⟩`
(truncated), where `β` is the bottom environment and `β_D = β − δ` its truncation,
`‖δ‖ = ε_D‖β‖`. The object that enters the importance weight is the **normalized
conditional ratio** `r(s) = p(s)/p_D(s)`.

Define the **relative truncation effect** on each config, `g(s) = a_D(s)/a(s)`
(`g = 1` means untouched). One line of algebra (`N_D/N = Σ_s p(s) g(s) = E_p[g]`) gives

> **`r(s) = E_p[g] / g(s)`,  so  `ρ = E_p[g] / min_s g(s)`.**   *(exact)*

Verified numerically to 1e-16. The physics is in the shape of this formula:

> **If truncation hits every config equally (`g = const`), then `ρ = 1`.**
> A uniform error is invisible — it cancels in the normalized conditional. **Only the
> *variation* of the truncation across configurations creates a mixing penalty.**

To first order (`g = E_p[g](1+u)`, `E_p[u]=0`):

> `ρ − 1 ≈ (E_p[g] − min_s g) / E_p[g]` = the **relative spread of `g`**.

**Confirmed:** across 30 regimes `(Ly, D, β)` spanning `ρ − 1` from `1e-5` to `0.13`,
the first-order prediction matches the exact `ρ − 1` to **0.3% median** (figure
`results/action4_spread.png`).

---

## A bound, and where it's loose

Writing `g(s) = 1 − η(s)` with `η(s) = ⟨v(s)|δ⟩/⟨v(s)|β⟩`, Cauchy–Schwarz gives the
crude bound `|η(s)| ≤ ε_D / cos∠(v(s), β)` — the dangerous configs are those misaligned
with the dominant environment. It can be tightened: since `δ` lives in the **discarded
Schmidt subspace**, only the part of `v(s)` leaking into that subspace counts:

> `|η(s)| ≤ ε_D · ‖P_disc v(s)‖ · ‖β‖ / ⟨v(s)|β⟩`.

**Verified** (clean single-truncation test, `Ly ≤ 8`): the tightened bound holds in
every case. **Two honest caveats** the numerics forced:

1. The tightening helps **only 1.2–2.9×**, not the large factor first conjectured —
   the discarded subspace is high-dimensional, so projecting onto it removes little of
   a generic `v(s)`. The dominant remaining slack is plain Cauchy–Schwarz, and the
   worst config is **not** simply the rarest one.
2. The naive "gap closes at `β_c` ⇒ `ρ` peaks there" is too simple. The boundary
   entanglement `S` rises *through* `β_c` and **saturates to `log 2`** (the ferromagnetic
   cat state) deep in the ordered phase, so `ε_D` (and `ρ − 1`) peak just *above* `β_c`
   and `ρ` rises monotonically with `β`. What survives cleanly: **`ρ − 1` tracks `ε_D`**
   (corr 0.66), and `ε_D` measures how far the entanglement spectrum spills past `D`.
   The critical signature lives in the `Ly`-scaling, not the `β`-profile.
   (Figure `results/action2_gap.png`.)

---

## Status & what's open

**Done and verified (per-row, Step A):**
- exact closed form `ρ = E_p[g]/min g`;
- `ρ − 1 = ` relative spread of `g` (0.3% over 30 regimes);
- tightened bound `|η| ≤ ε_D ‖P_disc v‖‖β‖/⟨v|β⟩` holds;
- `ε_D` set by boundary entanglement vs `D`.

**Still open:**
- A *tight* a-priori constant. The current bound is a few× above the truth (Cauchy–
  Schwarz slack), which has no obvious a-priori control. The "good-enough" bound
  `ρ ≤ 1 + O(ε_D)` ⇒ `C ≤ (1 + O(ε_D))^{Lx}` is in hand — which was the stated goal.
- A clean bound on `ε_D` itself from `D` and the entanglement-spectrum decay.
- **Step B: lift the per-row bound to the lattice.** `log C = Σ_x δ_x` is literally the
  accumulation curve (`accumulation_curve` in the tools produces the `δ_x` directly);
  summing the per-row bound finishes the program.

---

## How a result here gets made (the working method)

For reference, every result above came from the same loop, and the next ones will too:
**reduce to the minimal case → write the target as an explicit formula → expand in the
small parameter (`ε_D`) → look for the cancellation (here: uniform error cancels) →
bound the dangerous term (watch the denominators — that's where criticality enters) →
check against the numerical oracle at every step.** The two caveats above are that last
step working: plausible-looking claims that the numbers corrected.
