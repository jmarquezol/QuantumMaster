# FINDINGS — TNMH mixing-time study

Auto-appended by the phase notebooks (`log_finding`). Each phase writes a one-paragraph result here.

## Phase 0 — Infrastructure
- Control A: Σq = 1.0 (=1).
- Control B: C(3×4, D=4 exact) = 1.0 (→1); C(D=2)=1.001286.
- Pinned log q matches the sampler to <1e-10; Julia↔Python agree ≤1e-8 (see crosscheck CSVs).

## Phase 1 — Spectrum

**Liu's identity λ₂ = 1 − 1/C — CONFIRMED (max|Δ| = 4.24e-14), with a caveat.**
- Of the 5 table cases, only the two **3×4** ones actually truncate:
  C(D=2)=1.0012858 → λ₂=0.0012842, C(D=3)=1.0001426 → λ₂=0.0001426 — exact to ~6e-15.
- The other three (2×5, 2×6, 4×3 at D=2) have **C = 1.0 exactly** (proposal already
  lossless at that geometry ⇒ λ₂ = 0). They confirm the identity only trivially (0=0);
  the real verification rests on the two 3×4 points. ⇒ at enumerable sizes, D=2 is
  exact for short/narrow strips; truncation only bites once both Lx and Ly grow.
- Net: the relaxation time is **exactly** t_rel = C (Liu 1996), verified in-repo.

**TV-decay tightness — the bound IS matched; the figure is misleading. Read this.**
- The saved `tv[n]` matches `(1−1/C)ⁿ` to full precision for **n = 1..4**:
  tv = [1.284e-3, 1.649e-6, 2.118e-9, 2.713e-12], bound = [1.284e-3, 1.649e-6,
  2.118e-9, 2.720e-12]. The per-step ratio is exactly λ₂ = 1−1/C.
- At **n ≥ 5** the empirical curve flattens at ~1e-13 (slowly rising to 2.6e-13).
  This plateau is the **double-precision roundoff floor** of the dense `Pⁿ`
  mat-vec, NOT a breakdown of the bound: with C = 1.0013 ⇒ λ₂ = 0.0013, each step
  removes 99.87% of the deviation, so the chain reaches π to machine precision in
  **~5 steps**. The dashed `(1−1/C)ⁿ` line keeps descending to ~1e-228 because it
  is an analytic formula with no floor.
- **Why `phase1_tvdecay.png` looks like a failure:** the y-axis is auto-stretched
  to 10⁻²⁰⁰ by the analytic line, so the empirical floor at 1e-13 sits visually
  near the top and reads as a flat "plateau near 0.5" — it is actually ~1e-13, i.e.
  fully converged. The bound is confirmed everywhere it is numerically observable.
- **Limitation / TODO:** at enumerable sizes (Lx·Ly ≤ 12) with D=2 at β_c, C stays
  so close to 1 that the decay is over in a handful of steps — we cannot *display*
  an extended geometric line. To get a visually convincing long `(1−1/C)ⁿ` decay we
  need a case with C substantially > 1 (stronger truncation / wider strip), which
  fights the 2^N enumeration ceiling. Options: (i) plot only n ≤ 6 (above the
  floor); (ii) use higher-precision `P` (BigFloat) to push the floor down; (iii)
  pick the most-truncated enumerable geometry to maximise C. The current PNG should
  be regenerated with one of these before using it in the writeup.
