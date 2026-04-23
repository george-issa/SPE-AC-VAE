# Hubbard Model Simulation Plan

Square lattice, L=4, t=1, μ=0 (half-filling), Δτ=0.05, 8 MPI walkers.

## Completed

| U   | β  | Regime              |
|-----|----|---------------------|
| 2.0 | 6  | weak coupling metal |

## Phase 1 — Mott crossover at fixed T (β=6)

Scan U at the same temperature as the existing run to trace the full metal → Mott insulator evolution.

| U   | β  | Regime                  | Notes                                 |
|-----|----|-------------------------|---------------------------------------|
| 4.0 | 6  | correlated Fermi liquid | renormalized QP peak, Z < 1           |
| 6.0 | 6  | pseudogap onset         | partial gap at antinodal k-point      |
| 8.0 | 6  | Mott insulator          | split Hubbard bands, clear gap ~U     |

```julia
const US    = [4.0, 6.0, 8.0]
const MUS   = [0.0]
const BETAS = [6.0]
```

## Phase 2 — Temperature dependence in pseudogap regime (U=6)

Fix U=6 and lower T to watch the pseudogap deepen. Hardest regime for MaxEnt.

| U   | β  | Regime                  | Notes                                 |
|-----|----|-------------------------|---------------------------------------|
| 6.0 | 8  | pseudogap, moderate T   | gap more pronounced than β=6          |
| 6.0 | 10 | pseudogap, low T        | sharpest features — best VAE benchmark|

```julia
const US    = [6.0]
const MUS   = [0.0]
const BETAS = [8.0, 10.0]
```

## Runtime scaling

L_τ = β / Δτ, so runtime scales linearly with β.

| β  | L_τ | Relative cost |
|----|-----|---------------|
| 6  | 120 | 1.0×          |
| 8  | 160 | 1.33×         |
| 10 | 200 | 1.67×         |

No sign problem at μ=0 (ph_sym_form=true), so low temperatures are not restricted.

## Priority order

1. U=4, β=6 — extend existing correlated metal data
2. U=6, β=6 — pseudogap onset
3. U=8, β=6 — Mott insulator
4. U=6, β=10 — deepest pseudogap, primary benchmark target
5. U=6, β=8 — optional intermediate temperature
