# ITensorExpMPO.jl

> **Fork notice (Master-thesis use).** This directory is a fork of
> [`tipfom/ITensorExpMPO.jl`](https://github.com/tipfom/ITensorExpMPO.jl).
> **All code here is upstream work by [@tipfom](https://github.com/tipfom)**, with a
> single addition for this thesis: the
> **VD2 second-order kernel** — `makeW(::Algorithm"VD2", …)` in
> [`src/eulerbuilder.jl`](src/eulerbuilder.jl) — which implements the Appendix-A
> second-order MPO of Van Damme, Haegeman, McCulloch & Vanderstraeten,
> *SciPost Phys.* **17**, 135 (2024), so the next-nearest-neighbour ANNNI/Alcaraz
> model evolves with genuine 2nd-order accuracy. Credit for everything else belongs to
> the upstream authors.

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ITensor.github.io/ITensorExpMPO.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ITensor.github.io/ITensorExpMPO.jl/dev/)
[![Build Status](https://github.com/ITensor/ITensorExpMPO.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ITensor/ITensorExpMPO.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ITensor/ITensorExpMPO.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ITensor/ITensorExpMPO.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

## Installation instructions

This package resides in the `ITensor/ITensorRegistry` local registry.
In order to install, simply add that registry through your package manager.
This step is only required once.
```julia
julia> using Pkg: Pkg

julia> Pkg.Registry.add(url="https://github.com/ITensor/ITensorRegistry")
```
or:
```julia
julia> Pkg.Registry.add(url="git@github.com:ITensor/ITensorRegistry.git")
```
if you want to use SSH credentials, which can make it so you don't have to enter your Github ursername and password when registering packages.

Then, the package can be added as usual through the package manager:

```julia
julia> Pkg.add("ITensorExpMPO")
```

## Examples

````julia
using ITensorExpMPO: ITensorExpMPO
````

Examples go here.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

