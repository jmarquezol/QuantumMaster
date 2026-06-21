# ══════════════════════════════════════════════════════════════════════════════
# ALCARAZ (ANNNI-type) MODEL
#   H = -Σ_i [ Z_i Z_{i+1} + p λ X_i X_{i+1} + p Z_i Z_{i+2} + λ X_i ]
#   self-dual; p=0 recovers the integrable TFIM. Field on sigma_x, coupling on sigma_z.
# ══════════════════════════════════════════════════════════════════════════════

abstract type AbstractAlcarazRecipe <: ExpHRecipe end
struct AlcarazWI  <: AbstractAlcarazRecipe end
struct AlcarazWII <: AbstractAlcarazRecipe end
struct AlcarazVD2 <: AbstractAlcarazRecipe end

_alg_string(::AlcarazWI)  = "WI"
_alg_string(::AlcarazWII) = "WII"
_alg_string(::AlcarazVD2) = "VD2"

Base.@kwdef mutable struct AlcarazParams <: ModelParams
    lambda::Float64 = 1.0
    p::Float64      = 0.0
    phys_site::Index{Int64} = Index(2, "S=1/2")
end

AlcarazParams(lambda::Number, p::Number) = AlcarazParams(; lambda=Float64(lambda), p=Float64(p))
AlcarazParams(x::AlcarazParams; lambda=x.lambda, p=x.p) = AlcarazParams(; lambda, p, phys_site=x.phys_site)

"""Builds the Alcaraz (ANNNI-type) Hamiltonian as an OpSum."""
function alcaraz_opsum(N::Int, lambda::Number, p::Number)
    os = OpSum()
    for j in 1:(N - 1)                 # nearest-neighbour
        os += -1.0,        "Z", j, "Z", j + 1
        os += -p * lambda, "X", j, "X", j + 1
    end
    for j in 1:(N - 2)                 # next-nearest-neighbour
        os += -p, "Z", j, "Z", j + 2
    end
    for j in 1:N                       # transverse field
        os += -lambda, "X", j
    end
    return os
end

"""Direct U(dt)=exp(-i H dt) MPO for the Alcaraz model (Schrödinger pipeline). alg = {WI,WII,VD2}."""
function expH_alcaraz(sites::Vector{<:Index}, lambda::Number, p::Number; dt::Number, mpo_alg::String="VD2")
    os  = alcaraz_opsum(length(sites), lambda, p)
    return expmpo(os, sites, -im * dt; alg=Algorithm(mpo_alg))
end

# ITransverse hook: returns the spatial U(dt) MPO that ITransverse rotates into the tMPO.
function ITransverse.expH(sites::Vector{<:Index}, mp::AlcarazParams, recipe::AbstractAlcarazRecipe; dt::Number)
    os = alcaraz_opsum(length(sites), mp.lambda, mp.p)
    return expmpo(os, sites, -im * dt; alg=Algorithm(_alg_string(recipe)))
end

# ══════════════════════════════════════════════════════════════════════════════
# TRICRITICAL-ISING MODEL (optional variant)
#   H = -Σ X_i - Σ Z_i Z_{i+1} + λ Σ ( Z_i Z_{i+1} X_{i+2} + X_i Z_{i+1} Z_{i+2} )
# ══════════════════════════════════════════════════════════════════════════════

abstract type AbstractTricriticalRecipe <: ExpHRecipe end
struct TricriticalWI  <: AbstractTricriticalRecipe end
struct TricriticalWII <: AbstractTricriticalRecipe end
struct TricriticalVD2 <: AbstractTricriticalRecipe end

_alg_string(::TricriticalWI)  = "WI"
_alg_string(::TricriticalWII) = "WII"
_alg_string(::TricriticalVD2) = "VD2"

Base.@kwdef mutable struct TricriticalParams <: ModelParams
    lambda::Float64 = 0.0
    phys_site::Index{Int64} = Index(2, "S=1/2")
end

TricriticalParams(lambda::Number) = TricriticalParams(; lambda=Float64(lambda))
TricriticalParams(x::TricriticalParams; lambda=x.lambda) = TricriticalParams(; lambda, phys_site=x.phys_site)

"""Builds the tricritical-Ising Hamiltonian (with 3-body terms) as an OpSum."""
function tricritical_opsum(N::Int, lambda::Number)
    os = OpSum()
    for j in 1:N                       # transverse field
        os += -1.0, "X", j
    end
    for j in 1:(N - 1)                 # nearest-neighbour
        os += -1.0, "Z", j, "Z", j + 1
    end
    for j in 1:(N - 2)                 # 3-body terms
        os += lambda, "Z", j, "Z", j + 1, "X", j + 2
        os += lambda, "X", j, "Z", j + 1, "Z", j + 2
    end
    return os
end

"""Direct U(dt) MPO for the tricritical model (Schrödinger pipeline)."""
function expH_tricritical(sites::Vector{<:Index}, lambda::Number; dt::Number, mpo_alg::String="VD2")
    os  = tricritical_opsum(length(sites), lambda)
    return expmpo(os, sites, -im * dt; alg=Algorithm(mpo_alg))
end

function ITransverse.expH(sites::Vector{<:Index}, mp::TricriticalParams, recipe::AbstractTricriticalRecipe; dt::Number)
    os = tricritical_opsum(length(sites), mp.lambda)
    return expmpo(os, sites, -im * dt; alg=Algorithm(_alg_string(recipe)))
end
