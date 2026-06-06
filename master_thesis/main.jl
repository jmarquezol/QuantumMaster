using ITensors, ITensorMPS
using Combinatorics, LinearAlgebra
using ITensorExpMPO
using ITensors: Algorithm

# note: to update the new version of ITensorExpMPO, go to the julia terminal, activate the env, and then run:
# dev ./ITensorExpMPOv2.jl/

abstract type AbstractAlcarazRecipe <: ExpHRecipe end
struct AlcarazWI  <: AbstractAlcarazRecipe end
struct AlcarazWII <: AbstractAlcarazRecipe end
struct AlcarazVD2 <: AbstractAlcarazRecipe end

_alg_string(::AlcarazWI)  = "WI"
_alg_string(::AlcarazWII) = "WII"
_alg_string(::AlcarazVD2) = "VD2"

# Mutable struct to hold params of Alcaraz model
Base.@kwdef mutable struct AlcarazParams <: ModelParams
    lambda::Float64 = 1.0
    p::Float64      = 0.0
    phys_site::Index{Int64} = Index(2, "S=1/2")
end

AlcarazParams(lambda::Number, p::Number) = AlcarazParams(; lambda=Float64(lambda), p=Float64(p))
AlcarazParams(x::AlcarazParams; lambda=x.lambda, p=x.p) = AlcarazParams(; lambda, p, phys_site=x.phys_site)


"""
Builds theAlcaraz Hamiltonian as an OpSum
"""
function alcaraz_opsum(N::Int, lambda::Number, p::Number)
    os = OpSum()

    # Nearest-Neighbor terms
    for j in 1:(N - 1)
        os += -1.0, "Z", j, "Z", j + 1
        os += -p * lambda, "X", j, "X", j + 1
    end

    # Next-Nearest-Neighbor terms
    for j in 1:(N - 2)
        os += -p, "Z", j, "Z", j + 2
    end

    # On-site Transverse Field
    for j in 1:N
        os += -lambda, "X", j
    end

    return os
end

"""
Builds the Alcaraz MPO using the automated OpSum-to-MPO Euler builder
Defaults to "VD2" but accepts "WI" or "WII" too
"""
function expH_alcaraz(sites::Vector{<:Index}, lambda::Number, p::Number; dt::Number, mpo_alg::String="VD2") 
    os = alcaraz_opsum(length(sites), lambda, p)
    tau = -im * dt  
    return expmpo(os, sites, tau; alg=Algorithm(mpo_alg)) 
end


function ITransverse.expH(sites::Vector{<:Index}, mp::AlcarazParams, recipe::AbstractAlcarazRecipe; dt::Number)
    os = alcaraz_opsum(length(sites), mp.lambda, mp.p)
    return expmpo(os, sites, -im * dt; alg=Algorithm(_alg_string(recipe)))
end

####################################################################################################################################
# TRICRITICAL MODEL

Base.@kwdef mutable struct TricriticalParams <: ModelParams
    lambda::Float64 = 0.0
    phys_site::Index{Int64} = Index(2, "S=1/2")
end

TricriticalParams(lambda::Number) = TricriticalParams(; lambda=Float64(lambda))
TricriticalParams(x::TricriticalParams; lambda=x.lambda) = TricriticalParams(; lambda, phys_site=x.phys_site)

# Recipe Hierarchy for Multiple Dispatch
abstract type AbstractTricriticalRecipe <: ExpHRecipe end
struct TricriticalWI  <: AbstractTricriticalRecipe end
struct TricriticalWII <: AbstractTricriticalRecipe end
struct TricriticalVD2 <: AbstractTricriticalRecipe end

_alg_string(::TricriticalWI)  = "WI"
_alg_string(::TricriticalWII) = "WII"
_alg_string(::TricriticalVD2) = "VD2"

"""
Builds the Tricritical Ising Hamiltonian with 3-body terms as an OpSum
"""
function tricritical_opsum(N::Int, lambda::Number)
    os = OpSum()

    # On-site terms (Transverse Field)
    for j in 1:N
        os += -1.0, "X", j
    end

    # NN terms
    for j in 1:(N - 1)
        os += -1.0, "Z", j, "Z", j + 1
    end

    # 3-body terms
    for j in 1:(N - 2)
        os += lambda, "Z", j, "Z", j + 1, "X", j + 2
        os += lambda, "X", j, "Z", j + 1, "Z", j + 2
    end

    return os
end

"""
Builds the Tricritical MPO using the automated OpSum-to-MPO Euler builder
"""
function expH_tricritical(sites::Vector{<:Index}, lambda::Number; dt::Number, mpo_alg::String="VD2") 
    os = tricritical_opsum(length(sites), lambda)
    tau = -im * dt  
    return expmpo(os, sites, tau; alg=Algorithm(mpo_alg)) 
end

function ITransverse.expH(sites::Vector{<:Index}, mp::TricriticalParams, recipe::AbstractTricriticalRecipe; dt::Number)
    os = tricritical_opsum(length(sites), mp.lambda)
    return expmpo(os, sites, -im * dt; alg=Algorithm(_alg_string(recipe)))
end




function compute_tricritical_entropies(
    mpo_generator::Function, target_T::Float64;
    lambda::Float64=1.0,
    dt::Float64=0.1, cutoff::Float64=1e-12, maxdim::Int=256, 
    alg::String="RTM", eps_converged::Float64=1e-6, nbeta::Int=4,
    MPO_alg::String="VD2"
)
    Ntime_steps = round(Int, target_T / dt)
    Nsteps = Ntime_steps + nbeta
    dbeta = -im * dt

    s = Index(2, "S=1/2")
    # Initial state |Psi0>
    init_state = complex(state(s, "X+"))  # |X+> state in Z basis

    RECIPES = Dict(
        "WI"  => TricriticalWI(),
        "WII" => TricriticalWII(),
        "VD2" => TricriticalVD2()
    )
    if !haskey(RECIPES, MPO_alg)
        error("Fatal: Algorithm '$MPO_alg' is not supported. Use WI, WII, or VD2.")
    end
    
    # Build tMPO blocks
    mp_tricritical = TricriticalParams(lambda=lambda, phys_site=s)
    tp = tMPOParams(
        mp=mp_tricritical; 
        dt=dt, 
        nbeta=nbeta, 
        scheme=RECIPES[MPO_alg],
        dbeta=dbeta, 
        bl=init_state
    )
    b = FwtMPOBlocks(tp)
    
    # Power Method Params
    pm_params = PMParams(;
        truncp = (; cutoff=cutoff, maxdim=maxdim, alg=alg), 
        opt_method = :nosym, 
        cutoffs = [cutoff], 
        maxdims = 2:2:maxdim, 
        itermax = 5000, 
        eps_converged = eps_converged, 
        normalization = "overlap",
        stuck_after = 200,
        compute_fidelity = true
    )

    # Dynamically determine spatial bond dimension
    dummy_sites = siteinds("S=1/2", 4) 
    sample_mpo = mpo_generator(dummy_sites, lambda; dt=dt, mpo_alg=MPO_alg)
    spatial_bond_dim = dim(linkind(sample_mpo, 1))

    # The virtual links of the spatial MPO become the physical sites of the temporal MPS
    time_sites  = addtags(siteinds(spatial_bond_dim, Nsteps; conserve_qns=false), "time")

    # Build the transverse Transfer Matrix
    mpo         = fw_tMPO(b, time_sites, tr=init_state)
    # Temporal boundary (random so that it overlaps with the dominant eigenvector)
    start_mps   = fw_tMPS(b, time_sites; tr=init_state, LR=:right)
    for i in eachindex(start_mps)
        tensor_inds = inds(start_mps[i])
        
        start_mps[i] = randomITensor(ComplexF64, tensor_inds)
    end
    normalize!(start_mps)

    # Run Power Method
    psi_L, psi_R, pm_info = ITransverse.powermethod_lr(start_mps, mpo, mpo, pm_params)

    # Normalize just in case
    norm = overlap_noconj(psi_L, psi_R)
    psi_L ./= sqrt(norm)
    psi_R ./= sqrt(norm)

    # Calculate entropy profiles (real & imaginary)
    p_r2_real = real.(ITransverse.gen_renyi2(psi_L, psi_R))
    p_r2_imag = imag.(ITransverse.gen_renyi2(psi_L, psi_R))

    bonds = 1:length(p_r2_real)

    return bonds, p_r2_real, p_r2_imag, psi_L, psi_R, mpo
end


####################################################################################################################################
Base.@kwdef mutable struct BenchmarkParams <: ModelParams
    lambda::Float64 = 0.0
    p::Float64 = 0.0
    phys_site::Index{Int64} = Index(2, "S=1/2")
end

BenchmarkParams(lambda::Number, p::Number) = BenchmarkParams(; lambda=Float64(lambda), p=Float64(p))
BenchmarkParams(x::BenchmarkParams; lambda=x.lambda, p=x.p) = BenchmarkParams(; lambda, p, phys_site=x.phys_site)

# Recipe Hierarchy for Multiple Dispatch
abstract type AbstractBenchmarkRecipe <: ExpHRecipe end
struct BenchmarkWI  <: AbstractBenchmarkRecipe end
struct BenchmarkWII <: AbstractBenchmarkRecipe end
struct BenchmarkVD2 <: AbstractBenchmarkRecipe end

_alg_string(::BenchmarkWI)  = "WI"
_alg_string(::BenchmarkWII) = "WII"
_alg_string(::BenchmarkVD2) = "VD2"

"""
Builds a custom Ising Hamiltonian as an OpSum
"""
function benchmark_opsum(N::Int, lambda::Number, p::Number)
    os = OpSum()

    # Nearest-Neighbor terms
    for j in 1:(N - 1)
        os += -1.0, "Z", j, "Z", j + 1
        # os += -p * lambda, "X", j, "X", j + 1
    end

    # Next-Nearest-Neighbor terms
    for j in 1:(N - 2)
        os += -p, "Z", j, "Z", j + 2
    end

    # On-site Transverse Field
    for j in 1:N
        os += -lambda, "X", j
    end

    return os
end


function expH_benchmark(sites::Vector{<:Index}, lambda::Number, p::Number; dt::Number, mpo_alg::String="VD2") 
    os = benchmark_opsum(length(sites), lambda, p)
    tau = -im * dt  
    return expmpo(os, sites, tau; alg=Algorithm(mpo_alg)) 
end

function ITransverse.expH(sites::Vector{<:Index}, mp::BenchmarkParams, recipe::AbstractBenchmarkRecipe; dt::Number)
    os = benchmark_opsum(length(sites), mp.lambda, mp.p)
    return expmpo(os, sites, -im * dt; alg=Algorithm(_alg_string(recipe)))
end



####################################################################################################################################

# HELPER FUNCTIONS

""" 
Runs the Power Method alg to contract the 2D network encoding the dynamics of the system
"""
function compute_alcaraz_entropies(
    mpo_generator::Function, target_T::Float64;
    p::Float64=0.0, lambda::Float64=1.0,
    dt::Float64=0.1, cutoff::Float64=1e-12, maxdim::Int=256, 
    alg::String="RTM", eps_converged::Float64=1e-6, nbeta::Int=4,
    MPO_alg::String="VD2"
)
    Ntime_steps = round(Int, target_T / dt)
    Nsteps = Ntime_steps + nbeta
    dbeta = -im * dt

    s = Index(2, "S=1/2")
    # Initial state |Psi0>
    init_state = complex(state(s, "X+"))  # |X+> state in Z basis

    RECIPES = Dict(
        "WI"  => AlcarazWI(),
        "WII" => AlcarazWII(),
        "VD2" => AlcarazVD2()
    )
    if !haskey(RECIPES, MPO_alg)
        error("Fatal: Algorithm '$MPO_alg' is not supported. Use WI, WII, or VD2.")
    end
    
    # Build tMPO blocks
    mp_alcaraz = AlcarazParams(lambda=lambda, p=p, phys_site=s)
    tp = tMPOParams(
        mp=mp_alcaraz; 
        dt=dt, 
        nbeta=nbeta, 
        scheme=RECIPES[MPO_alg],
        dbeta=dbeta, 
        bl=init_state
    )
    b = FwtMPOBlocks(tp)
    
    # Power Method Params
    pm_params = PMParams(;
        truncp = (; cutoff=cutoff, maxdim=maxdim, alg=alg), 
        opt_method = :nosym, 
        cutoffs = [cutoff], 
        maxdims = 2:2:maxdim, 
        itermax = 5000, 
        eps_converged = eps_converged, 
        normalization = "overlap",
        stuck_after = 200,
        compute_fidelity = true
    )

    # Dynamically determine spatial bond dimension
    dummy_sites = siteinds("S=1/2", 4) 
    sample_mpo = mpo_generator(dummy_sites, lambda, p; dt=dt, mpo_alg=MPO_alg)
    spatial_bond_dim = dim(linkind(sample_mpo, 1))

    # The virtual links of the spatial MPO become the physical sites of the temporal MPS
    time_sites  = addtags(siteinds(spatial_bond_dim, Nsteps; conserve_qns=false), "time")

    # Build the transverse Transfer Matrix
    mpo         = fw_tMPO(b, time_sites, tr=init_state)
    # Temporal boundary (random so that it overlaps with the dominant eigenvector)
    start_mps   = fw_tMPS(b, time_sites; tr=init_state, LR=:right)
    for i in eachindex(start_mps)
        tensor_inds = inds(start_mps[i])
        
        start_mps[i] = randomITensor(ComplexF64, tensor_inds)
    end
    normalize!(start_mps)

    # Run Power Method
    psi_L, psi_R, pm_info = ITransverse.powermethod_lr(start_mps, mpo, mpo, pm_params)

    # Normalize just in case
    norm = overlap_noconj(psi_L, psi_R)
    psi_L ./= sqrt(norm)
    psi_R ./= sqrt(norm)

    # Calculate entropy profiles (real & imaginary)
    p_r2_real = real.(ITransverse.gen_renyi2(psi_L, psi_R))
    p_r2_imag = imag.(ITransverse.gen_renyi2(psi_L, psi_R))

    bonds = 1:length(p_r2_real)

    return bonds, p_r2_real, p_r2_imag, psi_L, psi_R, mpo
end




""" 
Plot the entropy profiles for a range of target times using the compute_alcaraz_entropies function
"""
function plot_entropy_profiles(
    mpo_generator::Function, 
    target_times::Vector{Float64};
    p::Float64=0.0, lambda::Float64=1.0,
    dt::Float64=0.1, cutoff::Float64=1e-12, maxdim::Int=256, 
    alg::String="RTM", eps_converged::Float64=1e-6, nbeta::Int=4,
    MPO_alg::String="VD2"
)
    plt_real = plot(title="Re(S₂) Entropy Profiles", xlabel="Temporal Cut (t/T)", ylabel="Re(S₂)", 
                    legend=:outerright, grid=true, framestyle=:box)
    plt_imag = plot(title="Im(S₂) Entropy Profiles", xlabel="Temporal Cut (t/T)", ylabel="Im(S₂)", 
                    legend=:outerright, grid=true, framestyle=:box)

    n_times = length(target_times)
    colors_real = cgrad(:viridis, n_times, categorical=true)
    colors_imag = cgrad(:plasma, n_times, categorical=true)

    @showprogress "Computing Profiles (alg=$MPO_alg, p=$p)..." for (i, T) in enumerate(target_times)
        bonds, r2_re, r2_im, _ = compute_alcaraz_entropies(
            mpo_generator, T; 
            lambda=lambda, p=p, dt=dt, cutoff=cutoff, maxdim=maxdim, 
            alg=alg, eps_converged=eps_converged, nbeta=nbeta,
            MPO_alg=MPO_alg 
        )

        # Normalize x axis (Temporal Cut t/T)
        n_bonds = length(r2_re)
        x_normalized = range(0.0, 1.0, length=n_bonds)
        
        time_label = "t = $(round(T, digits=1))"
        
        plot!(plt_real, x_normalized, r2_re, label=time_label, lw=2, color=colors_real[i])
        plot!(plt_imag, x_normalized, r2_im, label=time_label, lw=2, color=colors_imag[i])
    end

    # 1x2 layout with traceability in the title
    final_plot = plot(
        plt_real, plt_imag, 
        layout=(1, 2), 
        size=(1200, 450), 
        margin=5Plots.mm,
        plot_title="Alcaraz Temporal Entanglement (p=$p, λ=$lambda) | Alg: $MPO_alg"
    )
    
    return final_plot
end