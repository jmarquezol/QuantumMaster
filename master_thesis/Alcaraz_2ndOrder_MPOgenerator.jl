using ITensors, ITensorMPS
using Combinatorics, LinearAlgebra
using ITensorExpMPO
using ITensors: Algorithm


struct AlcarazWII <: ExpHRecipe end

Base.@kwdef mutable struct AlcarazParams <: ModelParams
    lambda::Float64 = 1.0
    p::Float64      = 0.0
    phys_site::Index{Int64} = Index(2, "S=1/2")
end

AlcarazParams(lambda::Number, p::Number) = AlcarazParams(; lambda=Float64(lambda), p=Float64(p))
AlcarazParams(x::AlcarazParams; lambda=x.lambda, p=x.p) = AlcarazParams(; lambda, p, phys_site=x.phys_site)

"""
Builds the Alcaraz MPO using the automated OpSum-to-MPO Euler builder.
Uses Algorithm("WII") exactly as specified in the package examples.
"""
function expH_alcaraz(sites::Vector{<:Index}, lambda::Number, p::Number; dt::Number) 
    N = length(sites) 
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

    # Time step conversion for exp(-i * H * dt) 
    tau = -im * dt  
     
    # Call the expmpo function using Algorithm("WII") syntax 
    return expmpo(os, sites, tau; alg=Algorithm("WII")) 
     
end

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

function ITransverse.expH(sites::Vector{<:Index}, mp::AlcarazParams, ::AlcarazWII; dt::Number)
    os = alcaraz_opsum(length(sites), mp.lambda, mp.p)
    return expmpo(os, sites, -im * dt; alg=Algorithm("WII"))
end


################################################################################################################

""" Builds the optimal 2nd-order MPO Generator for the Alcaraz model (WII construction) """

# HELPER FUNCTIONS

# Computes the Direct Product on virtual legs and Matrix Product on physical legs
function otimes(X::Matrix{Matrix{ComplexF64}}, Y::Matrix{Matrix{ComplexF64}})
    nx, mx = size(X)
    ny, my = size(Y)
    
    Z = Matrix{Matrix{ComplexF64}}(undef, nx * ny, mx * my)
    
    for ix in 1:nx, jx in 1:mx
        for iy in 1:ny, jy in 1:my
            row = (ix - 1) * ny + iy
            col = (jx - 1) * my + jy
            Z[row, col] = X[ix, jx] * Y[iy, jy] # Physical matrix multiplication
        end
    end
    return Z
end

# Computes the symmetric sum over all unique permutations of a set of operators
function sym_sum(ops::Vector{Matrix{Matrix{ComplexF64}}})
    perms = unique(permutations(ops))
    
    res = nothing
    for p in perms
        term = p[1]
        for k in 2:length(p)
            term = otimes(term, p[k])
        end
        
        if res === nothing
            res = copy(term)
        else
            res .+= term
        end
    end
    return res
end


# OPTMIAL 2ND ORDER MPO GENERATOR

function expH_alcaraz_2nd_opt(sites::Vector{<:Index}, lambda::Number, p::Number; dt::Number)
    N = length(sites)
    
    I2 = ComplexF64[1 0; 0 1]
    Z  = ComplexF64[1 0; 0 -1]
    X  = ComplexF64[0 1; 1 0]
    O2 = ComplexF64[0 0; 0 0]
    
    tau = -im * dt 

    # BASE HAMILTONIAN BLOCKS:

    D = Matrix{Matrix{ComplexF64}}(undef, 1, 1)
    D[1, 1] = -lambda .* Z
    
    C = Matrix{Matrix{ComplexF64}}(undef, 1, 3)
    C[1, 1] = Z; C[1, 2] = O2; C[1, 3] = Z
    
    B = Matrix{Matrix{ComplexF64}}(undef, 3, 1)
    B[1, 1] = -X; B[2, 1] = -p .* X; B[3, 1] = -p * lambda .* Z
    
    A = Matrix{Matrix{ComplexF64}}(undef, 3, 3)
    for i in 1:3, j in 1:3
        A[i, j] = O2
    end
    A[1, 2] = I2
    
    I_op = Matrix{Matrix{ComplexF64}}(undef, 1, 1)
    I_op[1, 1] = I2

    # Using formulasfrom Appendix A of Van Damme et al. 2023:
    
    W11 = I_op .+ tau .* D .+ (tau^2 / 2) .* otimes(D, D) .+ (tau^3 / 6) .* otimes(D, otimes(D, D))
    W12 = C .+ (tau / 2) .* sym_sum([C, D]) .+ (tau^2 / 6) .* sym_sum([C, D, D])
    W13 = otimes(C, C) .+ (tau / 3) .* sym_sum([C, C, D])

    W21 = tau .* B .+ (tau^2 / 2) .* sym_sum([B, D]) .+ (tau^3 / 6) .* sym_sum([B, D, D])
    W22 = A .+ (tau / 2) .* (sym_sum([B, C]) .+ sym_sum([A, D])) .+ (tau^2 / 6) .* (sym_sum([C, B, D]) .+ sym_sum([A, D, D]))
    W23 = sym_sum([A, C]) .+ (tau / 3) .* (sym_sum([A, C, D]) .+ sym_sum([C, C, B]))

    W31 = (tau^2 / 2) .* otimes(B, B) .+ (tau^3 / 6) .* sym_sum([B, B, D])
    W32 = (tau / 2) .* sym_sum([A, B]) .+ (tau^2 / 6) .* (sym_sum([A, B, D]) .+ sym_sum([B, B, C]))
    W33 = otimes(A, A) .+ (tau / 3) .* (sym_sum([A, B, C]) .+ sym_sum([A, A, D]))

    # Assemble full 13x13 block matrix
    W_opt = hvcat((3, 3, 3), W11, W12, W13, W21, W22, W23, W31, W32, W33)
    
    # Build the ITensor MPO (new bond dimension is 1 (W11) + 3 (W22) + 9 (W33) = 13)
    # Assemble full 13x13 block matrix of 2x2 matrices
    W_opt = hvcat((3, 3, 3), W11, W12, W13, W21, W22, W23, W31, W32, W33)
    dim_opt = 13
    
    # FLATTEN ONCE: Convert Matrix{Matrix} into a contiguous 4D Julia Array
    W_dense = zeros(ComplexF64, dim_opt, dim_opt, 2, 2)
    for row in 1:dim_opt, col in 1:dim_opt
        W_dense[row, col, :, :] = W_opt[row, col]
    end
    
    links = [Index(dim_opt, "Link,l=$i") for i in 1:N-1]
    U_dt = MPO(sites)
    
    for i in 1:N
        s = sites[i]
        
        if i == 1
            # Slice the 4D array to a 3D array (col, s', s) taking only the 1st row
            U_dt[i] = itensor(W_dense[1, :, :, :], links[1], s', s)
            
        elseif i == N
            # Slice the 4D array to a 3D array (row, s', s) taking only the 1st column
            U_dt[i] = itensor(W_dense[:, 1, :, :], links[i-1], s', s)
            
        else
            # Cast the full 4D array (row, col, s', s) in one shot
            U_dt[i] = itensor(W_dense, links[i-1], links[i], s', s)
        end
    end
    
    return U_dt
end


# 1ST ORDER MPO GENERATOR

function expH_alcaraz_1st_opt(sites::Vector{<:Index}, lambda::Number, p::Number; dt::Number)
    N = length(sites)
    
    I2 = ComplexF64[1 0; 0 1]
    Z  = ComplexF64[1 0; 0 -1]
    X  = ComplexF64[0 1; 1 0]
    O2 = ComplexF64[0 0; 0 0]
    
    tau = -im * dt 

    D = Matrix{Matrix{ComplexF64}}(undef, 1, 1); D[1, 1] = -lambda .* X
    C = Matrix{Matrix{ComplexF64}}(undef, 1, 3); C[1, 1] = Z; C[1, 2] = O2; C[1, 3] = X
    B = Matrix{Matrix{ComplexF64}}(undef, 3, 1); B[1, 1] = -Z; B[2, 1] = -p .* Z; B[3, 1] = -p * lambda .* X
    
    A = Matrix{Matrix{ComplexF64}}(undef, 3, 3)
    for i in 1:3, j in 1:3 A[i, j] = O2 end
    A[1, 2] = I2
    
    I_op = Matrix{Matrix{ComplexF64}}(undef, 1, 1); I_op[1, 1] = I2

    W11 = I_op .+ tau .* D .+ (tau^2 / 2) .* otimes(D, D)
    W12 = C .+ (tau / 2) .* sym_sum([C, D])

    W21 = tau .* B .+ (tau^2 / 2) .* sym_sum([B, D])
    W22 = A .+ (tau / 2) .* (sym_sum([A, D]) .+ sym_sum([B, C]))


    W_opt_1st = hvcat((2, 2), W11, W12, W21, W22)
    
    # Build ITensor MPO
    dim_opt = 4
    links = [Index(dim_opt, "Link,l=$i") for i in 1:N-1]
    U_dt = MPO(sites)
    
    for i in 1:N
        s = sites[i]
        if i == 1
            W = ITensor(ComplexF64, links[1], s', s)
            for col in 1:dim_opt, s1 in 1:2, s2 in 1:2
                W[links[1]=>col, s'=>s1, s=>s2] = W_opt_1st[1, col][s1, s2]
            end
            U_dt[i] = W
        elseif i == N
            W = ITensor(ComplexF64, links[i-1], s', s)
            for row in 1:dim_opt, s1 in 1:2, s2 in 1:2
                W[links[i-1]=>row, s'=>s1, s=>s2] = W_opt_1st[row, 1][s1, s2]
            end
            U_dt[i] = W
        else
            W = ITensor(ComplexF64, links[i-1], links[i], s', s)
            for row in 1:dim_opt, col in 1:dim_opt, s1 in 1:2, s2 in 1:2
                W[links[i-1]=>row, links[i]=>col, s'=>s1, s=>s2] = W_opt_1st[row, col][s1, s2]
            end
            U_dt[i] = W
        end
    end
    
    return U_dt
end



# 4x4 WII Zalatel

function expH_alcaraz_WII(sites::Vector{<:Index}, lambda::Number, p::Number; dt::Number)
    N = length(sites)
    
    I2 = ComplexF64[1 0; 0 1]
    Z  = ComplexF64[1 0; 0 -1]
    X  = ComplexF64[0 1; 1 0]
    O2 = ComplexF64[0 0; 0 0]
    
    # Define Hamiltonian Blocks
    D_op = -lambda .* X
    C_op = [Z, O2, X]
    B_op = [-Z, -p .* Z, -p * lambda .* X]
    
    A_op = fill(O2, 3, 3)
    A_op[1, 2] = I2
    
    tau = -im * dt
    
    W_II = Matrix{Matrix{ComplexF64}}(undef, 4, 4)  # matrix 4x4 of matrices 2x2 (8x8 total)
    
    # we loop over all combinations of a and a_bar (from 1 to 3 possible states) 
    # to construct the 8x8 matrix M that encodes the transition rules of the Zaletel MPO construction
    for a in 1:3
        for a_bar in 1:3
            M = zeros(ComplexF64, 8, 8)
            
            function set_M!(i, j, mat)
                M[2*(i-1)+1 : 2*i, 2*(j-1)+1 : 2*j] .= mat
            end
            
            # Zaletel 8x8 Boson transitions
            set_M!(1, 1, tau .* D_op)
            
            set_M!(2, 1, sqrt(tau) .* B_op[a])
            set_M!(2, 2, tau .* D_op)
            
            set_M!(3, 1, sqrt(tau) .* C_op[a_bar])
            set_M!(3, 3, tau .* D_op)
            
            set_M!(4, 1, A_op[a, a_bar])
            set_M!(4, 2, sqrt(tau) .* C_op[a_bar])
            set_M!(4, 3, sqrt(tau) .* B_op[a])
            set_M!(4, 4, tau .* D_op)
            
    
            E = exp(M)
            
            
            W_II[1, 1] = E[1:2, 1:2]                 
            W_II[1+a, 1] = E[3:4, 1:2]               
            W_II[1, 1+a_bar] = E[5:6, 1:2]           
            W_II[1+a, 1+a_bar] = E[7:8, 1:2]         
        end
    end
    
    links = [Index(4, "Link,l=$i") for i in 1:N-1]
    U_dt = MPO(sites)
    
    for i in 1:N
        s = sites[i]
        
        if i == 1
            W = ITensor(ComplexF64, links[1], s', s)
            for col in 1:4, s1 in 1:2, s2 in 1:2
                W[links[1]=>col, s'=>s1, s=>s2] = W_II[1, col][s1, s2]
            end
            U_dt[i] = W
        elseif i == N
            W = ITensor(ComplexF64, links[i-1], s', s)
            for row in 1:4, s1 in 1:2, s2 in 1:2
                W[links[i-1]=>row, s'=>s1, s=>s2] = W_II[row, 1][s1, s2]
            end
            U_dt[i] = W
        else
            W = ITensor(ComplexF64, links[i-1], links[i], s', s)
            for row in 1:4, col in 1:4, s1 in 1:2, s2 in 1:2
                W[links[i-1]=>row, links[i]=>col, s'=>s1, s=>s2] = W_II[row, col][s1, s2]
            end
            U_dt[i] = W
        end
    end
    
    return U_dt
end