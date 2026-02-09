import numpy as np
from matrix_product_states import MPS
import opt_einsum as oe
import itertools

class PEPS:

    def __init__(self, Lx, Ly, d_phys, D, A):
        """
        Initialize PEPS object (Projected Entangled Pair States)
        defined on a 2D grid of size Lx * Ly
        
        :param self: -
        :param Lx: horizontal size of the lattice
        :param Ly: vertical size of the lattice
        :param d_phys: physical dimension of quantum particles
        :param D: bond dimension of the PEPS tensors
        :param A: grid of tensors. A[x][y] = tensor @ site (x, y)

        Tensor Convention: (physical, left, up, right, down)
        """
        self.Lx = Lx
        self.Ly = Ly
        self.d_phys = d_phys
        self.D = D
        self.A = A

    def compute_norm(self, D_bound):
        """
        Computes squared norm <Psi|Psi> using the "Sandwich Contraction" method:
        1. Contract the 2D network row by row from top to bottom
        2. Boundary between contracted part and the rest is approximated
        as an MPS with bond dim D_bound
        
        :param D_bound: maximum bond dimension D' for the boundary MPS approx.
        """

        # 1. BOUNDARY MPS (top row)
        mps_tensors = []
        for y in range(self.Ly):
            # Tensor A and its conjugate. Each has shaphe (d_phys, L, 1, R, D)
            T = self.A[0][y]
            T_conj = T.conj()

            # Contract physical indices and reorder legs
            temp = np.tensordot(T, T_conj, axes=([0], [0]))         # shape: (L, 1, R, D, L*, 1*, R*, D*)
            temp = np.squeeze(temp, axis=(1, 5))                    # since U = U' = 1, we squeeze these axes
                                                                    # new shape = (L, R, D, L*, R*, D*)
            temp = np.transpose(temp, (2, 5, 0, 3, 1, 4))     # shape: (D, D*, L, L*, R, R*)

            # Reshape
            shape = temp.shape
            d_mps = shape[0] * shape[1]     # D * D
            b_left = shape[2] * shape[3]    # left leg
            b_right = shape[4] * shape[5]

            new_T = temp.reshape(d_mps, b_left, b_right)    # we ignore the last two legs which are 1

            mps_tensors.append(new_T)
        # Boundary MPS object_
        boundary_mps = MPS(self.Ly, self.D**2, mps_tensors)

        # 2. INTERMEDIATE ROWS (MPOs)
        for x in range(1, self.Lx):
            
            mpo_tensors = []
            for y in range(self.Ly):
                # Tensor A and its conjugate
                # Shape: (d_phys, L, U, R, D)
                T = self.A[x][y]
                T_conj = T.conj()
                
                # Contract physical indices
                temp = np.tensordot(T, T_conj, axes=([0], [0]))         # shape: (L, U, R, D, L*, U*, R*, D*)
                
                # MPO structure: (Phys_Out, Phys_In, Left, Right)
                #   Input (from prev row) = Up legs
                #   Output (to next row)  = Down legs
                temp = np.transpose(temp, (3, 7, 1, 5, 0, 4, 2, 6))     # shape: (D, D*, U, U*, L, L*, R, R*)
                
                # Reshape
                shape = temp.shape
                d_out = shape[0] * shape[1] # Down legs
                d_in = shape[2] * shape[3]  # Up legs (connecting to previous boundary)
                b_left = shape[4] * shape[5]
                b_right = shape[6] * shape[7]
                
                new_W = temp.reshape(d_out, d_in, b_left, b_right)
                mpo_tensors.append(new_W)
            
            # Apply the row as an MPO to the boundary state
            boundary_mps = boundary_mps.apply_mpo(mpo_tensors)

            # Truncate the boundary bond dimension to D_bound (D') 
            boundary_mps.compress(max_bond_dim=D_bound)

        # 3. FINAL CONTRACTION
        # Final MPS has open "Down" legs (of dim = 1) which we contract with the vector |00...0>
        final_idx = [0] * self.Ly  # list [0, 0, ...]

        result = boundary_mps.compute_amplitude(final_idx)

        return np.real(result)
    

    def contract_2d(self, D_bound, fixed_config=None):
        """"
        Computes contraction of rectangular and finite PEPS.
        We sum over the physical index of each tensor

        :param D_bound: max bond dim D' for the boundary MPS
        :fixed_config: 2D array where fixed sites have values 0 or 1
                       Sites with -1 value are not fixed and we sum over their physical index as usual
        :return: scalar result of contraction
        """

        # Helper function
        def get_tensor_with_fixed_config(x, y, T):
            val = None
            if fixed_config is not None:
                v = fixed_config[x][y]
                if v != -1: val = v

            if val is not None:
                # If site is fixed, we select the corresponding slice of the tensor
                # e.g. if val = 0, we take T[0, :, :, :, :]
                return T[val, :, :, :, :]
            else:
                return np.sum(T, axis=0) # sum over physical index as usual

        # 1. BOUNDARY MPS (top row)
        mps_tensors = []
        for y in range(self.Ly):
            T_orig = self.A[0][y]                # shape = (d_phys, L, U=1, R, D)

            # Sum over physical index to trace it out
            temp = get_tensor_with_fixed_config(0, y, T_orig)        # shape = (L, U=1, R, D)
            temp = np.squeeze(temp, axis=1) # squeeze over axis 1 to remove U=1. new shape = (L, R, D)

            # Reshape it as a MPS w/ shape = (D, L, R) 
            # Note down leg = phys leg of the MPS, and
            temp = np.transpose(temp, (2, 0, 1))

            shape = temp.shape
            d_mps = shape[0]
            b_left = shape[1]
            b_right = shape[2]

            new_T = temp.reshape(d_mps, b_left, b_right)
            mps_tensors.append(new_T)
        
        # Cretae MPS object, with physical dimension = D bond dim of PEPS
        boundary_mps = MPS(self.Ly, self.D, mps_tensors) 

        # 2. INTERMEDIATE ROWS (MPS - MPOs)
        for x in range(1, self.Lx):

            mpo_tensors = []
            for y in range(self.Ly):
                W_orig = self.A[x][y]

                temp = get_tensor_with_fixed_config(x, y, W_orig)      # shape = (L, U, R, D)

                # MPO structure/shape = (phys_out, phys_in, Left, Right)
                # input (from prev row) = up leg
                # output (to next row)  = down leg
                temp = np.transpose(temp, (3, 1, 0, 2))

                # Reshape
                shape = temp.shape
                d_out = shape[0]
                d_in = shape[1]
                b_left = shape[2]
                b_right = shape[3]

                new_W = temp.reshape(d_out, d_in, b_left, b_right)
                mpo_tensors.append(new_W)
            
            # Apply the next row as an MPO to boundary MPS
            boundary_mps = boundary_mps.apply_mpo(mpo_tensors)

            # Truncate to D_bound
            boundary_mps.compress(max_bond_dim=D_bound)
        
        # 3. FINAL CONTRACTION
        final_idx = [0] * self.Ly
        result = boundary_mps.compute_amplitude(final_idx)

        return np.real(result)
    
    def contract_2d_exact(self):
        """
        Computes EXACT contraction of the 2D PEPS grid by tracing out
        physical indices and using opt_einsum for global path optimization.
        
        Note: This scales exponentially and is only feasible for small grids.
        """
        tensors_list = []
        indices_list = []

        for x in range(self.Lx):
            for y in range(self.Ly):
                # 1. Trace out the physical dimension
                T = self.A[x][y]
                T_reduced = np.sum(T, axis=0) # Shape: (Left, Up, Right, Down)
                tensors_list.append(T_reduced)
                
                # 2. Assign unique string IDs to every horizontal (h) and vertical (v) bond
                # Left Leg
                if y == 0:          # 1st column
                    idx_L = f"bL_{x}_{y}"
                else:      
                    idx_L = f"h_{x}_{y-1}"
                
                # Up Leg
                if x == 0:          # 1st row
                    idx_U = f"bU_{x}_{y}"
                else:      
                    idx_U = f"v_{x-1}_{y}"
                
                # Right Leg
                if y == self.Ly-1:  # last column
                    idx_R = f"bR_{x}_{y}"
                else:         
                    idx_R = f"h_{x}_{y}"
                
                # Down Leg
                if x == self.Lx-1:  # last row
                    idx_D = f"bD_{x}_{y}"
                else:         
                    idx_D = f"v_{x}_{y}"
                
                # Note that the right index of tensor (0, 0) = left index of tensor (0, 1) = h_0_0
                # this match triggers the tensor contraction in opt_einsum
                
                indices_list.append([idx_L, idx_U, idx_R, idx_D])

        # 3. Pack arguments in format: tensor1, idx1, tensor2, idx2...
        contract_args = []
        for t, idx in zip(tensors_list, indices_list):
            contract_args.append(t)
            contract_args.append(idx)

        # 4. Perform the contraction
        # opt_einsum will automatically find the most efficient contraction order
        result = oe.contract(*contract_args)
        
        # Squeeze remaining boundary legs (dim=1) to return a scalar
        return np.real(float(np.squeeze(result)))


    @classmethod
    def create_random_2d_peps(cls, Lx, Ly, d_phys, D, seed=None):
        """
        Creates a 3D PEPS object with random tensors
        
        :param Lx: horizontal size of the lattice
        :param Ly: vertical size of the lattice
        :param d_phys: physical dimension of quantum particles
        :param D: bond dimension of the PEPS tensors
        :param seed: for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)

        grid_tensors = []
        for x in range(Lx):
            row_tensors = []
            for y in range(Ly):
                # set bond dim = 1 if tensor @ the edges
                # convention: (d_phys, left, up, right, down)
                dim_L = 1 if y == 0 else D
                dim_U = 1 if x == 0 else D
                dim_R = 1 if y == Ly-1 else D
                dim_D = 1 if x == Lx-1 else D

                T = np.random.rand(d_phys, dim_L, dim_U, dim_R, dim_D)
                T /= np.linalg.norm(T)

                row_tensors.append(T)
            grid_tensors.append(row_tensors)
            
        return cls(Lx, Ly, d_phys, D, grid_tensors)
    
    @classmethod
    def create_ising_2d(cls, Lx, Ly, beta, d_phys=2, D=2, J=1.0):
        """
        Mapping between classical Ising model and the PEPS TN:
            Creates a PEPS representing the 2D Ising partition function Z.
            Note the physical index (d_phys=2) represent local spin. Summing over this index gives Z.

        :param beta: inverse temp (1/kT)
        :param J: hopping constant
        """

        d_phys = 2  # spin up/down
        D = 2       # fix bond dim for Ising PEPS

        # Boltzmann Matrix W (containing interaction weights)
        # W = [W_00 W_01]
        #     [W_10 W_11]   st W_00 = interaction between spin up and neighbour spin up
        #                      W_01 = int between spin up and neighbour spin down
        #                      W_10 = int between spin down and neighbour spin up
        #                      W_11 = int between spin down and neighbour spin down
        W = np.array([[np.exp(beta * J), np.exp(- beta * J)],
                      [np.exp(- beta * J), np.exp(beta * J)]])
        # Find M = sqrt(W) to pull half of the interaction weight into each of the two adjacent sites (where tensors are)
        evals, evecs = np.linalg.eigh(W)
        M = evecs @ np.diag(np.sqrt(evals)) @ evecs.T

        grid_tensors = []
        for x in range(Lx):
            row_tensors = []
            for y in range(Ly):
                # Open Boundary Conditions
                dim_L = 1 if y == 0 else D
                dim_U = 1 if x == 0 else D
                dim_R = 1 if y == Ly-1 else D
                dim_D = 1 if x == Lx-1 else D

                # Initialize Tensor with (phys, L, U, R, D)
                T = np.zeros((d_phys, dim_L, dim_U, dim_R, dim_D))

                for s in range(d_phys):     # s = {0, 1} = spin -1/+1
                    # Each site s collects the (half-bond) weights from its 4 neighbouring bonds
                    #   For spin state s, we take the s-th row of M for each direction
                    #       - If s = 0, v = M[0, :] = [M_00, M_01]
                    #       - If s = 1, v = M[1, :] = [M_10, M_11]
                    #   If in boundary, set dummy 1.0 factor
                    v_L = M[s, :] if y > 0 else np.array([1.0])
                    v_U = M[s, :] if x > 0 else np.array([1.0])
                    v_R = M[s, :] if y < Ly-1 else np.array([1.0])
                    v_D = M[s, :] if x < Lx-1 else np.array([1.0])

                    # Using these 4 vectors and take the outerproduct -> rank-4 tensor
                    # vL_i x vU_j x vR_k x vD_l --> outer_ijkl with shape (dim_L, dim_U, dim_R, dim_D)
                    outer = np.einsum('i,j,k,l->ijkl', v_L, v_U, v_R, v_D)
                    T[s, :, :, :, :] = outer 
                row_tensors.append(T)
            grid_tensors.append(row_tensors)
        
        return cls(Lx, Ly, d_phys, D, grid_tensors)
    
    @classmethod
    def compute_Z_brute_force(cls, Lx, Ly, beta, J=1.0):
        """
        Computes exact partition function Z using brute-force summation over all spin configurations.
        Problem scales as O(2^(Lx*Ly)) so it's only feasible for small systems
        """
        N = Lx * Ly
        Z = 0.0

        # For loop over all spin configurations of N spins (-1, +1)
        for config_spins in itertools.product([-1, +1], repeat=N):
            # reshape config into 2D grid
            grid_spins = np.array(config_spins).reshape((Lx, Ly))

            energy = 0.0

            # Sum over all horizontal pair bonds
            horizontal_pairs = grid_spins[:, :-1] * grid_spins[:, 1:] # for each row, multiply spin with right neighbour
            energy += - J * np.sum(horizontal_pairs)

            # Sum over all vertical pair bonds
            vertical_pairs = grid_spins[:-1, :] * grid_spins[1:, :]   # for each column, multiply spin with bottom neighbour
            energy += - J * np.sum(vertical_pairs)

            Z += np.exp(- beta * energy)

        return Z


    def sample_configuration(self, D_bound):
        """
        Generates a single configuration sample from the PEPS distribution 
        using the Chain Rule and Baye's Rule.
        
        :param D_bound: bond dimension for the boundary contraction
        Returns: 
            (config, log_prob)
            - config: 2D array of spins (0 or 1)
            - log_prob: The natural log of the probability P(config) according to the TN.        
        """
        # Initialize config. ( -1 = Free )
        current_config = np.full((self.Lx, self.Ly), -1, dtype=int)
        log_prob = 0.0  # Accumulator for log(P(s_1)) + log(P(s_2|s_1)) + ...
                
        for x in range(self.Lx):
            for y in range(self.Ly):
                # Wompute the weights for the two possible spin states at each site
                # conditioned on the already sampled spins (current_config) and the TN structure

                # Weight for Spin UP (0)
                config_try_0 = current_config.copy()
                config_try_0[x, y] = 0
                
                weight_0 = self.contract_2d(D_bound, fixed_config=config_try_0)
                
                # Compute Weight for Spin DOWN (1)
                config_try_1 = current_config.copy()
                config_try_1[x, y] = 1
                
                weight_1 = self.contract_2d(D_bound, fixed_config=config_try_1)
                
                # Avoid negative weights due to numerical instability
                weight_0 = max(0.0, weight_0)
                weight_1 = max(0.0, weight_1)
                
                total_weight = weight_0 + weight_1
                
                # Conditional Probability P(s_i | s_{<i}) ---
                if total_weight < 1e-15:
                    p0 = 0.5 # in case of numerical instability, we assign equal probability to both spin states
                else:
                    p0 = weight_0 / total_weight
                
                # Sample a random number to decide the spin state based on the computed probability
                r = np.random.rand()
                
                if r < p0:
                    chosen_spin = 0
                    p_chosen = p0
                else:
                    chosen_spin = 1
                    p_chosen = 1.0 - p0
                
                # Update current configuration for the next iterations
                current_config[x, y] = chosen_spin

                # Accumulate Log Prob (we use log to avoid underflow)
                if p_chosen < 1e-15:
                    log_prob = - np.inf
                else:
                    log_prob += np.log(p_chosen)
                
        return current_config, log_prob