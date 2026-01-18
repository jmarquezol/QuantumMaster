import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# HOMEWORK 3: RESONANT TUNNELING DIODE (RTD) SIMULATION
# =============================================================================

# -----------------------------------------------------------------------------
# PHYSICAL CONSTANTS & PARAMETERS
# -----------------------------------------------------------------------------
Dx = 1e-10              # Spatial step (1 Angstrom)
L_device = 2e-8         # Simulation Box length (large enough to include the Reservoirs/Leads)
                        # ensuring electrons behave as free waves before and after hitting the barriers.
m = 0.063 * 9.109e-31   # Effective electron mass (kg)
h_bar = 1.054e-34       # Reduced Planck constant (J·s)
qe = 1.6e-19            # Electron charge (C)
Kb = 1.38e-23           # Boltzmann constant (J/K)
T = 298                 # Temperature (K)
Area = 1e-12            # Cross-sectional area (m^2)

# Geometry parameters
W_barrier = 4e-9        # 4 nm
W_well = 10e-9          # 10 nm
V0 = 0.3                # Barrier height/Potential offset (eV)
Ef = 0.005               # Fermi energy (eV)

# -----------------------------------------------------------------------------
# SPATIAL GRID SETUP (FOR PARTS 1 & 2)
# -----------------------------------------------------------------------------
xpoints = int(np.floor(L_device / Dx))
# x = np.arange(1, xpoints + 1) * Dx

# Define indices for the Double Barrier Structure V(x)
# Center the device in the simulation box
center_idx = xpoints // 2
half_well_pts = int((W_well / 2) / Dx)
barrier_pts = int(W_barrier / Dx)

# Define (Symmetric) Barrier Regions
# Left Barrier: [start_B1, end_B1]
end_B1 = center_idx - half_well_pts
start_B1 = end_B1 - barrier_pts

# Right Barrier: [start_B2, end_B2]
start_B2 = center_idx + half_well_pts
end_B2 = start_B2 + barrier_pts

# Initial Potential (V=0)
U0 = np.zeros(xpoints)
U0[start_B1:end_B1] = V0 * qe  # Left Barrier
U0[start_B2:end_B2] = V0 * qe  # Right Barrier

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def get_transmission(Energy, Potential):
    """
    Calculates Transmission T(E) using the Numerov Method for a given Potential profile.

    Inputs:
    :Energy: Energy values (in eV)
    :Potential: Potential Energy V(x) (in Joules) 
    """
    N_points = len(Potential)
    x = np.arange(1, N_points + 1) * Dx

    T_prob = np.zeros(len(Energy))
    
    # Numerov Constants (for Equation 1)
    A = (10 * Dx**2) / 12
    AA = Dx**2 / 12
    const_factor = (2 * m) / h_bar**2

    for i, E in enumerate(Energy):
        E_joule = E * qe
        
        # We ensure E > U at boundaries (leads) to avoid k being imaginary
        if E_joule <= Potential[0] or E_joule <= Potential[-1]:
            T_prob[i] = 0.0
            continue

        # Wavevectors in extremes (scattering states): k = sqrt(2m(E-V)) / h_bar
        k_left = np.sqrt(const_factor * (E_joule - Potential[0]))
        k_right = np.sqrt(const_factor * (E_joule - Potential[-1]))

        # Numerov Function: f(x) = 2m/h^2 * (V(x) - E)
        # Equation to be solved: psi'' = f(x)psi.
        func_numerov = const_factor * (Potential - E_joule)

        # Initialize Wavefunction (first two points at right boundary)
        # We assume outgoing plane waves on the right are prop. to exp(i * k_right * x) (scattering state)
        # This is the same as assuming a wave has successfully transmited to the right side
        psi = np.zeros(N_points, dtype=complex)
        psi[-1] = np.exp(1j * k_right * x[-1])
        psi[-2] = np.exp(1j * k_right * x[-2])

        # Backward Integration from Right to Left (Equation 1)
        for j in range(N_points - 3, -1, -1):
            term1 = (2 + A * func_numerov[j+1]) * psi[j+1]
            term2 = (1 - AA * func_numerov[j+2]) * psi[j+2]
            denom = (1 - AA * func_numerov[j])
            psi[j] = (term1 - term2) / denom

        # Now let's match boundary conditions at the left side:
        # Psi_left = A * exp(ikx) + B * exp(-ikx)
        # We want T = (k_right / k_left) * |Amplitude_Transmitted|^2 / |Amplitude_Incident|^2
        #           = (k_right / k_left) * (1 / |A|^2)
        # We need to isolate A. We do it using Psi_left and Psi'_Left = ik * A * exp(ikx) -ik * B * exp(-ikx)
        # Knowing Psi and Psi', we have two unkowns A and B --> A = ... = (1/2) exp(-ikx) (Psi(x) + Psi'(x) / (ik))
        
        # First, we get numerical values for psi and psi' in a safe point (not at the edges)
        idx_match = 5
        psi_val = psi[idx_match]
        # Finite difference derivative: dpsi/dx
        d_psi = (psi[idx_match+1] - psi[idx_match]) / Dx 
        
        # Second, we get the incident amplitude with the derived formula above:
        A_inc = 0.5 * np.exp(-1j * k_left * x[idx_match]) * (psi_val + d_psi / (1j * k_left))
        
        if np.abs(A_inc) == 0:
            T_prob[i] = 0
        else:
            T_prob[i] = (k_right / k_left) / (np.abs(A_inc)**2)
            
    return T_prob

def simulate_multibarrier(num_barriers, W_b=4e-9, W_w=10e-9, V_height=0.3):
    """
    Constructs a potential with 'num_barriers' and computes its transmission coeff.

    Returns:
        :x_new: set of new points for the simulation
        :U_multi: Potential Multi-barrier profile
        :T_multi: Transmission coeffs
    """
    # Define region for computation
    # Device length = (N * Barrier) + (N-1 * Well)
    device_active_len = (num_barriers * W_b) + ((num_barriers - 1) * W_w)
    
    # Add leads on both sides (10 nm each)
    L_leads = 10e-9 
    L_total = device_active_len + 2 * L_leads
    
    xpoints_new = int(np.floor(L_total / Dx))
    x_new = np.arange(1, xpoints_new + 1) * Dx
    
    # Potential
    U_multi = np.zeros(xpoints_new)
    
    current_pos_idx = int(L_leads / Dx)     # start painting the potential at the end of the left lead
    barrier_width = int(W_b / Dx)
    well_width = int(W_w / Dx)
    
    for k in range(num_barriers):
        start = current_pos_idx
        end = start + barrier_width
        U_multi[start:end] = V_height * qe
        
        # move current position to the next barrier position start
        current_pos_idx = end + well_width
        
    # Compute Transmission coeff
    T_multi = get_transmission(Energies, U_multi)
    
    return x_new, U_multi, T_multi


# -----------------------------------------------------------------------------
# PART 1: TRANSMISSION COEFFICIENT (V=0)
# -----------------------------------------------------------------------------

# Energy Grid
E_max = 0.5 # eV (Up to barrier height + bit more)
E_steps = 5000
Energies = np.linspace(0.001, E_max, E_steps) 

# Plotting T(E) for V=0 first (no bias)
T_zero_bias = get_transmission(Energies, U0)

plt.figure(figsize=(10, 6))
plt.plot(Energies, T_zero_bias, 'b-', linewidth=2)
plt.title(f'Transmission Probability at V=0\nBarrier={W_barrier*1e9}nm, Well={W_well*1e9}nm')
plt.xlabel('Energy (eV)')
plt.ylabel('T(E)')
plt.grid(True)
plt.show()

# -----------------------------------------------------------------------------
# PART 2: I-V CHARACTERISTIC
# -----------------------------------------------------------------------------

# Voltage Grid
V_max = 0.6 # max value enough to see the negative slope region, in Volts
V_steps = 150
Voltages = np.linspace(0, V_max, V_steps)
Currents = np.zeros(V_steps)

# Active region (where voltage drops) defined by:
start_active = start_B1
end_active = end_B2
points_active = end_active - start_active       # number of points in the active region

for i, Vb in enumerate(Voltages):
    # 1. Tilt Potential in the active region
    # Left reservoir has V = 0, while right one has V = - Vb = constant
    U_bias = U0.copy()
    U_bias[end_active:] = U_bias[end_active:] - (Vb * qe)

    # Create (linear) voltage drop in active region as Vb * fraction
    # fraction = 0 at the start, and = 1 at the end
    points_slope = np.arange(points_active)
    fraction = (points_slope / points_active)
    voltage_drop = (Vb * qe) * fraction
    U_bias[start_active:end_active] -= voltage_drop
    
    # 2. Calculate T(E, V) for this specific (tilted) potential profile
    Trans_V = get_transmission(Energies, U_bias)
    
    # 3. Define chemical potentials (energy window)
    mu_L = Ef           # Source fixed
    mu_R = Ef - Vb      # Drain drops by Vb
    
    # 4. Equation 3 (to compute J(V))
    # Note Energies are in eV, so we multiply by qe -> Joules
    arg_L = (mu_L - Energies) * qe / (Kb * T)
    arg_R = (mu_R - Energies) * qe / (Kb * T)
    
    integrand = Trans_V * np.log( (1 + np.exp(arg_L)) / (1 + np.exp(arg_R)))
    integral = np.trapezoid(integrand, Energies) * qe   # Numerical integration (in Joules)

    J_V = (qe * m * Kb * T) / (2 * np.pi**2 * h_bar**3) * integral
    
    Currents[i] = J_V * Area


plt.figure(figsize=(10, 6))
plt.plot(Voltages, Currents, 'r-', linewidth=2)
plt.title('I-V Characteristic of RTD')
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A)')
plt.grid(True)
plt.show()

# -----------------------------------------------------------------------------
# PART 3: BAND STRUCTURE (MULTI-BARRIER)
# -----------------------------------------------------------------------------

# Plotting Band Structure Development
N_values = [2, 5, 10, 20]

for i, N in enumerate(N_values):
    x_mb, U_mb, T_mb = simulate_multibarrier(N, W_barrier, W_well, V0)
    
    plt.figure(figsize=(8, 8))
    
    # Potential Plots
    plt.subplot(2, 1, 1)
    plt.plot(x_mb * 1e9, U_mb / qe, 'k-')
    plt.title('Potential Profile (N = ' + str(N) + ')')
    plt.ylabel('Potential (eV)')
    plt.xlabel('Position (nm)')
    plt.grid(True)
    
    # Transmission Plots
    plt.subplot(2, 1, 2)
    plt.plot(Energies, T_mb, 'b-')
    plt.title('Transmission Spectrum (N = ' + str(N) + ')')
    plt.ylabel('Transmission')
    plt.xlabel('Energy (eV)')
    plt.grid(True)
    
    plt.subplots_adjust(hspace=0.4)
    plt.show()