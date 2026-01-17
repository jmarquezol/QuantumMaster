import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Homework 3: Resonant Tunneling Diode Simulation
# ------------------------------------------------------------

# ------------------------------------------------------------
# 1. Physical constants & Parameters
# ------------------------------------------------------------
Dx = 1e-10              # Spatial step (1 Angstrom)
L_device = 2e-8         # Simulation Box length (large enough to include leads)
m = 0.063 * 9.109e-31   # Effective electron mass (kg)
h_bar = 1.054e-34       # Reduced Planck constant (J·s)
qe = 1.6e-19            # Electron charge (C)
Kb = 1.38e-23           # Boltzmann constant (J/K)
T = 298                 # Temperature (K)
Area = 1e-12            # Cross-sectional area (m^2) NOT used in 1D current density but kept for consistency

# Geometry parameters
W_barrier = 4e-9        # 4 nm
W_well = 10e-9          # 10 nm
V0 = 0.3                # Barrier height (eV)
Ef = 0.05               # Fermi energy (eV) - Assumed typical value for GaAs reservoir

# ------------------------------------------------------------
# 2. Spatial Grid Setup
# ------------------------------------------------------------
xpoints = int(np.floor(L_device / Dx))
x = np.arange(1, xpoints+1) * Dx

# Define indices for the Double Barrier Structure
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

# ------------------------------------------------------------
# 3. Helper Functions
# ------------------------------------------------------------

def get_transmission(Energy_arr, Potential_arr):
    """
    Calculates Transmission T(E) using the Numerov Method for a given Potential profile.
    """
    T_prob = np.zeros(len(Energy_arr))
    
    # Numerov Constants (for Equation 1)
    A = (10 * Dx**2) / 12
    AA = Dx**2 / 12
    const_factor = (2 * m) / h_bar**2

    for i, E in enumerate(Energy_arr):
        E_joule = E * qe
        
        # Check for classic turning points at boundaries to avoid divergence
        # We ensure E > U at boundaries (leads) for plane waves
        if E_joule <= Potential_arr[0] or E_joule <= Potential_arr[-1]:
            T_prob[i] = 0.0
            continue

        # Wavevectors in extremes (scattering states)
        k_left = np.sqrt(const_factor * (E_joule - Potential_arr[0]))
        k_right = np.sqrt(const_factor * (E_joule - Potential_arr[-1]))

        # Numerov Function: f(x) = 2m/h^2 * (V(x) - E)
        # Equation to be solved: psi'' = f(x)psi.
        func_numerov = const_factor * (Potential_arr - E_joule)

        # Initialize Wavefunction (first two points at right boundary)
        # We assume outgoing plane waves on the right are prop. to exp(i * k_right * x) (scattering state)
        psi = np.zeros(xpoints, dtype=complex)
        psi[-1] = np.exp(1j * k_right * x[-1])
        psi[-2] = np.exp(1j * k_right * x[-2])

        # Backward Integration from Right to Left (Equation 1)
        for j in range(xpoints - 3, -1, -1):
            term1 = (2 + A * func_numerov[j+1]) * psi[j+1]
            term2 = (1 - AA * func_numerov[j+2]) * psi[j+2]
            denom = (1 - AA * func_numerov[j])
            psi[j] = (term1 - term2) / denom

        # Match boundary conditions at the Left side to extract Transmission
        # Plane wave form: Psi_left = A * exp(ikx) + B * exp(-ikx)
        # We want T = |1/A|^2 (normalized assuming incident amplitude 1)
        
        # Using a simple 2-point derivative estimate at the start (index 5 to avoid edge effects)
        idx_match = 5
        psi_val = psi[idx_match]
        # Finite difference derivative: dpsi/dx
        d_psi = (psi[idx_match+1] - psi[idx_match]) / Dx 
        
        # From scattering theory matching:
        # A = (psi + (1/ik)*psi') * exp(-ikx) / 2
        # However, your base code used a direct formula. Let's use the standard T definition:
        # T = (k_right / k_left) * |Amplitude_Transmitted|^2 / |Amplitude_Incident|^2
        # Here Transmitted Amp is 1 (assumed). Incident Amp is A.
        # So T = (k_right / k_left) * (1 / |A|^2)
        
        # Reconstructing Incident Amplitude A from calculated psi at left boundary:
        # A = 0.5 * exp(-i * k_left * x) * (psi + (1 / (1j * k_left)) * d_psi)
        
        A_inc = 0.5 * np.exp(-1j * k_left * x[idx_match]) * (psi_val + d_psi / (1j * k_left))
        
        if np.abs(A_inc) == 0:
            T_prob[i] = 0
        else:
            T_prob[i] = (k_right / k_left) / (np.abs(A_inc)**2)
            
    return T_prob

def fermi_dirac(E, Mu, T):
    """ Fermi-Dirac distribution """
    return 1.0 / (1.0 + np.exp((E - Mu) * qe / (Kb * T)))

# ------------------------------------------------------------
# 4. Simulation Loops
# ------------------------------------------------------------

# A. Energy Grid for integration
E_max = 0.5 # eV (Up to barrier height + bit more)
E_steps = 500
Energies = np.linspace(0.001, E_max, E_steps) 
dE = Energies[1] - Energies[0]

# B. Voltage Grid
V_max = 0.6 # Volts
V_steps = 21
Voltages = np.linspace(0, V_max, V_steps)
Currents = np.zeros(V_steps)

# Plotting T(E) for V=0 first (Part 1 of assignment)
T_zero_bias = get_transmission(Energies, U0)

plt.figure(figsize=(10, 6))
plt.plot(Energies, T_zero_bias, 'r-', linewidth=2)
plt.title(f'Transmission Probability at V=0\nBarrier={W_barrier*1e9}nm, Well={W_well*1e9}nm')
plt.xlabel('Energy (eV)')
plt.ylabel('T(E)')
plt.grid(True)
plt.show()

# C. I-V Calculation Loop
print("Starting I-V Scan...")
for i, Vb in enumerate(Voltages):
    # 1. Tilt Potential: U(x) = U0(x) - e * V * (x/L)
    # Applying linear drop across the whole simulation box
    U_bias = U0 - (Vb * qe) * (x / (x[-1]))
    
    # 2. Calculate Transmission T(E, V)
    Trans_V = get_transmission(Energies, U_bias)
    
    # 3. Define Chemical Potentials
    # Source (Left) fixed at Ef, Drain (Right) drops by Vb
    mu_L = Ef
    mu_R = Ef - Vb
    
    # 4. Landauer-Büttiker Integration
    # I = (2e/h) * Integral [ T(E) * (fL - fR) ] dE
    fL = fermi_dirac(Energies, mu_L, T)
    fR = fermi_dirac(Energies, mu_R, T)
    
    integrand = Trans_V * (fL - fR)
    current_integral = np.trapezoid(integrand, Energies) # Numerical integration
    
    Currents[i] = (2 * qe / 6.626e-34) * current_integral * qe # Factor qe converts dE (eV) to Joules

    if i % 5 == 0:
        print(f"Calculated V = {Vb:.2f} V")

# ------------------------------------------------------------
# 5. Final I-V Plot
# ------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(Voltages, Currents, 'b-o', linewidth=2)
plt.title('I-V Characteristic of RTD')
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A)')
plt.grid(True)
plt.show()