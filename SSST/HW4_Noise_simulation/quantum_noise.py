import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# PHYSICAL CONSTANTS & PARAMETERS
# -----------------------------------------------------------------------------
Dx = 1e-10              # Spatial step (1 Angstrom)
L_device = 2e-8         # Simulation Box length
m = 0.063 * 9.109e-31   # Effective electron mass (kg)
h_bar = 1.054e-34       # Reduced Planck constant (J·s)
qe = 1.6e-19            # Electron charge (C)
Kb = 1.38e-23           # Boltzmann constant (J/K)
T = 298                 # Temperature (K)
Area = 1e-12            # Cross-sectional area (m^2)
Ef = 0.1              # Fermi energy (eV)

# Theoretical Landauer conductance G0
G0 = qe**2 / (np.pi * h_bar)

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def get_transmission(Energy, Potential):
    """
    Calculates Transmission T(E) using the Numerov Method for a given Potential profile.
    From HW3: RTD simulation

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

def compute_current(Voltages, Energies, U_base, T):
    """
    Computes Mean Current (I) for a given potential profile U_base and temperature T.
    """
    Currents = np.zeros(len(Voltages))
    
    # Define active region indices
    N_pts = len(U_base)
    start_active = int(0.25 * N_pts)
    end_active = int(0.75 * N_pts)
    points_active = end_active - start_active
    # Linear slope drop:
    points_slope = np.arange(points_active)
    fraction = (points_slope / points_active)
    
    for i, Vb in enumerate(Voltages):
        # 1. Tilt potential
        U_bias = U_base.copy()
        
        # Right reservoir drops by Vb
        U_bias[end_active:] = U_bias[end_active:] - (Vb * qe)
        
        # Voltage drop across active region
        voltage_drop = (Vb * qe) * fraction
        U_bias[start_active:end_active] -= voltage_drop
        
        # 2. Compute T(E, V)
        T = get_transmission(Energies, U_bias)
        
        # 3. Compute Current Integral
        mu_L = Ef
        mu_R = Ef - Vb
        
        # LOW TEMPERATURE (step function approx)
        if T < 1e-6:
            # At T~0, Fermi-Dirac becomes a step function 
            # s.t. f_L (f_R) = 1 for E < mu_L (mu_R) and 0 otherwise
            # Current is integral of T(E) between mu_R and mu_L
            # Find indices where E is between mu_R and mu_L
            idx_window = np.where((Energies >= mu_R) & (Energies <= mu_L))[0]
            if len(idx_window) > 0:
                integral = np.trapezoid(T[idx_window], Energies[idx_window]) * qe
            else:
                integral = 0

        # FINITE TEMPERATURE
        else:
            arg_L = (Energies - mu_L) * qe / (Kb * T)
            arg_R = (Energies - mu_R) * qe / (Kb * T)
            
            # Fermi Functions
            fL = 1.0 / (1.0 + np.exp(arg_L))
            fR = 1.0 / (1.0 + np.exp(arg_R))
            
            integrand = T * (fL - fR)
            integral = np.trapezoid(integrand, Energies) * qe
        
        # I = (e / pi hbar) * Integral
        Currents[i] = (qe / (np.pi * h_bar)) * integral

    return Currents

def compute_noise(Voltages, Energies, U_base, T):
    """
    Computes Noise Power Spectral Density (S) 
    for a given potential profile U_base and temperature T.
    """
    Noise_S = np.zeros(len(Voltages))
    
    # Define active region indices
    N_pts = len(U_base)
    start_active = int(0.25 * N_pts)
    end_active = int(0.75 * N_pts)
    points_active = end_active - start_active
    points_slope = np.arange(points_active)
    fraction = (points_slope / points_active)
    
    for i, Vb in enumerate(Voltages):
        # 1. Tilt potential
        U_bias = U_base.copy()
        # Right reservoir drops by Vb
        U_bias[end_active:] = U_bias[end_active:] - (Vb * qe)
        # Voltage drop across active region
        voltage_drop = (Vb * qe) * fraction
        U_bias[start_active:end_active] -= voltage_drop
        
        # 2. Compute T(E, V)
        T = get_transmission(Energies, U_bias)
        
        # 3. Compute Noise Integral
        # Integrand: T * [fL(1-fR) + fR(1-fL)] - T^2 * (fL - fR)^2
        mu_L = Ef
        mu_R = Ef - Vb
        
        # LOW TEMPERATURE (step function approx)
        if T < 1e-6:
            idx_window = np.where((Energies >= mu_R) & (Energies <= mu_L))[0]
            if len(idx_window) > 0:
                # Noise Integrand: T * (1 - T)
                integrand =T[idx_window] * (1 - T[idx_window])
                integral = np.trapezoid(integrand, Energies[idx_window]) * qe
            else:
                integral = 0
        # FINITE TEMPERATURE
        else:
            arg_L = (Energies - mu_L) * qe / (Kb * T)
            arg_R = (Energies - mu_R) * qe / (Kb * T)
            
            # Fermi Functions
            fL = 1.0 / (1.0 + np.exp(arg_L))
            fR = 1.0 / (1.0 + np.exp(arg_R))
            
            # Noise Integrand
            integrand = T * (fL * (1 - fR) - fR * (1 - fL)) - T**2 * (fL - fR)**2
            integral = np.trapezoid(integrand, Energies) * qe
        
        # S = (2e^2 / pi hbar) * Integral
        Noise_S[i] = (2.0 * qe**2 / (np.pi * h_bar)) * integral

    return Noise_S

# -----------------------------------------------------------------------------
# PART 1: LANDAUER CONDUCTANCE
# -----------------------------------------------------------------------------

# Set up (perfect wire, no barrier)
xpoints = int(np.floor(L_device / Dx))
U_wire = np.zeros(xpoints) # Flat potential V=0 everywhere

# Energy Grid
E_max = 0.2
E_steps = 2000
Energies = np.linspace(0.001, E_max, E_steps)

# Voltage Sweep (Very small bias for Conductance limit)
V_max = 0.005 # 5 mV
V_steps = 60
Voltages = np.linspace(1e-6, V_max, V_steps) # Avoid exactly 0 to prevent div by zero

# Case A: Ideal Case (T -> 0 K, No Barrier)
print("Simulating Case A: T -> 0 K, No Barrier...")
I_ideal = compute_current(Voltages, Energies, U_wire, T=1e-9)
G_ideal = I_ideal / Voltages

# Case B: Room Temperature (T = 298 K, No Barrier)
print("Simulating Case B: T = 298 K, No Barrier...")
I_room = compute_current(Voltages, Energies, U_wire, T=298)
G_room = I_room / Voltages

# Case C: With Barrier (T -> 0 K)
print("Simulating Case C: T -> 0 K, With Barrier...")
# Define a small barrier
U_barrier = np.zeros(xpoints)
center = xpoints // 2
width = int(4e-9 / Dx)
U_barrier[center - width//2 : center + width//2] = 0.3 * qe # 0.3 eV barrier
I_bar = compute_current(Voltages, Energies, U_barrier, T=1e-9)
G_bar = I_bar / Voltages

# Plot G/G0 vs Voltage
plt.figure(figsize=(10, 8))
plt.axhline(y=1.0, color='k', linestyle='--')
plt.plot(Voltages, G_ideal / G0, 'b-o', label='Ideal')
plt.plot(Voltages, G_room / G0, 'r-s', label='Room Temp')
plt.plot(Voltages, G_bar / G0, 'g-^', label='Barrier')

plt.xlabel('Voltage (V)')
plt.ylabel(r'$G/G_0$')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# -----------------------------------------------------------------------------
# PART 2: FANO FACTOR AND EQUILIBRIUM NOISE
# -----------------------------------------------------------------------------
print("\nPart 2: Fano Factor and Equilibrium Noise")

# Set up: Double Barrier of 0.4 eV
U_double = np.zeros(xpoints)
W_barrier = 4e-9
W_well = 10e-9
V0 = 0.4

