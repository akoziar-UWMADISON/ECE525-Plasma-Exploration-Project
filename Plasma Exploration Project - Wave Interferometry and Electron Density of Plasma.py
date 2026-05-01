import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.constants import c, e, eps0, m_e
from plasmapy.formulary import plasma_frequency
from scipy.integrate import simpson, trapezoid

def run_full_diagnostic_loop():
    # ---------------------------------------------------------
    # STEP 1: Define Physical Parameters
    # ---------------------------------------------------------
    a = 1.0 * u.m                          
    n_0 = 5e18 * u.m**-3                   
    probe_freq = 100 * u.GHz               
    omega_probe = 2 * np.pi * probe_freq * u.rad 
    
    # ---------------------------------------------------------
    # STEP 2 & 3: Spatial Grid and Density Profile
    # ---------------------------------------------------------
    x = np.linspace(-a.value, a.value, 500)
    y = np.linspace(-a.value, a.value, 500)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    
    n_e_value = n_0.value * (1 - (R / a.value)**2)
    n_e_value[R > a.value] = 0
    n_e = n_e_value * u.m**-3 
    
    # ---------------------------------------------------------
    # STEP 4 & 5: Compute True Phase Shift
    # ---------------------------------------------------------
    wp = plasma_frequency(n_e, particle='e-')
    ratio = (wp / omega_probe).decompose().value
    N = np.sqrt(1 - ratio**2)
    
    integrand = 1 - N
    integral_result = simpson(integrand, dx=dx, axis=1) * u.m 
    
    true_phase = ((omega_probe / c) * integral_result).decompose().value
    
    # ---------------------------------------------------------
    # STEP 6: Simulate Hardware (Phase Wrapping) & Software Fix
    # ---------------------------------------------------------
    wrapped_phase = np.angle(np.exp(1j * true_phase))
    unwrapped_phase = np.unwrap(wrapped_phase)
    unwrapped_phase = unwrapped_phase - unwrapped_phase[0] # Baseline correction

    # ---------------------------------------------------------
    # STEP 7: ABEL INVERSION (Reconstructing the Density)
    # ---------------------------------------------------------
    mid_idx = len(y) // 2
    y_half = y[mid_idx:]
    phase_half = unwrapped_phase[mid_idx:]
    
    # Force omega_probe into standard SI units to prevent the unit-stripping bug
    omega_si = omega_probe.to(u.rad / u.s).value
    
    K = (e.si.value**2) / (2 * c.value * eps0.value * m_e.value * omega_si)
    N_y = phase_half / K
    
    dN_dy = np.gradient(N_y, dy)
    
    r_eval = np.linspace(0, a.value * 0.98, 100) 
    n_e_recon = np.zeros_like(r_eval)
    
    for i, r in enumerate(r_eval):
        mask = y_half > r
        y_sub = y_half[mask]
        dN_dy_sub = dN_dy[mask]
        
        abel_integrand = dN_dy_sub / np.sqrt(y_sub**2 - r**2)
        n_e_recon[i] = -(1 / np.pi) * trapezoid(abel_integrand, y_sub)

    # ---------------------------------------------------------
    # Visualization (2x2 Grid)
    # ---------------------------------------------------------
    # Set up a 2x2 grid for the 4 plots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    ax1, ax2, ax3, ax4 = axs.flatten()
    
    # Plot 1: The 2D Plasma Density Cross-Section
    im = ax1.imshow(n_e.value, extent=[-a.value, a.value, -a.value, a.value], 
                    origin='lower', cmap='plasma')
    ax1.set_title('Plasma Density Cross-Section')
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m) - Chord Height')
    fig.colorbar(im, ax=ax1, label='Electron Density (m^-3)', fraction=0.046, pad=0.04)
    
    # Plot 2: The Raw Hardware Signal
    ax2.plot(y, wrapped_phase, color='red', linewidth=1.5)
    ax2.set_title('Raw Detector Signal (Wrapped)')
    ax2.set_xlabel('Chord Height (m)')
    ax2.set_ylabel('Phase Angle (Radians)')
    ax2.set_yticks([-np.pi, 0, np.pi])
    ax2.set_yticklabels(['$-\pi$', '0', '$+\pi$'])
    ax2.grid(True)
    
    # Plot 3: The Unwrapped Diagnostic Signal
    ax3.plot(y, unwrapped_phase, color='green', linewidth=2)
    ax3.set_title('Unwrapped Phase Shift')
    ax3.set_xlabel('Chord Height (m)')
    ax3.set_ylabel('Total Phase Shift (Radians)')
    ax3.grid(True)
    
    # Plot 4: The Abel Inversion Output vs. Reality
    original_core_density = n_0.value * (1 - (r_eval / a.value)**2)
    ax4.plot(r_eval, original_core_density, color='black', linestyle='--', linewidth=3, label='Actual Input Density')
    ax4.plot(r_eval, n_e_recon, color='blue', linewidth=1.5, label='Abel Reconstructed Density')
    ax4.set_title('Diagnostic Output: Density Profile')
    ax4.set_xlabel('Radial Distance from Core (m)')
    ax4.set_ylabel('Electron Density (m^-3)')
    ax4.legend()
    ax4.grid(True)
    
    # pad=3.0 ensures sufficient white space between the rows and columns
    plt.tight_layout(pad=3.0)
    plt.show()

if __name__ == "__main__":
    run_full_diagnostic_loop()