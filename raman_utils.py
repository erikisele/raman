import numpy as np
from matplotlib import pyplot as plt
import xraydb
import h5py
import time
import multiprocessing as mp


# CGS units
# h = 4.135667696e-15
# hbar = h/(2*np.pi)
# alpha = 1/137.035999177
# dipole_au2cm = 2.541765 * 3.33564e-30/1.602e-19 * 1e2 # 1 au = 2.541765 D, 1 D = 3.33564e-30 C*m, 1 cm = 1e-2 m 
# q = 1
# C = 1.60218e-19

# Atomic units
h = 2*np.pi
hbar = 1
alpha = 1/137.035999177
dipole_au2cm = 2.541765 * 3.33564e-30/1.602e-19 * 1e2 # 1 au = 2.541765 D, 1 D = 3.33564e-30 C*m, 1 cm = 1e-2 m 
q = 1
C = 1
esp0 = 1/(4*np.pi)

au2ev = 27.211
au2as = 24.419
au_2_wcm2 = 3.509e16

ionization_calc_offset = 4.177 # this is a fudge factor that aligns the edge with the x-ray data base and the calculations being used


# photon energy points

def ionization_xc_element(element, energy):
    mu_elam = xraydb.mu_elam(element, energy)
    rho = xraydb.atomic_density(element)
    xc = mu_elam*rho
    return xc

def ionization_xc_pAp(energy):
    elements = {'C': 6, 'O': 1, 'N': 1, 'H': 7}
    xc = 0
    for element, count in elements.items():
        mu_elam = xraydb.mu_elam(element, energy)
        mass = xraydb.atomic_mass(element)
        xc += mu_elam * mass * count / 6.0221408e+23
    return xc

photon_energy_lut = np.arange(350, 600, 0.1)
xc_lut = np.zeros(len(photon_energy_lut))
for i in range(len(xc_lut)):
    xc_lut[i] = ionization_xc_pAp(float(photon_energy_lut[i]))

def ionization_xc_pAp_interp(energy):
    if (energy < photon_energy_lut[0]) or (energy > photon_energy_lut[-1]):
        raise Exception(f"Photon energy out of range of interpolation values. Photon energy: {energy}.")

    return np.interp(energy, photon_energy_lut, xc_lut)

def ionization_xc_pAp_interp_shaped(energy1, energy2):
    energy = (energy1+energy2)/2
    if (energy < photon_energy_lut[0]) or (energy > photon_energy_lut[-1]):
        raise Exception("Photon energy out of range of interpolation values")

    return np.interp(energy, photon_energy_lut, xc_lut)

def ionization_xc_pAp_bw_weighted(photon_energy_eV, sigma_t_au):
    """Bandwidth-weighted ionization cross section for a Gaussian x-ray pulse.

    Weights xc(E) by the pulse power spectrum exp(-(E-E0)^2 * (sigma_t/au2ev)^2),
    which is the Fourier-transform power spectrum of exp(-t^2/(2*sigma_t^2))*cos(omega_0*t).
    Uses the precomputed LUT for efficiency — O(N) dot product, no xraydb calls.

    photon_energy_eV : central photon energy [eV]
    sigma_t_au       : temporal 1/e half-width of the envelope [atomic units]
    """
    weights = np.exp(-((photon_energy_lut - (photon_energy_eV+ionization_calc_offset)) * ( sigma_t_au / (au2ev * 0.44*2*np.pi))) ** 2) # note no factor of two because this uses the electric field RMS duration to calculate the bandwidth of the intensity
    return np.dot(weights, xc_lut) / weights.sum()

def ionization_xc_pAp_bw_weighted_shaped(photon_energy1_eV, photon_energy2_eV, sigma1_t_au, sigma2_t_au):
    """Bandwidth-weighted ionization cross section for a two-component shaped Gaussian pulse.

    Total spectral weight is the incoherent sum of the two pulse power spectra,
    valid when the two center frequencies are separated by more than their bandwidths.
    Uses the precomputed LUT for efficiency — O(N) dot product, no xraydb calls.

    photon_energy1_eV, photon_energy2_eV : center energies [eV]
    sigma1_t_au, sigma2_t_au             : temporal 1/e half-widths [atomic units]
    """
    w1 = np.exp(-((photon_energy_lut - (photon_energy1_eV + ionization_calc_offset)) * ( sigma1_t_au / (au2ev * 0.44*2*np.pi))) ** 2) # note no factor of two because this uses the electric field RMS duration to calculate the bandwidth of the intensity
    w2 = np.exp(-((photon_energy_lut - (photon_energy2_eV+ionization_calc_offset)) * ( sigma2_t_au / (au2ev * 0.44*2*np.pi))) ** 2)
    weights = w1 + w2
    return np.dot(weights, xc_lut) / weights.sum()

def I2(t0, t1, sigma, E, energy):
    dt_loc = t1 - t0
    I2_int = (np.abs(E_field(t0, sigma, E, energy))**2 + E_field(t1, sigma, E, energy)**2)/2 * dt_loc
    return I2_int

def I2_shaped(t0, t1, mu1, mu2, sigma1, sigma2, E1, E2, energy1, energy2):
    dt_loc = t1 - t0
    I2_int = (np.abs(E_field_shaped(t0, mu1, mu2, sigma1, sigma2, E1, E2, energy1, energy2))**2 + np.abs(E_field_shaped(t1, mu1, mu2, sigma1, sigma2, E1, E2, energy1, energy2))**2)/2 * dt_loc
    return I2_int

def I1(t0, t1, sigma, E, energy):
    dt_loc = t1 - t0
    I1_int = (E_field(t0, sigma, E, energy) + E_field(t1, sigma, E, energy))/2 * dt_loc
    return I1_int

def I1_shaped(t0, t1, mu1, mu2, sigma1, sigma2, E1, E2, energy1, energy2):
    dt_loc = t1 - t0
    I1_int = (E_field_shaped(t0, mu1, mu2, sigma1, sigma2, E1, E2, energy1, energy2) + E_field_shaped(t1, mu1, mu2, sigma1, sigma2, E1, E2, energy1, energy2))/2 * dt_loc
    return I1_int

def E_field(t, sigma, E, energy):
    return E*np.exp(-(t**2)/(2*sigma**2))*np.cos(energy*t/hbar)

def E_field_shaped(t, mu1, mu2, sigma1, sigma2, E1, E2, energy1, energy2):
    E_field_1 = E1*np.exp(-(t-mu1)**2/(2*sigma1**2))*np.cos(energy1*t/hbar)
    E_field_2 = E2*np.exp(-(t-mu2)**2/(2*sigma2**2))*np.cos(energy2*t/hbar)
    return E_field_1 + E_field_2

def eV2omega(eV):
    return 2*np.pi*eV/h

def U0(t0_loc, t1_loc, energies):
    return np.exp(-1.0j * energies * (t1_loc - t0_loc) / hbar)

# def U1(t0_loc, t1_loc, photon_energy, sigma, E):
#     U0_loc = U0(t0_loc, t1_loc)
#     eye = np.ones(U0_loc.shape[0])
#     omega = eV2omega(photon_energy)
#     # xc_factor = 3*ionization_xc_pAp(photon_energy)/(4*np.pi**2*alpha*omega)
#     xc_factor = ionization_xc_pAp(photon_energy)
#     U1_loc = np.diag(np.exp(-eye*xc_factor*I2(t0_loc, t1_loc, sigma, E, photon_energy) / C)) @ U0_loc
#     return U0_loc
#     # return U1_loc

def U1_au(t0_loc, t1_loc, photon_energy, sigma, E, D_hat, Z, energies):
    return U0(t0_loc, t1_loc, energies)

def U1_au_shaped(t0_loc, t1_loc, mu1, mu2, photon_energy1, photon_energy2, sigma1, sigma2, E1, E2, D_hat, Z, energies):
    u0_diag = U0(t0_loc, t1_loc, energies)
    xc_factor = ionization_xc_pAp_bw_weighted_shaped(photon_energy1*27.2114, photon_energy2*27.2114, sigma1, sigma2)
    xc_scale = xc_factor * (1.0 / (5.29177e-9)**2) / ((photon_energy1 + photon_energy2) / 2.0)
    i2 = I2_shaped(t0_loc, t1_loc, mu1, mu2, sigma1, sigma2, E1, E2, photon_energy1, photon_energy2)
    return np.exp(-xc_scale * i2) * u0_diag

def U2(t0_loc, t1_loc, photon_energy, sigma, E, D_hat, Z, verbose=True):
    I1_loc = I1(t0_loc, t1_loc, sigma, E, photon_energy)
    phase = np.exp(-1.0j * np.diag(D_hat) * I1_loc * q / hbar)
    U2_loc = (Z * phase) @ Z.T.conj()
    if np.any(np.isnan(U2_loc)):
        print("nan in U2_loc")
    return U2_loc

def U2_shaped(t0_loc, t1_loc, mu1, mu2, photon_energy1, photon_energy2, sigma1, sigma2, E1, E2, D_hat, Z, verbose=True):
    I1_loc = I1_shaped(t0_loc, t1_loc, mu1, mu2, sigma1, sigma2, E1, E2, photon_energy1, photon_energy2)
    phase = np.exp(-1.0j * np.diag(D_hat) * I1_loc * q / hbar)
    U2_loc = (Z * phase) @ Z.T.conj()
    if np.any(np.isnan(U2_loc)):
        print("nan in U2_loc")
    return U2_loc

def U_tilde(t0_loc, t1_loc, photon_energy, sigma, E, D_hat, Z, energies):
    dt_loc = t1_loc - t0_loc
    U2_local = U2(t0_loc, t1_loc, photon_energy, sigma, E, D_hat, Z)
    u1_diag = U1_au(t0_loc + dt_loc/2, t1_loc + dt_loc/2, photon_energy, sigma, E, D_hat, Z, energies)
    return u1_diag[:, None] * U2_local

def U_tilde_shaped(t0_loc, t1_loc, mu1, mu2, photon_energy1, photon_energy2, sigma1, sigma2, E1, E2, D_hat, Z, energies):
    dt_loc = t1_loc - t0_loc
    U2_local = U2_shaped(t0_loc, t1_loc, mu1, mu2, photon_energy1, photon_energy2, sigma1, sigma2, E1, E2, D_hat, Z)
    u1_diag = U1_au_shaped(t0_loc + dt_loc/2, t1_loc + dt_loc/2, mu1, mu2, photon_energy1, photon_energy2, sigma1, sigma2, E1, E2, D_hat, Z, energies)
    return u1_diag[:, None] * U2_local

def U_p_shaped(t0_loc, t1_loc, dt, mu1, mu2, photon_energy1, photon_energy2, sigma1, sigma2, E1, E2, D_hat, Z, energies):
    d_hat_diag = np.diag(D_hat)

    # Compute time-independent quantities once instead of once per timestep (~160k steps)
    xc_factor = ionization_xc_pAp_bw_weighted_shaped(
        photon_energy1 * au2ev, photon_energy2 * au2ev, sigma1, sigma2)
    xc_scale = xc_factor * (1.0 / (5.29177e-9)**2) / ((photon_energy1 + photon_energy2) / 2.0)
    u0_dt = np.exp(-1.0j * energies * dt)  # U0 diagonal for inner steps (span = dt, constant)

    def _u1_diag(t0, t1):
        i2 = I2_shaped(t0, t1, mu1, mu2, sigma1, sigma2, E1, E2, photon_energy1, photon_energy2)
        return np.exp(-xc_scale * i2) * np.exp(-1.0j * energies * (t1 - t0))

    def _u_tilde(t0_s):
        I1_loc = I1_shaped(t0_s, t0_s + dt, mu1, mu2, sigma1, sigma2, E1, E2, photon_energy1, photon_energy2)
        phase = np.exp(-1.0j * d_hat_diag * I1_loc)
        U2_loc = (Z * phase) @ Z.conj().T
        i2_mid = I2_shaped(t0_s + dt/2, t0_s + 1.5*dt, mu1, mu2, sigma1, sigma2, E1, E2, photon_energy1, photon_energy2)
        u1_loc = np.exp(-xc_scale * i2_mid) * u0_dt
        return u1_loc[:, None] * U2_loc  # diag(u1) @ U2 without materializing NxN diagonal

    u1_init = _u1_diag(t0_loc, t1_loc + dt/2)
    time_steps = np.arange(t0_loc, t1_loc, dt)

    if len(time_steps) == 0:
        U_p_loc = np.diag(u1_init)
    else:
        # First step: fuse with initial diagonal via column scaling (avoids one O(N^3) matmul)
        U_p_loc = _u_tilde(time_steps[0]) * u1_init[None, :]
        for t0_s in time_steps[1:]:
            U_p_loc = _u_tilde(t0_s) @ U_p_loc

    u1_final = _u1_diag(t1_loc, t1_loc + dt/2)
    return u1_final[:, None] * U_p_loc

def sample_from_2d_hist(H, xedges, yedges, n_samples):
    """
    Sample points from a 2D histogram.

    Parameters
    ----------
    H : 2D array
        Histogram counts or probabilities (shape: [nx, ny])
    xedges : 1D array
        Bin edges along x (length nx+1)
    yedges : 1D array
        Bin edges along y (length ny+1)
    n_samples : int
        Number of samples to generate

    Returns
    -------
    samples : (n_samples, 2) array
        Sampled (x, y) points
    """

    # Normalize histogram to probabilities
    P = H.astype(float)
    P /= P.sum()

    # Flatten to 1D distribution
    P_flat = P.ravel()

    # Sample bin indices
    flat_indices = np.random.choice(len(P_flat), size=n_samples, p=P_flat)

    # Convert back to 2D indices
    ix, iy = np.unravel_index(flat_indices, H.shape)

    # Sample uniformly inside each bin
    x = xedges[ix] + (xedges[ix+1] - xedges[ix]) * np.random.rand(n_samples)
    y = yedges[iy] + (yedges[iy+1] - yedges[iy]) * np.random.rand(n_samples)

    return np.column_stack((x, y))

import numpy as np

def sample_from_3d_hist(H, xedges, yedges, zedges, n_samples):
    """
    Sample points from a 3D histogram.

    Parameters
    ----------
    H : 3D array
        Histogram counts or probabilities (shape: [nx, ny, nz])
    xedges : 1D array
        Bin edges along x (length nx+1)
    yedges : 1D array
        Bin edges along y (length ny+1)
    zedges : 1D array
        Bin edges along z (length nz+1)
    n_samples : int
        Number of samples to generate

    Returns
    -------
    samples : (n_samples, 3) array
        Sampled (x, y, z) points
    """

    # Normalize histogram to probabilities
    P = H.astype(float)
    total = P.sum()
    if total == 0:
        raise ValueError("Histogram is empty (sum = 0).")
    P /= total

    # Flatten to 1D distribution
    P_flat = P.ravel()

    # Sample flat indices
    flat_indices = np.random.choice(len(P_flat), size=n_samples, p=P_flat)

    # Convert back to 3D indices
    ix, iy, iz = np.unravel_index(flat_indices, H.shape)

    # Sample uniformly inside each bin
    x = xedges[ix] + (xedges[ix+1] - xedges[ix]) * np.random.rand(n_samples)
    y = yedges[iy] + (yedges[iy+1] - yedges[iy]) * np.random.rand(n_samples)
    z = zedges[iz] + (zedges[iz+1] - zedges[iz]) * np.random.rand(n_samples)

    return np.column_stack((x, y, z))

def pulse_fluence_integral(mu1, mu2, sigma1, sigma2, E1, E2, energy1, energy2):
    """
    Returns int |E(t)|^2 dt for the two-color Gaussian pulse defined in
    E_field_shaped. To get fluence per unit area, multiply by c*eps_0
    (in atomic units, I_au = E_au^2, so multiplying by 1 gives intensity
    in atomic units; conversion to W/cm^2 is via au_2_wcm2 = 3.509e16).
    Using units and conventions from:
    https://onlinelibrary.wiley.com/doi/pdf/10.1002/3527605606.app9
    """
    omega1 = energy1 / hbar
    omega2 = energy2 / hbar

    # Self terms
    self1 = 0.5 * np.sqrt(np.pi) * E1**2 * sigma1
    self2 = 0.5 * np.sqrt(np.pi) * E2**2 * sigma2

    # Cross term
    Sigma2 = sigma1**2 + sigma2**2
    sigma_eff_sq = sigma1**2 * sigma2**2 / Sigma2
    sigma_eff = np.sqrt(sigma_eff_sq)
    mu_bar = (mu1 * sigma2**2 + mu2 * sigma1**2) / Sigma2
    domega = omega1 - omega2

    cross = (E1 * E2 * np.sqrt(2 * np.pi) * sigma_eff
             * np.exp(-(mu1 - mu2)**2 / (2 * Sigma2))
             * np.exp(-domega**2 * sigma_eff_sq / 2)
             * np.cos(domega * mu_bar))

    return self1 + self2 + cross