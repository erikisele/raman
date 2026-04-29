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

au2ev = 27.211
au2as = 24.419
au_2_wcm2 = 3.509e16


# photon energy points

def ionization_xc_element(element, energy):
    mu_elam = xraydb.mu_elam(element, energy)
    rho = xraydb.atomic_density(element)
    xc = mu_elam*rho
    return xc

def ionization_xc_pAp(energy):
    elements = {'C': 6, 'O': 1, 'N': 1, 'H': 7}
    xc = 0
    for element in elements.keys():
        for i in range(elements[element]):
            mu_elam = xraydb.mu_elam(element, energy)
            mass = xraydb.atomic_mass(element)
            xc += mu_elam*mass/6.0221408e+23

    return xc

photon_energy_lut = np.arange(350, 600)
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

def ionization_xc_element(element, energy):
    mu_elam = xraydb.mu_elam(element, energy)
    rho = xraydb.atomic_density(element)
    xc = mu_elam*rho
    return xc

def ionization_xc_pAp(energy):
    elements = {'C': 6, 'O': 1, 'N': 1, 'H': 7}
    xc = 0
    for element in elements.keys():
        for i in range(elements[element]):
            mu_elam = xraydb.mu_elam(element, energy)
            rho = xraydb.atomic_density(element)
            mass = xraydb.atomic_mass(element)
            xc += mu_elam*mass/6.0221408e+23

    return xc

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
    dt_loc = t1_loc - t0_loc
    return np.diag(np.exp(-1.0j*energies*dt_loc/hbar)) 

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
    U0_loc = U0(t0_loc, t1_loc, energies)
    eye = np.ones(U0_loc.shape[0])
    omega = eV2omega(photon_energy)
    # xc_factor = 3*ionization_xc_pAp(photon_energy)/(4*np.pi**2*alpha*omega)
    xc_factor = ionization_xc_pAp_interp(photon_energy*27.2114)
    U1_loc = np.diag(np.exp(-eye*xc_factor*I2(t0_loc, t1_loc, sigma, E, photon_energy) * (1/(5.29177e-9)**2) / photon_energy)) @ U0_loc
    return U0_loc
    # return U1_loc

def U1_au_shaped(t0_loc, t1_loc, mu1, mu2, photon_energy1, photon_energy2, sigma1, sigma2, E1, E2, D_hat, Z, energies):
    U0_loc = U0(t0_loc, t1_loc, energies)
    eye = np.ones(U0_loc.shape[0])
    # xc_factor = 3*ionization_xc_pAp(photon_energy)/(4*np.pi**2*alpha*omega)
    xc_factor = ionization_xc_pAp_interp_shaped(photon_energy1*27.2114, photon_energy2*27.2114)
    U1_loc = np.diag(np.exp(-eye*xc_factor*I2_shaped(t0_loc, t1_loc, mu1, mu2, sigma1, sigma2, E1, E2, photon_energy1, photon_energy2) * (1/(5.29177e-9)**2) / ((photon_energy1 + photon_energy2)/2))) @ U0_loc
    # return U0_loc
    return U1_loc

def U2(t0_loc, t1_loc, photon_energy, sigma, E, D_hat, Z, verbose=True):
    # if verbose: print(f"E value: {E}")
    I1_loc = I1(t0_loc, t1_loc, sigma, E, photon_energy)
    U2_loc = Z @ np.diag(np.exp(-1.0j * np.diag(D_hat) * I1_loc * q / hbar)) @ Z.T.conj()
    # print(I1_loc)
    if np.any(np.isnan(U2_loc)):
        print("nan in U2_loc")
    return U2_loc

def U2_shaped(t0_loc, t1_loc, mu1, mu2, photon_energy1, photon_energy2, sigma1, sigma2, E1, E2, D_hat, Z, verbose=True):
    # if verbose: print(f"E value: {E}")
    I1_loc = I1_shaped(t0_loc, t1_loc, mu1, mu2, sigma1, sigma1, E1, E2, photon_energy1, photon_energy2)
    U2_loc = Z @ np.diag(np.exp(-1.0j * np.diag(D_hat) * I1_loc * q / hbar)) @ Z.T.conj()
    # print(I1_loc)
    if np.any(np.isnan(U2_loc)):
        print("nan in U2_loc")
    return U2_loc

def U_tilde(t0_loc, t1_loc, photon_energy, sigma, E, D_hat, Z, energies):
    dt_loc = t1_loc - t0_loc
    U2_local = U2(t0_loc, t1_loc, photon_energy, sigma, E, D_hat, Z)
    U1_local = U1_au(t0_loc + dt_loc/2, t1_loc + dt_loc/2, photon_energy, sigma, E, D_hat, Z, energies)
    return U1_local @ U2_local

def U_tilde_shaped(t0_loc, t1_loc, mu1, mu2, photon_energy1, photon_energy2, sigma1, sigma2, E1, E2, D_hat, Z, energies):
    dt_loc = t1_loc - t0_loc
    U2_local = U2_shaped(t0_loc, t1_loc, mu1, mu2, photon_energy1, photon_energy2, sigma1, sigma2, E1, E2, D_hat, Z)
    U1_local = U1_au_shaped(t0_loc + dt_loc/2, t1_loc + dt_loc/2, mu1, mu2, photon_energy1, photon_energy2, sigma1, sigma2, E1, E2, D_hat, Z, energies)
    return U1_local @ U2_local

def U_p_shaped(t0_loc, t1_loc, dt, mu1, mu2, photon_energy1, photon_energy2, sigma1, sigma2, E1, E2, D_hat, Z, energies):
    U_p_loc = U1_au_shaped(t0_loc, t1_loc+dt/2, mu1, mu2, photon_energy1, photon_energy2, sigma1, sigma2, E1, E2, D_hat, Z, energies)
        
    for i, t0_step in enumerate(np.arange(t0_loc, t1_loc, dt)):
        t1_step = t0_step + dt
        U_tilde_loc = U_tilde_shaped(t0_step, t1_step, mu1, mu2, photon_energy1, photon_energy2, sigma1, sigma2, E1, E2, D_hat, Z, energies)
        U_p_loc = U_tilde_loc @ U_p_loc
    
    U_p_loc = U1_au_shaped(t1_loc, t1_loc+dt/2, mu1, mu2, photon_energy1, photon_energy2, sigma1, sigma2, E1, E2, D_hat, Z, energies) @ U_p_loc
    return U_p_loc

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