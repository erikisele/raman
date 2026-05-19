from mpi4py import MPI
import numpy as np
import time
from raman_utils import U_p
import h5py

# Atomic units
h = 2*np.pi
hbar = 1
alpha = 1/137.035999177
dipole_au2cm = 2.541765 * 3.33564e-30/1.602e-19 * 1e2
q = 1
C = 1

au2ev = 27.211
au2as = 24.419
au_2_wcm2 = 3.509e16

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def split_list(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]


# Nitrogen dipole moments
file = "/sdf/home/i/isele/tmo100827624/results/erik/raman_calculation/input.rassi.oxygen.h5"
with h5py.File(file, 'r') as hf:
    dips = hf['SFS_EDIPMOM'][:]
    energies_h5 = hf['SFS_ENERGIES'][:]

idcs = np.arange(dips.shape[1])
dips[:, idcs, idcs] = 0

energies = energies_h5 - np.min(energies_h5)
energies = energies.astype(np.complex64)

auger_lifetime = 5000/au2as
energies[20:] = energies[20:] + -1.0j * 1/auger_lifetime

# Wider time window for single pulse centered at t=0
t0 = -24e-15/2.419e-17
t1 = 24e-15/2.419e-17
dt = 0.1e-18/2.419e-17

# --------------------------------------------------
# Worker function
# --------------------------------------------------
def compute_block(args):
    k, photon_energy, E_val, sigma, D_hat, Z = args

    result = U_p(
        t0, t1, dt,
        photon_energy,
        sigma, E_val,
        D_hat, Z, energies
    )

    return k, result

# Scan parameters (from single_pulse_scan.py)
dims = np.array([2])
photon_energies = np.linspace(520, 548, 32)/au2ev

# Bandwidths in eV (FWHM), converted to temporal sigma in atomic units
sigmas_ev = np.array([0.3298308241640277, 1.6894796270170467, 3.772467287185564]) * np.sqrt(2)
sigmas_au = 0.44*2*np.pi/(sigmas_ev/au2ev)

# Fluence value (translates to 10 uJ at the given geometry)
F_val = 227/0.15569/10 / 2

savefile = f'single_pulse_scan_mpi_f_val_{F_val}_half_normal_energy.h5'

data = np.load('/sdf/home/i/isele/tmo100827624/results/erik/raman_calculation/angle_points_28.npz')
theta_points = data['theta_points']
phi_points = data['phi_points']

# --------------------------------------------------
# Precompute E_val for each sigma (fluence → peak intensity)
# For E(t) = E0*exp(-t²/(2σ²))*cos(ωt), matches single_pulse_scan.py convention.
I_vals = F_val / (sigmas_au / np.sqrt(2) * np.sqrt(2*np.pi))
E_vals = np.sqrt(I_vals)

# Index arrays for reassembling results — constant across all loops
photon_energy_idx_v, sigma_idx_v = np.meshgrid(
    np.arange(len(photon_energies)), np.arange(len(sigmas_au))
)
photon_energy_idx_v = photon_energy_idx_v.flatten()
sigma_idx_v = sigma_idx_v.flatten()

# Flat arrays of (photon_energy, E_val, sigma) over the full (sigma × photon_energy) grid
photon_energies_v, E_vals_v = np.meshgrid(photon_energies, E_vals)
_, sigmas_v = np.meshgrid(photon_energies, sigmas_au)
photon_energies_v = photon_energies_v.flatten()
E_vals_v = E_vals_v.flatten()
sigmas_v = sigmas_v.flatten()

# --------------------------------------------------
# Main computation
# --------------------------------------------------
tic = time.time()

if rank == 0:
    U_p_int_scan = np.zeros(
        (len(dims), len(sigmas_au), len(photon_energies), len(phi_points), len(energies), len(energies)),
        dtype=np.complex64
    )

for di, i in enumerate(dims):
    for ai, (phi, theta) in enumerate(zip(phi_points, theta_points)):

        # Eigendecomposition done once per angle, shared across all (sigma, photon_energy) tasks
        dip_loc = (np.sin(phi)*np.cos(theta)*dips[0]
                   + np.sin(phi)*np.sin(theta)*dips[1]
                   + np.cos(phi)*dips[2])
        D_vals_loc, Z_loc = np.linalg.eigh(dip_loc)
        D_hat_loc = np.diag(D_vals_loc)

        if rank == 0:
            tasks = [(k, photon_energy, E_val, sigma, D_hat_loc, Z_loc)
                     for k, (photon_energy, E_val, sigma)
                     in enumerate(zip(photon_energies_v, E_vals_v, sigmas_v))]
            task_chunks = split_list(tasks, size)
        else:
            task_chunks = None

        local_tasks = comm.scatter(task_chunks, root=0)
        local_results = [compute_block(task) for task in local_tasks]
        gathered = comm.gather(local_results, root=0)

        if rank == 0:
            results = [item for sublist in gathered for item in sublist]

            for k, result in results:
                p_idx = photon_energy_idx_v[k]
                s_idx = sigma_idx_v[k]
                U_p_int_scan[di, s_idx, p_idx, ai] = result

toc = time.time()

if rank == 0:
    with h5py.File(savefile, 'w') as hf:
        hf.create_dataset('U_p_int_scan', data=U_p_int_scan)
        hf.create_dataset('photon_energies', data=photon_energies)
        hf.create_dataset('sigmas_au', data=sigmas_au)
        hf.create_dataset('sigmas_ev', data=sigmas_ev)
        hf.create_dataset('dims', data=dims)
        hf.create_dataset('phi_points', data=phi_points)
        hf.create_dataset('theta_points', data=theta_points)
        hf.create_dataset('F_val', data=np.array([F_val]))
        hf.create_dataset('t_axis', data=np.arange(t0, t1, dt))
        hf.create_dataset('energies', data=energies)

    print("Elapsed time:", toc - tic)
