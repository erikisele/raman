from mpi4py import MPI
import numpy as np
import time
from raman_utils import sample_from_3d_hist, U_p_shaped, ionization_xc_pAp_bw_weighted_shaped
import h5py
import pickle

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

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def split_list(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

savefile = 'sample_bandwidths_blue_before_54_angles_sample_1.h5'

# Nitrogen dipole moments
file = "/sdf/home/i/isele/tmo100827624/results/erik/raman_calculation/meta_dipoles/input.rassi.h5"
with h5py.File(file, 'r') as hf:
    dips = hf['SFS_EDIPMOM'][:]
    energies_h5 = hf['SFS_ENERGIES'][:]

idcs = np.arange(dips.shape[1])

dips[:, idcs, idcs] = 0

dips_full = np.copy(dips)

D_tild = dips[2]

D_hat, Z = np.linalg.eigh(D_tild)
D_hat = np.diag(D_hat)

energies = energies_h5 - np.min(energies_h5)

energies = energies.astype(np.complex64)

auger_lifetime = 5000/au2as # lifetime in a.u.
energies[20:] = energies[20:] + -1.0j * 1/auger_lifetime

# Atomic units
# t0 = -24e-15/2.419e-17
# t1 = 24e-15/2.419e-17
# dt = 0.1e-18/2.419e-17

t0 = -8e-15/2.419e-17
t1 = 8e-15/2.419e-17
dt = 0.1e-18/2.419e-17

# --------------------------------------------------
# Worker function (unchanged)
# --------------------------------------------------
def compute_block(args):
    k, photon_energy, splitting, sigma_lower, sigma_upper, theta, phi, E_val, D_hat, Z, energies = args
    # D_hat and Z are precomputed once per angle outside the task list

    # blue before
    photon_energy1 = photon_energy
    photon_energy2 = photon_energy - (splitting)

    # red before
    # photon_energy1 = photon_energy - (splitting / 2)
    # photon_energy2 = photon_energy + (splitting / 2)

    # blue before
    result = U_p_shaped(
        t0, t1, dt,
        mu1, mu2, photon_energy1, photon_energy2,
        sigma_upper, sigma_lower, E_val, E_val,
        D_hat, Z, energies
    )

    # red before
    # result = U_p_shaped(
    #     t0, t1, dt,
    #     mu1, mu2, photon_energy1, photon_energy2,
    #     sigma_lower, sigma_upper, E_val, E_val,
    #     D_hat, Z, energies
    # )

    return k, result



photon_energies = np.linspace(390, 408, 20)/au2ev  


data = np.load('/sdf/data/lcls/ds/tmo/tmo101269225/results/erik/sigma_distribution_coherent_fitting_diff_filt_run226.npz')
bins_lower = data['bins_lower'] + np.diff(data['bins_lower'])[0]/2
bins_upper = data['bins_upper'] + np.diff(data['bins_upper'])[0]/2
hist = data['hist']

filename = '/sdf/home/i/isele/tmo101269225/results/erik/processed_data/bandwidth_splitting_histogram.pkl'
with open(filename, 'rb') as f:
    hist_3d = pickle.load(f)

nsamples = 200
samples = sample_from_3d_hist(hist_3d[221], hist_3d['bins'][0], hist_3d['bins'][1], hist_3d['bins'][2], nsamples)

splitting_samples = samples[:, 0]/au2ev
sigmas_au_lower = 0.44*2*np.pi/(samples[:, 1]/au2ev)
sigmas_au_upper = 0.44*2*np.pi/(samples[:, 2]/au2ev)

# splitting_samples = np.array([5.5])/au2ev
# sigmas_au_lower = 0.44*2*np.pi/(np.array([2.5])/au2ev)
# sigmas_au_upper = 0.44*2*np.pi/(np.array([2.5])/au2ev)

splitting = 5.5/au2ev
mu1 = -31
mu2 = 31

I_val = 1e16/au_2_wcm2
E_val = np.sqrt(I_val)

# phi_points = np.array([0.0])
# theta_points = np.array([0.0])

data = np.load('angle_points_54.npz')
theta_points = data['theta_points']
phi_points = data['phi_points']


# --------------------------------------------------
# Broadcast shared data
# --------------------------------------------------
# dims = comm.bcast(dims, root=0)
# photon_energies = comm.bcast(photon_energies, root=0)
# sigmas_au_lower = comm.bcast(sigmas_au_lower, root=0)
# sigmas_au_upper = comm.bcast(sigmas_au_upper, root=0)
# phi_points = comm.bcast(phi_points, root=0)
# theta_points = comm.bcast(theta_points, root=0)
# E_val = comm.bcast(E_val, root=0)

# --------------------------------------------------
# Main computation
# --------------------------------------------------
tic = time.time()

all_results = []


for j, (phi, theta) in enumerate(zip(phi_points, theta_points)):

    print('angle index: ', j)

    # Eigendecomposition for this angle — computed once, shared across all tasks at this angle
    dip_loc = np.sin(phi)*np.cos(theta)*dips[0] + np.sin(phi)*np.sin(theta)*dips[1] + np.cos(phi)*dips[2]
    D_vals_loc, Z_loc = np.linalg.eigh(dip_loc)
    D_hat_loc = np.diag(D_vals_loc)

    photon_energies_v, sigmas_au_lower_v = np.meshgrid(photon_energies, sigmas_au_lower)
    _, sigmas_au_upper_v = np.meshgrid(photon_energies, sigmas_au_upper)
    _, splitting_samples_v = np.meshgrid(photon_energies, splitting_samples)

    photon_energies_v = photon_energies_v.flatten()
    sigmas_au_lower_v = sigmas_au_lower_v.flatten()
    sigmas_au_upper_v = sigmas_au_upper_v.flatten()
    splitting_samples_v = splitting_samples_v.flatten()

    print('photon_energies_v: ', photon_energies_v)

    # Build full task list (only rank 0)
    if rank == 0:
        tasks = [(k, photon_energy, splitting_sample, sigma_lower, sigma_upper, phi, theta, E_val, D_hat_loc, Z_loc, energies)
                 for k, (photon_energy, splitting_sample, sigma_lower, sigma_upper)
                 in enumerate(zip(photon_energies_v, splitting_samples_v, sigmas_au_lower_v, sigmas_au_upper_v))]

        # Split tasks across ranks
        task_chunks = split_list(tasks, size)
    else:
        task_chunks = None

    # Scatter tasks
    # print('task_chucks len: ', len(task_chunks))
    local_tasks = comm.scatter(task_chunks, root=0)

    # Each rank computes its chunk
    local_results = [compute_block(task) for task in local_tasks]

    # Gather results
    gathered = comm.gather(local_results, root=0)

    # --------------------------------------------------
    # Reassemble on rank 0
    # --------------------------------------------------
    if rank == 0:
        results = [item for sublist in gathered for item in sublist]

        print('results length: ', len(results))

        photon_energy_idx = np.arange(len(photon_energies))
        sigma_idx = np.arange(len(sigmas_au_lower))

        photon_energy_idx_v, sigma_idx_v = np.meshgrid(photon_energy_idx, sigma_idx)
        photon_energy_idx_v = photon_energy_idx_v.flatten()
        sigma_idx_v = sigma_idx_v.flatten()

        # allocate output once
        if 'U_p_int_scan' not in globals():
            U_p_int_scan = np.zeros(
                (len(phi_points), len(photon_energies),
                 len(sigmas_au_lower), len(energies), len(energies)),
                dtype=np.complex64
            )

        for k, result in results:
            p_idx = photon_energy_idx_v[k]
            print('p_idx: ', p_idx)
            s_idx = sigma_idx_v[k]
            U_p_int_scan[j, p_idx, s_idx] = result
                
if rank==0:
    with h5py.File(f'{savefile}', 'w') as hf:
        hf.create_dataset('U_p_int_scan', data=U_p_int_scan)
        hf.create_dataset('splitting_samples', data=splitting_samples)
        hf.create_dataset('sigmas_au_lower', data=sigmas_au_lower)
        hf.create_dataset('sigmas_au_upper', data=sigmas_au_upper)
        hf.create_dataset('photon_energies', data=photon_energies)
        hf.create_dataset('I_val', data=np.array([I_val]))
        hf.create_dataset('phi_points', data=phi_points)
        hf.create_dataset('theta_points', data=theta_points)
        hf.create_dataset('t_axis', data=np.arange(t0, t1, dt))

toc = time.time()

if rank == 0:
    print("Elapsed time:", toc - tic)