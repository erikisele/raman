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

pulse_type = 'blue_before'

if not (pulse_type=='red_before' or pulse_type=='blue_before'):
    raise ValueError("Invalid pulse type")

# Nitrogen dipole moments
file = "/sdf/home/i/isele/tmo100827624/results/erik/raman_calculation/ANO-S_calcs/mAp_nitrogen/input.rassi.h5"
with h5py.File(file, 'r') as hf:
    dips_original = hf['SFS_EDIPMOM'][:]
    energies_h5_original = hf['SFS_ENERGIES'][:]

t0 = -8e-15/2.419e-17
t1 = 8e-15/2.419e-17
dt = 0.1e-18/2.419e-17

# --------------------------------------------------
# Worker function (unchanged)
# --------------------------------------------------
def compute_block(args):
    k, photon_energy, splitting, sigma_lower, sigma_upper, theta, phi, E_val, D_hat, Z, energies = args

    if pulse_type=='red_before':
        photon_energy2 = photon_energy
        photon_energy1 = photon_energy - (splitting)
    else: # blue before
        photon_energy1 = photon_energy
        photon_energy2 = photon_energy - (splitting)

    if pulse_type=='red_before':
        result = U_p_shaped(
            t0, t1, dt,
            mu1, mu2, photon_energy1, photon_energy2,
            sigma_lower, sigma_upper, E_val, E_val,
            D_hat, Z, energies
        )
    else: # blue before
        result = U_p_shaped(
            t0, t1, dt,
            mu1, mu2, photon_energy1, photon_energy2,
            sigma_upper, sigma_lower, E_val, E_val,
            D_hat, Z, energies
        )

    return k, result


photon_energies = np.linspace(392, 420, 40)/au2ev

data = np.load('/sdf/data/lcls/ds/tmo/tmo101269225/results/erik/sigma_distribution_coherent_fitting_diff_filt_run226.npz')
bins_lower = data['bins_lower'] + np.diff(data['bins_lower'])[0]/2
bins_upper = data['bins_upper'] + np.diff(data['bins_upper'])[0]/2
hist = data['hist']

filename = '/sdf/home/i/isele/tmo101269225/results/erik/processed_data/bandwidth_splitting_histogram.pkl'
with open(filename, 'rb') as f:
    hist_3d = pickle.load(f)

nsamples = 400
if pulse_type=='red_before':
    samples = sample_from_3d_hist(hist_3d[226], hist_3d['bins'][0], hist_3d['bins'][1], hist_3d['bins'][2], nsamples)
else:
    samples = sample_from_3d_hist(hist_3d[221], hist_3d['bins'][0], hist_3d['bins'][1], hist_3d['bins'][2], nsamples)

splitting_samples = samples[:, 0]/au2ev
sigmas_au_lower = 0.44*2*np.pi/(samples[:, 1]/au2ev)
sigmas_au_upper = 0.44*2*np.pi/(samples[:, 2]/au2ev)

splitting = 5.5/au2ev
mu1 = -44.24
mu2 = 44.24

I_val = 1e16/au_2_wcm2
E_val = np.sqrt(I_val)

phi_points = np.array([0.0])
theta_points = np.array([0.0])

auger_lifetime = 5000/au2as # lifetime in a.u.

# --------------------------------------------------
# Scan over each valence excited state (indices 1-19)
# --------------------------------------------------
for val_state in range(1, 20):

    savefile = f'sample_bandwidths_{pulse_type}_version_6_I_1e16_valence_state_{val_state}.h5'

    states_selected = np.concatenate((np.array([0]), np.array([val_state]), np.arange(20, 40, 1)))
    dips = dips_original[np.ix_(np.arange(3), states_selected, states_selected)]
    energies_h5 = energies_h5_original[states_selected]

    idcs = np.arange(dips.shape[1])
    dips[:, idcs, idcs] = 0

    energies = energies_h5 - np.min(energies_h5)
    energies = energies.astype(np.complex64)
    energies[2:] = energies[2:] + -1.0j * 1/auger_lifetime  # core states start at index 2 (0=ground, 1=valence)

    # --------------------------------------------------
    # Main computation
    # --------------------------------------------------
    tic = time.time()

    if rank == 0:
        U_p_int_scan = np.zeros((len(phi_points), len(photon_energies),
                                 nsamples, len(energies), len(energies)),
                                 dtype=np.complex64)

    for j, (phi, theta) in enumerate(zip(phi_points, theta_points)):

        if rank == 0:
            print(f'val_state {val_state}, angle index: {j}')

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

        if rank == 0:
            tasks = [(k, photon_energy, splitting_sample, sigma_lower, sigma_upper, phi, theta, E_val, D_hat_loc, Z_loc, energies)
                     for k, (photon_energy, splitting_sample, sigma_lower, sigma_upper)
                     in enumerate(zip(photon_energies_v, splitting_samples_v, sigmas_au_lower_v, sigmas_au_upper_v))]
            task_chunks = split_list(tasks, size)
        else:
            task_chunks = None

        local_tasks = comm.scatter(task_chunks, root=0)
        local_results = [compute_block(task) for task in local_tasks]
        gathered = comm.gather(local_results, root=0)

        if rank == 0:
            results = [item for sublist in gathered for item in sublist]

            photon_energy_idx = np.arange(len(photon_energies))
            sigma_idx = np.arange(len(sigmas_au_lower))

            photon_energy_idx_v, sigma_idx_v = np.meshgrid(photon_energy_idx, sigma_idx)
            photon_energy_idx_v = photon_energy_idx_v.flatten()
            sigma_idx_v = sigma_idx_v.flatten()

            for k, result in results:
                p_idx = photon_energy_idx_v[k]
                s_idx = sigma_idx_v[k]
                U_p_int_scan[j, p_idx, s_idx] = result

    if rank == 0:
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
            hf.create_dataset('energies', data=energies)
            hf.create_dataset('states_selected', data=states_selected)
            hf.create_dataset('val_state', data=np.array([val_state]))

        toc = time.time()
        print(f"val_state {val_state} elapsed time: {toc - tic:.1f}s")
