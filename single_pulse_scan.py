# Atomic units
t0 = -24e-15/2.419e-17
t1 = 24e-15/2.419e-17
dt = 0.1e-18/2.419e-17

# --------------------------------------------------
# Worker function (no i anymore — fixed per outer loop)
# --------------------------------------------------
def compute_block(args):
    k, photon_energy, theta, phi, E_val, sigma, D_hat, Z = args

    dip_loc = np.sin(phi)*np.cos(theta)*dips[0] + np.sin(phi)*np.sin(theta)*dips[1] + np.cos(phi)*dips[2]

    D_vals, Z = np.linalg.eigh(dip_loc)
    D_hat = np.diag(D_vals)

    result = U_p(
        t0, t1, dt,
        photon_energy,
        sigma, E_val,
        D_hat, Z, energies
    )

    return k, result

dims = np.array([2])
photon_energies = np.linspace(520, 548, 32)/au2ev  
# sigma = 0.2e-15/2.419e-17
# Measured bandwidths to sample: 0.3298308241640277 1.6894796270170467 3.772467287185564
sigmas_ev = np.array([0.3298308241640277, 1.6894796270170467, 3.772467287185564])*np.sqrt(2)
# sigmas_ev = np.array([3.772467287185564])*np.sqrt(2)

# sigmas_ev = np.array([0.6660858818403884, 2.0152944566333453, 3.8001556586829706])
# phi_points = np.array([0.0])
# theta_points = np.array([0.0])
# sigmas_ev = np.array([0.6660858818403884])
sigmas_au = 0.44*2*np.pi/(sigmas_ev/au2ev)
# sigmas_au = np.array([0.2e-15/2.419e-17])
# sigmas_au = np.array([0.2e-15/2.419e-17, .4e-15/2.419e-17, .8e-15/2.419e-17])
# I_val = 1e20/au_2_wcm2
F_val = 227/0.15569/10 # Translates to 10 uJ
E_val = np.sqrt(I_val)


U_p_int_scan = np.zeros((3, len(sigmas_au), len(photon_energies), len(phi_points), len(energies), len(energies))).astype(np.complex64)


tic = time.time()

with mp.Pool(mp.cpu_count()) as pool:

    for i in dims:
        for j, sigma in enumerate(sigmas_au):
            # --------------------------------------------------
            # Compute eigen decomposition ONCE for this i
            # --------------------------------------------------
            I_val = F_val/(sigma/np.sqrt(2)*np.sqrt(2*np.pi))
            E_val = np.sqrt(I_val)
            # norm_factor = 1
            D_vals, Z = np.linalg.eigh(dips[i])
            D_hat = np.diag(D_vals)
    
            # If U_p_shaped depends on D_hat and Z,
            # they must be visible inside the worker.
            # Best practice: pass them explicitly (see note below).

            photon_energies_v, phi_points_v = np.meshgrid(photon_energies, phi_points)
            _, theta_points_v = np.meshgrid(photon_energies, theta_points)

            shape = photon_energies_v.shape

            photon_energies_v = photon_energies_v.flatten()
            phi_points_v = phi_points_v.flatten()
            theta_points_v = theta_points_v.flatten()
    
            # Build task list only over j
            tasks = [(k, photon_energy, phi, theta, E_val, sigma, D_hat, Z)
                     for k, (photon_energy, phi, theta) in enumerate(zip(photon_energies_v, phi_points_v, theta_points_v))]
    
            results = pool.map(compute_block, tasks)

            photon_energy_idx = np.arange(len(photon_energies))
            angle_idx = np.arange(len(phi_points))
            photon_energy_idx_v, angle_idx_v = np.meshgrid(photon_energy_idx, angle_idx)
            photon_energy_idx_v = photon_energy_idx_v.flatten()
            angle_idx_v = angle_idx_v.flatten()
    
            # Reassemble results for this i
            for k, result in results:
                p_idx = photon_energy_idx_v[k]
                a_idx = angle_idx_v[k]
                U_p_int_scan[i, j, p_idx, a_idx] = result

toc = time.time()
print("Elapsed time:", toc - tic)
