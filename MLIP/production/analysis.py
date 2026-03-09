"""RDF and MSD analysis for MLIP production MD trajectories.

Implements radial distribution function and mean squared displacement
computation using only numpy and ASE, avoiding external analysis packages.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid
from ase import Atoms
from ase.geometry import get_distances

import numpy as np
from ase import Atoms
from ase.geometry import get_distances

class idxFormatError(Exception):
    pass

def compute_rdf(images, idx1, idx2, exclude_idx=None, binwidth=0.1, rmax=None, surface=False):
    ''' Calculate radial distribution function.

        The reason that this exists even though ASE already
        has ase.geometry.rdf.get_rdf() is that the latter
        only accepts a single Atoms object. Also, this
        implementation can do g(z) for surface models.

        Parameters:

        images : obj
            ASE Atoms object containing the system or
            ASE Trajectory object containing a series of images.
            If a Trajectory is supplied, all images need to be
            of the same length and have the same atomic indexing.

        idx1, idx2 : str or list of int
            Atoms to use for the RDF calculation. Can be string
            (name of elements) or list of strings, or can be list
            of atomic indices (int) referring to the Atoms object.

        exclude_idx : int or list of int
            Atomic indices to be ignored used to refine selection
            made by idx1 and idx2.

        bindwidth : float
            The distance increments (in Angstrom) at which sphere
            segments will be evaluated.

        surface : bool
            If True, returns the g(z) instead of the g(r). Chose
            idx1 or idx2 appropriately to contain an atom within
            the surface that the g(z) should be in reference to.

        Return values:

        g_r : list of tuples
            Returns list containing the g(r) as (r, g(r))
            tuples.

        Usage example:

            >>> import matplotlib.pyplot as plt
            >>> from ase.build import molecule
            >>> from asetools.analysis.rdf import rdf
            >>>
            >>> # Set up periodic water cell.
            >>> atoms = molecule("H2O")
            >>> atoms.cell = [3, 3, 3]
            >>> atoms.center()
            >>> atoms = atoms.repeat((10, 10, 10))
            >>>
            >>> # Calculate RDF and visualize results.
            >>> g_r = rdf(atoms, "O", "O")
            >>> x, y = zip(*g_r)
            >>> plt.plot(x, y)
            >>> plt.xlabel("$r$ / $\mathrm{\AA}$")
            >>> plt.ylabel("$g(r)$")
            >>> plt.show()
    '''

    # Pick first image in case of trajectory to do setup with.
    if isinstance(images, Atoms):
        # If a singular Atoms object is passed, put into list
        # for consistency with the code below.
        images = [images]
    atoms = images[0]

    # If there is no cell, we need to add one based on the
    # dimensions of the system since we need a reference volume.
    if not atoms.cell:
        print("No cell vectors found, using model dimensions to",
              "estimate them to get the normalization volume.")
        # Translate so that all positions are positive.
        pos = atoms.get_positions()
        shiftmin = np.amin(atoms.get_positions(), axis=0)
        atoms.set_positions(atoms.get_positions() - shiftmin)

        # Determine largest x, y, z coordinates, make that the cell.
        atoms.cell = np.amax(atoms.get_positions(), axis=0)
    vol = atoms.get_volume()

    if exclude_idx != None:
        # Make exclude_idx into list if it isn't already.
        if not isinstance(exclude_idx, list):
            exclude_idx = [exclude_idx]
        # Make sure exclude_idx only contains int.
        if not all([isinstance(ei, int) for ei in exclude_idx]):
            raise idxFormatError(" ".join(
                ["Parameter exclude_idx must be of type int",
                 "or a list of int."]))

    # Assert that idx1/2 is either string or list of int.
    # If str (element symbol): expand to list of indices.
    indices = [[],[]]
    allElements = set(atoms.get_chemical_symbols())
    for i, idx in enumerate([idx1, idx2]):
        # Always convert to list for consistency.
        if not isinstance(idx, list):
            idx = [idx]
        # Expand index list based on int (index) or str (element).
        for id in idx:
            if isinstance(id, int):
                indices[i].append(id)
            elif isinstance(id, str):
                if not id in allElements:
                    raise idxFormatError(
                        " ".join(
                            ["The element symbol '{:s}'".format(id),
                             "you provided does not exist",
                             "in the Atoms object."]))
                else:
                    indices[i].extend([atom.index for atom in atoms
                                       if atom.symbol == id])
            else:
                raise idxFormatError(
                    " ".join(["You must either pass a string (elemental",
                              "symbol), list of str, int (atomic index),",
                              "list of int, or mixed int/str list",
                              "to this function."]))

    # Because a mixed list of indices and symbols can be provided,
    # there can be duplicates in the selection -> remove.
    # If exclude_idx was defined, we remove those from the lists.
    # Finally, we're also sanity-checking the selection here.
    for i, idx in enumerate(indices):
        if idx:
            if exclude_idx != None:
                idx = [id for id in idx if id not in exclude_idx]
            indices[i] = list(set(idx))
        else:
            raise idxFormatError(" ".join(["Index selection is empty.",
                                       "There is an issue with the",
                                       "choice of idx1 and idx2."]))
    idx1, idx2 = indices

    # Determine the number of unique atoms to be evaluated.
    ncnt = len(set(idx1 + idx2))
    n_target = len(set(idx1))

    # Determine number of bins from largest cell dimension and
    # binwidth parameter. Determine number of images in trajectory.
    cell = atoms.cell

    if surface:
        # Adjust cell z parameter internally for better
        # viewing (cell might be much higher than maximum
        # distances because of vacuum slab).
        pos = atoms.get_positions()
        maxz = np.max(pos)
        cell[2][2] = maxz + 10.0
        cellhmax = cell[2][2]
    else:
        cellh = cell / 2.0
        cellhmax = np.amax(cellh)
    
    if rmax is not None:
        cellhmax = rmax

    nbins = int(cellhmax / binwidth)
    nimages = len(images)

    # Set up collection bins.
    bins = np.array([0] * (nbins + 1))

    # Determine summation increment dr.
    dr = cellhmax / nbins

    # Create mask to remove self-interaction
    # (distance between the same atoms).
    combinations = []
    for id1 in idx1:
        combinations.extend([(id1, id2) for id2 in idx2])
    mask = np.asarray([False if pair[0] == pair[1] else True
                       for pair in combinations])

    # For each image in images, calculate pair distances
    # and assign to bins.
    for atoms in images:
        pos = atoms.get_positions()

        # In case of surface, reduce to g(z).
        if surface:
            pos *= [0,0,1]

        # Calculate all pair distances.
        _, dist = get_distances(pos[idx1], pos[idx2],
                                cell=cell, pbc=atoms.pbc)
        dist_flat = dist.flatten()

        # Remove self-interaction.
        dist_dr = dist_flat[mask] / dr

        dist_dr_int = dist_dr.astype(int) + 1
        bins += [np.count_nonzero(dist_dr_int == i + 1)
                 for i in range(nbins + 1)]

    # Normalize the g(r) and return list of (r, g(r) tuples.
    rho = ncnt / vol
    norm = 4 * np.pi * rho * n_target
    g_r = []
    for i, value in enumerate(bins):
        rr  = (i + 0.5) * dr
        value /= (norm * rr * rr * dr * nimages)
        g_r.append((rr, value))
    
    n_particles = len(images[0])
    volume = images[0].get_volume()
    number_density = n_particles / volume  # 1/Å³
    g_r = np.array(g_r)
    r, g = g_r[:, 0], g_r[:, 1]
    cn_integrand = 4 * np.pi * number_density * np.array(g) * np.array(r)**2
    n = cumulative_trapezoid(cn_integrand, r, initial=0)

    return r, g, n

def compute_rdf_old(images, species_i, species_j, rmax=8.0, binwidth=0.05):
    """Compute the radial distribution function g(r) from trajectory frames.

    Uses ASE's minimum-image-convention distance matrix and standard
    shell-volume normalization.

    Args:
        images: List of ASE Atoms objects (trajectory frames).
        species_i: Chemical symbol of reference species (e.g., "Li").
        species_j: Chemical symbol of target species (e.g., "S").
        rmax: Maximum distance in Angstrom.
        binwidth: Bin width in Angstrom.

    Returns:
        (r, g_r, n_r): Arrays of bin centers (Angstrom), g(r) values,
        and running coordination number n(r).
    """
    nbins = int(rmax / binwidth)
    r_edges = np.linspace(0, rmax, nbins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    hist = np.zeros(nbins)

    same_species = (species_i == species_j)

    # Reference counts from first frame (constant for NVT)
    symbols_ref = np.array(images[0].get_chemical_symbols())
    idx_i_ref = np.where(symbols_ref == species_i)[0]
    idx_j_ref = np.where(symbols_ref == species_j)[0]
    n_i = len(idx_i_ref)
    n_j = len(idx_j_ref)

    for atoms in images:
        symbols = np.array(atoms.get_chemical_symbols())
        idx_i = np.where(symbols == species_i)[0]
        idx_j = np.where(symbols == species_j)[0]

        # Full distance matrix with minimum image convention
        dist_matrix = atoms.get_all_distances(mic=True)
        sub = dist_matrix[np.ix_(idx_i, idx_j)]

        if same_species:
            # Upper triangle only (avoid double counting and self-pairs)
            triu = np.triu_indices_from(sub, k=1)
            distances = sub[triu]
        else:
            distances = sub.ravel()

        h, _ = np.histogram(distances[distances < rmax], bins=r_edges)
        hist += h

    # Normalize
    n_frames = len(images)
    V = images[0].get_volume()

    # Shell volumes
    shell_vol = (4.0 / 3.0) * np.pi * (r_edges[1:] ** 3 - r_edges[:-1] ** 3)

    # Number density of target species
    rho_j = n_j / V

    if same_species:
        # For identical species, exclude self-interactions
        rho_target = (n_i - 1) / V
    else:
        rho_target = rho_j

    # Ideal gas: expected count per shell
    # g(r) = h(r) / (n_frames * n_i * rho_target * shell_vol)
    ideal = n_i * rho_target * shell_vol

    with np.errstate(divide="ignore", invalid="ignore"):
        g_r = np.where(ideal > 0, hist / (n_frames * ideal), 0.0)

    # Running coordination number: n(r) = integral of 4*pi*rho*g(r)*r^2 dr
    # Use rho_target to properly account for same_species case
    integrand = 4.0 * np.pi * rho_target * g_r * r_centers ** 2
    n_r = np.zeros_like(r_centers)
    n_r[1:] = cumulative_trapezoid(integrand, r_centers)

    return r_centers, g_r, n_r


def compute_msd(images, species, timestep_fs, traj_interval):
    """Compute mean squared displacement using FFT autocorrelation.

    Uses the Calandrini/Knuth algorithm for efficient O(N log N) MSD
    computation. PBC unwrapping is done via fractional coordinate
    differences with nearest-image correction.

    Args:
        images: List of ASE Atoms objects (trajectory frames).
        species: Chemical symbol of diffusing species (e.g., "Li").
        timestep_fs: MD timestep in femtoseconds.
        traj_interval: Trajectory write interval in MD steps.

    Returns:
        (dt_fs, msd, msd_xyz):
            dt_fs: Time lag array in femtoseconds.
            msd: Total MSD in Angstrom^2.
            msd_xyz: Per-component MSD, shape (N_frames, 3).
    """
    # Identify species indices
    symbols = np.array(images[0].get_chemical_symbols())
    idx = np.where(symbols == species)[0]
    n_atoms = len(idx)
    n_frames = len(images)

    # Collect fractional positions
    frac = np.zeros((n_frames, n_atoms, 3))
    for i, atoms in enumerate(images):
        frac[i] = atoms.get_scaled_positions()[idx]

    # Unwrap PBC via fractional coordinate differences
    dfrac = np.diff(frac, axis=0)
    dfrac -= np.round(dfrac)
    unwrapped_frac = np.zeros_like(frac)
    unwrapped_frac[0] = frac[0]
    unwrapped_frac[1:] = frac[0] + np.cumsum(dfrac, axis=0)

    # Convert to Cartesian
    cell = np.array(images[0].get_cell())
    unwrapped = np.einsum("ijk,kl->ijl", unwrapped_frac, cell)

    # Compute MSD via FFT autocorrelation (per atom, per component)
    msd_xyz = np.zeros((n_frames, 3))
    m_arr = np.arange(n_frames)
    counts = (n_frames - m_arr).astype(float)

    for atom_i in range(n_atoms):
        for d in range(3):
            x = unwrapped[:, atom_i, d]
            N = len(x)

            # S2: autocorrelation <x(t+m) * x(t)> via FFT
            F = np.fft.rfft(x, n=2 * N)
            acf = np.fft.irfft(F * np.conj(F))[:N]
            acf /= counts

            # S1: <x(t+m)^2 + x(t)^2>
            x2 = x ** 2
            cumsum = np.zeros(N + 1)
            cumsum[1:] = np.cumsum(x2)
            s1 = (cumsum[N] - cumsum[m_arr] + cumsum[N - m_arr]) / counts

            # MSD(m) = S1(m) - 2 * S2(m)
            msd_xyz[:, d] += s1 - 2 * acf

    # Average over atoms
    msd_xyz /= n_atoms
    msd = np.sum(msd_xyz, axis=1)

    # Time array in femtoseconds
    dt_per_frame_fs = timestep_fs * traj_interval
    dt_fs = m_arr * dt_per_frame_fs

    return dt_fs, msd, msd_xyz


def compute_rdf_mae(r_pred, gr_pred, r_gt, gr_gt):
    """Compute MAE of g(r) over the overlapping r range.

    Interpolates prediction onto ground truth grid where they overlap.

    Args:
        r_pred, gr_pred: Predicted RDF arrays.
        r_gt, gr_gt: Ground truth RDF arrays.

    Returns:
        MAE value (float), or None if no overlap.
    """
    r_min = max(r_pred[0], r_gt[0])
    r_max = min(r_pred[-1], r_gt[-1])
    if r_max <= r_min:
        return None

    mask_gt = (r_gt >= r_min) & (r_gt <= r_max)
    r_overlap = r_gt[mask_gt]
    gr_gt_overlap = gr_gt[mask_gt]

    gr_pred_interp = np.interp(r_overlap, r_pred, gr_pred)
    return float(np.mean(np.abs(gr_pred_interp - gr_gt_overlap)))


def compute_msd_mae(t_pred, msd_pred, t_gt, msd_gt):
    """Compute MAE of MSD over the overlapping time range.

    Both time arrays should be in the same units (femtoseconds).
    Interpolates prediction onto ground truth grid where they overlap.

    Args:
        t_pred, msd_pred: Predicted MSD arrays.
        t_gt, msd_gt: Ground truth MSD arrays.

    Returns:
        MAE value (float), or None if no overlap.
    """
    t_min = max(t_pred[0], t_gt[0])
    t_max = min(t_pred[-1], t_gt[-1])
    if t_max <= t_min:
        return None

    mask_gt = (t_gt >= t_min) & (t_gt <= t_max)
    t_overlap = t_gt[mask_gt]
    msd_gt_overlap = msd_gt[mask_gt]

    msd_pred_interp = np.interp(t_overlap, t_pred, msd_pred)
    return float(np.mean(np.abs(msd_pred_interp - msd_gt_overlap)))


def compute_diffusivity(t_fs, msd, fit_start_frac=0.2, fit_end_frac=0.8):
    """Extract diffusion coefficient from linear fit of MSD vs time.

    D = slope / (2 * d), where d=3 for 3D diffusion, and
    slope = d(MSD)/dt.

    Args:
        t_fs: Time array in femtoseconds.
        msd: Total MSD array in Angstrom^2.
        fit_start_frac: Start of fitting window as fraction of total time.
        fit_end_frac: End of fitting window as fraction of total time.

    Returns:
        D in cm^2/s, or None if fitting fails.
    """
    t_max = t_fs[-1]
    if t_max <= 0:
        return None

    mask = (t_fs >= fit_start_frac * t_max) & (t_fs <= fit_end_frac * t_max)
    t_fit = t_fs[mask]
    msd_fit = msd[mask]

    if len(t_fit) < 2:
        return None

    # Linear fit: MSD = slope * t + intercept
    slope, _ = np.polyfit(t_fit, msd_fit, 1)

    # slope is in Angstrom^2 / fs
    # D = slope / (2 * 3) in Angstrom^2 / fs
    # Convert to cm^2/s: 1 Angstrom^2/fs = 1e-16 cm^2 / 1e-15 s = 0.1 cm^2/s
    D_cm2_s = slope / 6.0 * 1e-16 / 1e-15  # = slope / 6.0 * 0.1
    return float(D_cm2_s)


def mae_to_score(mae):
    """Convert MAE to a 0-1 score: score = 1 / (1 + MAE).

    Maps MAE in (0, inf) to score in (1, 0), so lower MAE = higher score.
    Perfect prediction (MAE=0) gives score=1.
    """
    return 1.0 / (1.0 + mae)
