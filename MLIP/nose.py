import numpy as np

from ase import units, Atoms


def nose_mass(temperature, ndof, t0, L):
    '''
    Suggested Q:
        Shuichi Nosé, J. Chem. Phys., 81, 511(1984).
    input:
    temperaute: in unit of Kelvin
    ndof: No. of degrees of freedom
    t0: The oscillation time in fs
    L: the length of the first basis vector
    '''

    # Q in Energy * Times**2
    qtmp = (t0 * 1E-15 / np.pi / 2)**2 * \
        2 * ndof * units.kB * temperature \
        * units._e

    # Q in eV * fs**2
    q_eV_fs = qtmp*(1E+15*units.fs)**2/units._e

    # Q in AMU * Angstrom**2
    Q = qtmp / units._amu / (L * 1E-10)**2

    return q_eV_fs


def cnt_dof(atoms):
    '''
    Count No. of Degrees of Freedom
    '''
    if atoms.constraints:
        from ase.constraints import FixAtoms, FixScaled, FixedPlane, FixedLine
        sflags = np.zeros((len(atoms), 3), dtype=bool)
        for constr in atoms.constraints:
            if isinstance(constr, FixScaled):
                sflags[constr.a] = constr.mask
            elif isinstance(constr, FixAtoms):
                sflags[constr.index] = [True, True, True]
            elif isinstance(constr, FixedPlane):
                mask = np.all(np.abs(np.cross(constr.dir, atoms.cell)) < 1e-5,
                              axis=1)
                if sum(mask) != 1:
                    raise RuntimeError(
                        'VASP requires that the direction of FixedPlane '
                        'constraints is parallel with one of the cell axis')
                sflags[constr.a] = mask
            elif isinstance(constr, FixedLine):
                mask = np.all(np.abs(np.cross(constr.dir, atoms.cell)) < 1e-5,
                              axis=1)
                if sum(mask) != 1:
                    raise RuntimeError(
                        'VASP requires that the direction of FixedLine '
                        'constraints is parallel with one of the cell axis')
                sflags[constr.a] = ~mask

        return np.sum(~sflags)
    else:
        return len(atoms) * 3 - 3


def get_nose_q(atoms: Atoms, unit="fs", temperature=300, frequency=40):

    if unit == 'cm-1':
        THzToCm = 33.3564095198152
        t0 = 1000 * THzToCm / frequency
    else:
        t0 = frequency

    L    = np.linalg.norm(atoms.cell, axis=1)[0]
    ndof = cnt_dof(atoms)
    Q    = nose_mass(temperature, ndof, t0, L)

    #print("SMASS = {}".format(Q))
    return Q

