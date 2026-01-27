import pickle

import numpy as np

from math import ceil, sin, cos, sqrt, pi
from itertools import combinations
from collections import defaultdict

def uniformRandomRotation():
    """
    Return a uniformly distributed rotation 3 x 3 matrix

    The initial description of the calculation can be found in the section 5 of "How to generate random matrices from
    the classical compact groups" of Mezzadri (PDF: https://arxiv.org/pdf/math-ph/0609050.pdf; arXiv:math-ph/0609050;
    and NOTICES of the AMS, Vol. 54 (2007), 592-604). Sample code is provided in that section as the ``haar_measure``
    function.

    Apparently this code can randomly provide flipped molecules (chirality-wise), so a fix found in
    https://github.com/tmadl/sklearn-random-rotation-ensembles/blob/5346f29855eb87241e616f6599f360eba12437dc/randomrotation.py
    was applied.

    Returns
    -------
    M : np.ndarray
        A uniformly distributed rotation 3 x 3 matrix
    """
    q, r = np.linalg.qr(np.random.normal(size=(3, 3)))
    M = np.dot(q, np.diag(np.sign(np.diag(r))))
    if np.linalg.det(M) < 0:  # Fixing the flipping
        M[:, 0] = -M[:, 0]  # det(M)=1
    return M

def rotate(coords, rotMat, center=(0,0,0)):
    """
    Rotate a selection of atoms by a given rotation around a center
    """

    newcoords = coords - center
    return np.dot(newcoords, np.transpose(rotMat)) + center

def rotation_matrix(axis, theta):
    """Counterclockwise rotation about a given axis by theta radians"""

    if not isinstance(axis, (np.ndarray, list, tuple)):
        raise TypeError('axis must be an array of floats of shape (3,)')
    try:
        # axis = np.asarray(axis, dtype=np.float)
        axis = np.array(axis).astype(np.float64)
    except ValueError:
        raise ValueError('axis must be an array of floats of shape (3,)')

    if axis.shape != (3,):
        raise ValueError('axis must be an array of floats of shape (3,)')

    if not isinstance(theta, (float, int)):
        raise TypeError('theta must be a float')

    axis = axis / sqrt(np.dot(axis, axis))
    a = cos(theta / 2.0)
    b, c, d = -axis * sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def rotation(lig_coords):
    rrot = uniformRandomRotation()
    lig_coords = rotate(lig_coords, rrot)
    return lig_coords

def get_all_rotation():
    # Create matrices for all possible 90* rotations of a box
    ROTATIONS = [rotation_matrix([1, 1, 1], 0)]

    # about X, Y and Z - 9 rotations
    for a1 in range(3):
        for t in range(1, 4):
            axis = np.zeros(3)
            axis[a1] = 1
            theta = t * pi / 2.0
            ROTATIONS.append(rotation_matrix(axis, theta))

    # about each face diagonal - 6 rotations
    for (a1, a2) in combinations(range(3), 2):
        axis = np.zeros(3)
        axis[[a1, a2]] = 1.0
        theta = pi
        ROTATIONS.append(rotation_matrix(axis, theta))
        axis[a2] = -1.0
        ROTATIONS.append(rotation_matrix(axis, theta))

    # about each space diagonal - 8 rotations
    for t in [1, 2]:
        theta = t * 2 * pi / 3
        axis = np.ones(3)
        ROTATIONS.append(rotation_matrix(axis, theta))
        for a1 in range(3):
            axis = np.ones(3)
            axis[a1] = -1
            ROTATIONS.append(rotation_matrix(axis, theta))
