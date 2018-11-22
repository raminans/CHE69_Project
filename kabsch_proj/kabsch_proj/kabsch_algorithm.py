"""
kabsch_algorithm.py
The Kabsch algorithm, is a method for calculating the optimal rotation matrix that minimizes
the RMSD (root mean squared deviation) between two paired sets of points.

I added two paired sets of points (mol1.xyz and mol2.xyz) to the directory. In order to test and run the code.
You can run the code by:
    $   python kabsch_algorithm.py --mol1='../../mol1.xyz' --mol2='../../mol2.xyz'
to get the least RMSD.

Also, you can run the code with two flags (--normal and --rotation)
    --normal:       to get the normal RMSD of the two paired sets
    --rotation:     to get the optimal rotation matrix that minimizes the RMSD between two paired sets of points.

    $   python kabsch_algorithm.py --mol1='../../mol1.xyz' --mol2='../../mol2.xyz' --normal=True --rotation=True

"""

import sys
import argparse
import numpy as np


def parse_xyz(fname):
    """Parses .xyz file."""
    try:
        with open(fname, 'r') as foo:
            data = foo.read()
    except FileNotFoundError as e:
        print('{} does not exist!'.format(fname))
        raise e
    mol_data = []
    for line in data.split('\n'):
        splitted = line.split(' ' * 6)
        if len(splitted) == 4:
            mol_data.append([])
            for i in range(1, 4):
                mol_data[-1].append(float(splitted[i]))
    return np.array(mol_data)


# def warning(*objs):
#     """Writes a message to stderr."""
#     print("WARNING: ", *objs, file=sys.stderr)



def kabsch(P, Q, normal=False, rotation=False):
    """
     Find the Least Root Mean Square distance
     between two sets of N points in D dimensions
     and the rigid transformation (i.e. translation and rotation)
     to employ in order to bring one set that close to the other,
     Using the Kabsch algorithm.
     Note that the points are paired, i.e. we know which point in one set
     should be compared to a given point in the other set.

     P (& Q) are N*D matrices where P(i,a) (or Q) is the a-th coordinate of the i-th point in the 1st(2nd) representation

     The kabsch algorithm works in three steps
     1. computation of r : a D-dimensional column vector, representing the translation
     2. Computation of the covariance matrix C
     3. computation of U : a proper orthogonal D*D matrix, representing the rotation
     4. computation of lrms: the Least Root Mean Square
    """

    C = np.dot(np.transpose(P), Q)  #Computation of the covariance matrix C

    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
    if(d):
        S[-1] = -S[-1]
        V[:,-1] = -V[:,-1]

    U = np.dot(V, W)     # Create Rotation matrix U
    P_rotated = np.dot(P, U)     # Rotate P
    output = (rmsd(P_rotated,Q),)
    if normal:
        output = output + (rmsd(P,Q), )
    if rotation:
        output = output + (U, )
    return output


def centroid(X):
    """ Calculate the centroid from a vectorset X """
    if len(X):
        C = sum(X)/len(X)
    else:
        raise ValueError('Empty vector!')
    return C


def rmsd(V, W):
    """ Calculate Root-mean-square deviation from two sets of vectors V and W.
    """
    assert V.shape[-1] == W.shape[-1], 'Dimensions are not equal!'
    D = len(V[0])
    N = len(V)
    rmsd = 0.0
    for v, w in zip(V, W):
        rmsd += sum([(v[i]-w[i])**2.0 for i in range(D)])
    return np.sqrt(rmsd/N)


def parse_arguments(argv):
    """Parses command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--mol1', type=str,
                        help='Address to first molecule .xyz file')
    parser.add_argument('--mol2', type=str,
                        help='Address to second molecule .xyz file')
    parser.add_argument('--normal', type=bool,
                        help='Returns normal rmsd in addition.',
                        default=False)
    parser.add_argument('--rotation', type=bool,
                        help='If True returns rotation matrix.',
                        default=False)
    return parser.parse_args(argv)


def main(args):

    mol1 = parse_xyz(args.mol1)
    mol2 = parse_xyz(args.mol2)
    output = kabsch(mol1, mol2, normal=args.normal, rotation=args.rotation)
    #print('Min Distance = ', output[0])
    return output  # success


if __name__ == "__main__":
    status = main(parse_arguments(sys.argv[1:]))
    sys.exit(status)
