#!/usr/bin/python
"""
Convert polarised CST element files to OSKAR scalar element pattern format.
"""


def load_cst_file(filename):
    """"
    Loads a CST element pattern file into a numpy matrix.

    Parameters
    ----------
    filename : string
        Path of the CST element pattern file to load.

    Returns
    -------
    Matrix of values from the CST file.
    """

    import numpy as np

    f = open(filename, 'r')
    lines = f.readlines()
    f.close()
    X = []
    for line in lines:
        values = line.split()
        if not len(values) == 8:
            continue
        else:
            x_all = np.array(values, dtype=np.double)
            X.append(x_all)
    return np.array(X, dtype=np.double)


def convert(cst_file_in, scalar_file_out):
    """
    Calculates a scalar element pattern file from a CST element pattern file

    Parameters
    ----------
    cst_file_in : string
        Input CST format element pattern file
    scalar_file_out : string
        Output scalar format element pattern file

    Notes
    -----
    This function is designed to be used to create scalar element input files
    for the oskar_fit_element_data application.
    """

    import numpy as np

    # Load the CST element pattern data for X. (Ignore lines that don't consist
    # of 8 floats)
    X = load_cst_file(cst_file_in)
    # Only require a columns for:
    # Theta, Phi, Abs(Theta), Phase(Theta), Abs(Phi), Phase(Phi)
    X = np.copy(X[:, [0, 1, 3, 4, 5, 6]])

    # Generate the rotated data for Y from X by adding 90 degrees to the phi
    # values
    Y = np.copy(X)
    Y[:, 1] += 90.0
    Y[Y[:, 1] >= 360.0, 1] -= 360.0

    # Linked column sort by phi and then theta for both X and Y.
    X = X[np.lexsort((X[:, 1], X[:, 0])), :]
    Y = Y[np.lexsort((Y[:, 1], Y[:, 0])), :]
    assert(np.sum(X[:, 0] == Y[:, 0]) == len(X[:, 0]))
    assert(np.sum(X[:, 1] == Y[:, 1]) == len(X[:, 1]))

    # Generate scalar values from sorted data.
    X_theta = X[:, 2] * np.exp(1j * X[:, 3] * np.pi / 180.0)
    X_phi = X[:, 4] * np.exp(1j * X[:, 5] * np.pi / 180.0)
    Y_theta = Y[:, 2] * np.exp(1j * Y[:, 3] * np.pi / 180.0)
    Y_phi = Y[:, 4] * np.exp(1j * Y[:, 5] * np.pi / 180.0)
    s = X_theta * np.conj(X_theta) + X_phi * np.conj(X_phi) + \
        Y_theta * np.conj(Y_theta) + Y_phi * np.conj(Y_phi)

    # Take the sqrt to convert to a 'voltage'
    s = np.sqrt(0.5 * s)
    s_amp = np.absolute(s)
    s_phase = np.angle(s, deg=True)

    # Write scalar values to file Columns = (theta, phi, amp, phase).
    o = np.column_stack((X[:, 0], X[:, 1], s_amp, s_phase))
    np.savetxt(scalar_file_out, o, fmt=['%12.4f', '%12.4f', '%20.6e',
                                        '%12.4f'])

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print "Usage: oskar_convert_cst_to_scalar.py " \
            "<input CST file> <output scalar file>"
        sys.exit(1)

    filename = sys.argv[1]
    outname = sys.argv[2]

    convert(filename, outname)
