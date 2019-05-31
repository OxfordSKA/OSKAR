#!/usr/bin/python
"""
Convert polarised CST element files to OSKAR scalar element pattern format.
"""

from __future__ import print_function
import sys
import numpy

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
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()
    X = []
    for line in lines:
        values = line.split()
        if not len(values) == 8:
            continue
        else:
            x_all = numpy.array(values, dtype=numpy.dtype('f8'))
            X.append(x_all)
    return numpy.array(X, dtype=numpy.dtype('f8'))


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
    # Load the CST element pattern data.
    X = load_cst_file(cst_file_in)
    # Only require columns for:
    # Theta, Phi, Abs(Theta), Phase(Theta), Abs(Phi), Phase(Phi)
    X = numpy.copy(X[:, [0, 1, 3, 4, 5, 6]])

    # Discard any data at values of phi >= 360 degrees,
    # as any duplicated entries will cause this method to fail.
    X = X[X[:, 1] < 360.0, :]

    # Generate the rotated data for Y from X by adding 90 degrees to the phi
    # values
    Y = numpy.copy(X)
    Y[:, 1] += 90.0
    Y[Y[:, 1] >= 360.0, 1] -= 360.0

    # Linked column sort by phi and then theta for both X and Y.
    X = X[numpy.lexsort((X[:, 0], X[:, 1])), :]
    Y = Y[numpy.lexsort((Y[:, 0], Y[:, 1])), :]

    # Check that the coordinate columns in X and Y now match.
    assert numpy.sum(numpy.abs(X[:, 0] - Y[:, 0])) < 1e-6
    assert numpy.sum(numpy.abs(X[:, 1] - Y[:, 1])) < 1e-6

    # Generate scalar values from sorted data.
    X_theta = X[:, 2] * numpy.exp(1j * numpy.radians(X[:, 3]))
    X_phi = X[:, 4] * numpy.exp(1j * numpy.radians(X[:, 5]))
    Y_theta = Y[:, 2] * numpy.exp(1j * numpy.radians(Y[:, 3]))
    Y_phi = Y[:, 4] * numpy.exp(1j * numpy.radians(Y[:, 5]))
    s = X_theta * numpy.conj(X_theta) + X_phi * numpy.conj(X_phi) + \
        Y_theta * numpy.conj(Y_theta) + Y_phi * numpy.conj(Y_phi)

    # Take the sqrt to convert to a 'voltage'
    s = numpy.sqrt(0.5 * s)
    s_amp = numpy.absolute(s)
    s_phase = numpy.angle(s, deg=True)

    # Write scalar values to file Columns = (theta, phi, amp, phase).
    o = numpy.column_stack((X[:, 0], X[:, 1], s_amp, s_phase))
    numpy.savetxt(scalar_file_out, o,
                  fmt=['%12.4f', '%12.4f', '%20.6e', '%12.4f'])

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: oskar_convert_cst_to_scalar.py "
              "<input CST file> <output scalar file>")
        sys.exit(1)

    convert(sys.argv[1], sys.argv[2])
