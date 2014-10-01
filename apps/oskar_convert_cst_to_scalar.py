#!/usr/bin/python

def convert(cst_file_in, scalar_file_out):
    """
    convert(cst_file_in, scalar_file_out)
    
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

    # Load the CST element pattern data for X.
    X = np.loadtxt(cst_file_in, skiprows=2, usecols=(0, 1, 3, 4, 5, 6))

    # Generate the rotated data for Y from X.
    Y = np.copy(X)
    Y[:, 1] += 90.0
    Y[Y[:, 1] >= 360.0, 1] -= 360.0

    # Linked column sort by phi and then theta for both X and Y.
    X = np.sort(X.view('f8,f8,f8,f8,f8,f8'), \
        order=['f1','f0'], axis=0).view(np.double)
    Y = np.sort(Y.view('f8,f8,f8,f8,f8,f8'), \
        order=['f1','f0'], axis=0).view(np.double)

    # Generate scalar values from sorted data.
    X_theta = X[:, 2] * np.exp(1j * X[:, 3] * np.pi / 180.0)
    X_phi   = X[:, 4] * np.exp(1j * X[:, 5] * np.pi / 180.0)
    Y_theta = Y[:, 2] * np.exp(1j * Y[:, 3] * np.pi / 180.0)
    Y_phi   = Y[:, 4] * np.exp(1j * Y[:, 5] * np.pi / 180.0)
    s = X_theta * np.conj(X_theta) + X_phi * np.conj(X_phi) + \
        Y_theta * np.conj(Y_theta) + Y_phi * np.conj(Y_phi)

    # Take the sqrt to convert to a 'voltage'
    s = np.sqrt(0.5 * s)

    s_amp   = np.absolute(s)
    s_phase = np.angle(s, deg=True)

    # Write scalar values to file.
    o = np.column_stack((X[:, 0], X[:, 1], s_amp, s_phase))
    np.savetxt(scalar_file_out, o, fmt=['%12.4f', '%12.4f', '%20.6e', '%12.4f'])

if __name__ == "__main__":
    import numpy as np
    import sys

    if len(sys.argv) < 3:
        print "Usage: oskar_convert_cst_to_scalar.py " \
            "<input CST file> <output scalar file>"
        sys.exit(1)

    filename = sys.argv[1]
    outname  = sys.argv[2]

    convert(filename, outname)
