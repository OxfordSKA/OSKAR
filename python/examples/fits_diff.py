#!/usr/bin/env python
"""Module to difference two FITS images."""

import sys
import os
import numpy as np
import pyfits


def save_fits_image(filename, data, header, img1=None, img2=None):
    """Save a FITS image."""
    # Reshape to add the frequency axis
    data = np.reshape(data, (1, 1, data.shape[0], data.shape[1]))
    new_hdr = pyfits.header.Header()
    for i, item in enumerate(header.items()):
        if item[0] != 'HISTORY':
            new_hdr.append(item)
    if img1 and img2:
        new_hdr.append(('HISTORY', '' * 60))
        new_hdr.append(('HISTORY', '-' * 60))
        new_hdr.append(('HISTORY', 'Diff created from image1 - image2:'))
        new_hdr.append(('HISTORY', '- image1 : %s' % img1))
        new_hdr.append(('HISTORY', '- image2 : %s' % img2))
        new_hdr.append(('HISTORY', '-' * 60))
        new_hdr.append(('HISTORY', '  - Max       : % .3e' % np.max(data)))
        new_hdr.append(('HISTORY', '  - Min       : % .3e' % np.min(data)))
        new_hdr.append(('HISTORY', '  - Mean      : % .3e' % np.mean(data)))
        new_hdr.append(('HISTORY', '  - STD       : % .3e' % np.std(data)))
        new_hdr.append(('HISTORY', '  - RMS       : % .3e' %
                        np.sqrt(np.mean(data**2))))
        new_hdr.append(('HISTORY', '-' * 60))
        new_hdr.append(('HISTORY', '' * 60))

    if (os.path.exists(filename)):
        print '+ WARNING, output FITS file already exsits, overwriting.'
        os.remove(filename)
    pyfits.writeto(filename, data, new_hdr)


def load_fits_image(filename):
    """Load a FITS image."""

    hdulist = pyfits.open(filename)
    data = hdulist[0].data
    hdr = hdulist[0].header
    return np.squeeze(data), hdr


def fits_diff(outname, file1, file2):
    if not os.path.exists(file1):
        print 'ERROR: missing input file', file1
        return
    if not os.path.exists(file2):
        print 'ERROR: missing input file', file2
        return
    if os.path.exists(outname):
        return

    f1, h1 = load_fits_image(file1)
    f2, h2 = load_fits_image(file2)
    diff = f1 - f2
    print '-' * 80
    print '+ Image size  : %i x %i' % (f1.shape[0], f1.shape[1])
    print '+ File 1      : %s' % file1
    print '+ File 2      : %s' % file2
    print '+ Diff        : file1 - file2'
    print '+ Output name : %s' % (outname)
    print '+ Diff stats:'
    print '  - Max       : % .3e' % np.max(diff)
    print '  - Min       : % .3e' % np.min(diff)
    print '  - Mean      : % .3e' % np.mean(diff)
    print '  - STD       : % .3e' % np.std(diff)
    print '  - RMS       : % .3e' % np.sqrt(np.mean(diff**2))
    print '-' * 80
    save_fits_image(outname, diff, h1, file1, file2)


if __name__ == '__main__':

    if len(sys.argv) - 1 != 3:
        print 'Usage: fits_diff.py <diff name> <file1> <file2>'
        print ''
        print 'Performs: diff = file1 - file2'
        sys.exit(1)

    outname = sys.argv[1]
    file1 = sys.argv[2]
    file2 = sys.argv[3]

    f1, h1 = load_fits_image(file1)
    f2, h2 = load_fits_image(file2)

    diff = f1 - f2

    print '-' * 80
    print '+ Image size  : %i x %i' % (f1.shape[0], f1.shape[1])
    print '+ File 1      : %s' % file1
    print '+ File 2      : %s' % file2
    print '+ Diff        : file1 - file2'
    print '+ Output name : %s' % (outname)
    print '+ Diff stats:'
    print '  - Max       : % .3e' % np.max(diff)
    print '  - Min       : % .3e' % np.min(diff)
    print '  - Mean      : % .3e' % np.mean(diff)
    print '  - STD       : % .3e' % np.std(diff)
    print '  - RMS       : % .3e' % np.sqrt(np.mean(diff**2))
    print '-' * 80

    save_fits_image(outname, diff, h1, file1, file2)
