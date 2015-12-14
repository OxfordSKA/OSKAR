#!/usr/bin/python


def load_fits_image(filename):
    import pyfits
    import numpy as np
    hdulist = pyfits.open(filename)
    img = hdulist[0].data
    hdr = hdulist[0].header
    return np.squeeze(img), hdr

if __name__ == '__main__':

    import sys
    import numpy as np

    if len(sys.argv)-1 != 1:
        print 'Usage: fits_image_stats.py <filename>'
        sys.exit(1)

    filename = sys.argv[1]

    img, hdr = load_fits_image(filename)

#     print 'Mean = %f'  % np.mean(img)
#     print 'RMS  = %f'  % np.sqrt(np.mean(img**2))
#     print 'STD  = %f'  % np.std(img)
    print np.std(img)
