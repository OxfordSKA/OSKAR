#!/usr/bin/python

"""
Module to plot an OSKAR format beam pattern text file.
"""


def load_beam_file(filename):
    """Load and OSKAR format beam pattern text file."""
    import numpy as np
    header = []
    fh = open(filename, 'r+')
    print '-'*80
    for i in range(0, 9):
        line = fh.readline()
        header.append(line[:-1])
        print header[-1]
    fh.close()
    print '-'*80
    num_chunks = int(header[4].split()[-1])
    num_times = int(header[5].split()[-1])
    num_channels = int(header[6].split()[-1])
    chunk_size = int(header[7].split()[-1])
    num_pixels = num_chunks*chunk_size
    imsize = np.sqrt(num_pixels)

    print 'No. chunks   = %i' % num_chunks
    print 'No. times    = %i' % num_times
    print 'No. channels = %i' % num_channels
    print 'Chunk size   = %i' % chunk_size
    print '-'*80
    print ''

    data = np.loadtxt(filename)
    img = np.zeros((num_pixels, num_times, num_channels), dtype=np.double)
    for chunk in range(0, num_chunks):
        for chan in range(0, num_channels):
            for time in range(0, num_times):
                idata0 = (chunk * num_channels * num_times * chunk_size) +\
                         (chan * num_times * chunk_size) +\
                         (time*chunk_size)
                idata1 = idata0+chunk_size
                ipix0 = chunk*chunk_size
                ipix1 = ipix0+chunk_size
                img[ipix0:ipix1, time, chan] = data[idata0:idata1]
    img = img.reshape((imsize, imsize, num_times, num_channels))
    return img

if __name__ == '__main__':

    import sys
    import numpy as np
    import matplotlib.pyplot as pp
    import matplotlib.cm as cm
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if len(sys.argv)-1 < 1:
        print 'Usage: plot_beam_text_file.py <filename> [channel=0] [time=0]'
        sys.exit(1)

    filename = sys.argv[1]

    chan = 0
    time = 0
    print len(sys.argv)-1
    if len(sys.argv)-1 >= 2:
        chan = int(sys.argv[2])
    if len(sys.argv)-1 == 3:
        time = int(sys.argv[3])

    img = load_beam_file(filename)

    fig = pp.figure(1, figsize=(10, 10))
    pp.clf()

    ax = fig.add_subplot(111, aspect='equal')
    data = img[:, :, time, chan]
    data = np.flipud(data)
    data[data == 0.0] = 1e-10
    datamax = np.nanmax(data)
    absdata = np.abs(data)
    data = 10.0*np.log10(absdata/datamax)
    pp.imshow(data, interpolation='nearest', cmap=cm.seismic)
    pp.clim([-60, 0])  # db range
    ax.set_title('Beam time:%i channel:%i' % (time, chan))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    cbar = pp.colorbar(cax=cax)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('Decibels', fontsize=8)
    ax.set_xlabel('East <-> West', fontsize=8)
    ax.set_ylabel('North <-> South', fontsize=8)
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])

    pp.show()
    # pp.savefig('beam.png', transparent=True, frameon=False)
