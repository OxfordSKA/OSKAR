# -*- coding: utf-8 -*-
"""Simulates visibility data using OSKAR and sends it using SPEAD.

Usage: python spead_send.py <spead_send.json> <oskar_sim_interferometer.ini>

The command line arguments are:
    - spead_send.json:              Path to a JSON file containing
                                    the SPEAD configuration. See below.
    - oskar_sim_interferometer.ini: Path to a settings file for the
                                    oskar_sim_interferometer app.

An example SPEAD configuration JSON file could be:

    {
        "stream_config":
        {
            "max_packet_size": 1472,
            "rate": 0.0,
            "burst_size": 8000,
            "max_heaps": 4
        },
        "streams":
        [
            {
                "port": 41000,
                "host": "127.0.0.1"
            }
        ]
    }

- ``stream_config`` is a dictionary describing the stream configuration.
  See ``https://spead2.readthedocs.io/en/v1.3.2/py-send.html``
- ``streams`` is a list of dictionaries to describe where SPEAD heap data
  should be sent.
- Each stream dictionary can have the keys ``port``, ``host`` and ``threads``.
- ``port`` and ``host`` are required and give the address to which
  SPEAD heaps should be sent.
- ``threads`` is optional (default 1) and sets the number of threads used
  to send the heap.

To send data over multiple SPEAD streams, either run this script multiple
times with different configuration files, or specify multiple stream
dictionaries in the list. The visibility data will be split by frequency
channel among the specified list of streams. Note that the latter option will
be slower than running multiple senders in parallel on different machines.

Launch one receiver script per SPEAD stream.
"""

from __future__ import division, print_function
import logging
import sys

import numpy
import oskar
import simplejson as json
import spead2
import spead2.send


class SpeadSender(oskar.Interferometer):
    """Simulates visibility data using OSKAR and sends it using SPEAD.

    Inherits oskar.Interferometer to send data in the process_block() method.
    SPEAD is configured using a Python dictionary passed to the constructor.
    """
    def __init__(self, log, spead_config, precision=None, oskar_settings=None):
        oskar.Interferometer.__init__(self, precision, oskar_settings)
        self._log = log
        self._streams = []
        self._vis_pack = None

        # Construct UDP streams and associated item groups.
        stream_config = spead2.send.StreamConfig(
            spead_config['stream_config']['max_packet_size'],
            spead_config['stream_config']['rate'],
            spead_config['stream_config']['burst_size'],
            spead_config['stream_config']['max_heaps'])
        for stream in spead_config['streams']:
            threads = stream['threads'] if 'threads' in stream else 1
            thread_pool = spead2.ThreadPool(threads=threads)
            log.info("Creating SPEAD stream for host {} on port {} ..."
                     .format(stream['host'], stream['port']))
            udp_stream = spead2.send.UdpStream(thread_pool, stream['host'],
                                               stream['port'], stream_config)
            item_group = spead2.send.ItemGroup(
                flavour=spead2.Flavour(4, 64, 40, 0))

            # Append udp_stream and item_group to the stream list as a tuple.
            self._streams.append((udp_stream, item_group))

    def finalise(self):
        """Called automatically by the base class at the end of run()."""
        oskar.Interferometer.finalise(self)
        # Send the end of stream message to each stream.
        for stream, item_group in self._streams:
            stream.send_heap(item_group.get_end())

    def process_block(self, block, block_index):
        """Sends the visibility block using SPEAD.

        Args:
            block (oskar.VisBlock): The block to be processed.
            block_index (int):      The index of the visibility block.
        """
        # Write the block to any open files (reimplements base class method).
        self.write_block(block, block_index)

        # Get number of streams and maximum number of channels per stream.
        num_streams = len(self._streams)
        hdr = self.vis_header()
        max_channels_per_stream = (hdr.num_channels_total +
                                   num_streams - 1) // num_streams

        # Initialise SPEAD heaps if required.
        if block_index == 0:
            self._create_heaps(block)

            # Write the header information to each SPEAD stream.
            for stream_index, (_, heap) in enumerate(self._streams):
                channel_start = stream_index * max_channels_per_stream
                channel_end = (stream_index + 1) * max_channels_per_stream - 1
                if channel_end > hdr.num_channels_total - 1:
                    channel_end = hdr.num_channels_total - 1
                heap['freq_inc_hz'].value = hdr.freq_inc_hz
                heap['freq_start_hz'].value = (
                    hdr.freq_start_hz + channel_start * hdr.freq_inc_hz)
                heap['num_baselines'].value = block.num_baselines
                heap['num_channels'].value = 1 + channel_end - channel_start
                heap['num_pols'].value = block.num_pols
                heap['num_stations'].value = block.num_stations
                heap['phase_centre_ra_deg'].value = hdr.phase_centre_ra_deg
                heap['phase_centre_dec_deg'].value = hdr.phase_centre_dec_deg
                heap['time_start_mjd_utc'].value = hdr.time_start_mjd_utc
                heap['time_inc_sec'].value = hdr.time_inc_sec
                heap['time_average_sec'].value = hdr.time_average_sec

        # Loop over all times and channels in the block.
        self._log.info("Sending visibility block {}/{}"
                       .format(block_index + 1, self.num_vis_blocks))
        for t in range(block.num_times):
            for c in range(block.num_channels):
                # Get the SPEAD stream for this channel index.
                channel_index = block.start_channel_index + c
                stream_index = channel_index // max_channels_per_stream
                stream, heap = self._streams[stream_index]

                # Pack the visibility data into array of structures,
                # ready for sending.
                self._vis_pack['uu'] = block.baseline_uu_metres()[t, :]
                self._vis_pack['vv'] = block.baseline_vv_metres()[t, :]
                self._vis_pack['ww'] = block.baseline_ww_metres()[t, :]
                self._vis_pack['amp'] = block.cross_correlations()[t, c, :, :]

                # Channel index is relative to the channels in the stream.
                heap['channel_index'].value = (
                    channel_index - stream_index * max_channels_per_stream)

                # Update the heap and send it.
                heap['vis'].value = self._vis_pack
                heap['time_index'].value = block.start_time_index + t
                stream.send_heap(heap.get_heap())

    def _create_heaps(self, block):
        """Create SPEAD heap items based on content of the visibility block.

        Args:
            block (oskar.VisBlock): Visibility block.
        """
        # SPEAD heap descriptor.
        # One channel and one time per heap: num_channels is used to tell
        # the receiver how many channels it will be receiving in total.
        amp_type = block.cross_correlations().dtype.name
        descriptor = {
            'channel_index':        {'dtype': 'i4'},
            'freq_inc_hz':          {'dtype': 'f8'},
            'freq_start_hz':        {'dtype': 'f8'},
            'num_baselines':        {'dtype': 'i4'},
            'num_channels':         {'dtype': 'i4'},
            'num_pols':             {'dtype': 'i4'},
            'num_stations':         {'dtype': 'i4'},
            'phase_centre_ra_deg':  {'dtype': 'f8'},
            'phase_centre_dec_deg': {'dtype': 'f8'},
            'time_average_sec':     {'dtype': 'f8'},
            'time_index':           {'dtype': 'i4'},
            'time_inc_sec':         {'dtype': 'f8'},
            'time_start_mjd_utc':   {'dtype': 'f8'},
            'vis': {
                'dtype': [
                    ('uu', block.baseline_uu_metres().dtype.name),
                    ('vv', block.baseline_vv_metres().dtype.name),
                    ('ww', block.baseline_ww_metres().dtype.name),
                    ('amp', amp_type, (block.num_pols,))
                ],
                'shape': (block.num_baselines,)
            }
        }

        # Allocate array of structures for the packed visibility data.
        self._vis_pack = numpy.zeros((block.num_baselines,),
                                     dtype=descriptor['vis']['dtype'])

        # Add items to the item group based on the heap descriptor.
        for stream, item_group in self._streams:
            for key, item in descriptor.items():
                item_shape = item['shape'] if 'shape' in item else tuple()
                item_group.add_item(
                    id=None, name=key, description='',
                    shape=item_shape, dtype=item['dtype'])

            # Send the start of stream message to each stream.
            stream.send_heap(item_group.get_start())


def main():
    """Main function for OSKAR SPEAD sender module."""
    # Check command line arguments.
    if len(sys.argv) < 3:
        raise RuntimeError('Usage: python spead_send.py '
                           '<spead_send.json> <oskar_sim_interferometer.ini>')

    # Get logger.
    log = logging.getLogger()
    log.addHandler(logging.StreamHandler(stream=sys.stdout))
    log.setLevel(logging.DEBUG)

    # Load SPEAD configuration from JSON file.
    with open(sys.argv[-2]) as f:
        spead_config = json.load(f)

    # Load the OSKAR settings INI file for the application.
    settings = oskar.SettingsTree('oskar_sim_interferometer', sys.argv[-1])

    # Set up the SPEAD sender and run it (see method, above).
    sender = SpeadSender(log, spead_config, oskar_settings=settings)
    sender.run()


if __name__ == '__main__':
    main()
