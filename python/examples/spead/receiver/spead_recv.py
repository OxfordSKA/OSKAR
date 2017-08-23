# -*- coding: utf-8 -*-
"""Receives visibility data using SPEAD and writes it to a Measurement Set.

To be used in conjunction with spead_send.py in `python/examples/spead/sender`.
Launch this script before the sender to avoid missing any data.

Usage: python spead_recv.py <port> <output_root_name>

The command line arguments are:
    - port:             The UDP port number on which to listen.
    - output_root_name: The root name of the Measurement Set to write.
                        The port number and ".ms" suffixes will be added.
"""

from __future__ import division, print_function
import logging
import math
import sys

import oskar
import spead2
import spead2.recv


class SpeadReceiver(object):
    """Receives visibility data using SPEAD and writes it to a Measurement Set.
    """
    def __init__(self, log, port, file_name):
        self._log = log
        self._port = port
        self._stream = spead2.recv.Stream(spead2.ThreadPool(), 0)
        self._stream.add_udp_reader(port)
        self._measurement_set = None
        self._file_name = file_name
        self._header = {}

    def run(self):
        """Runs the receiver."""
        item_group = spead2.ItemGroup()

        # Iterate over all heaps in the stream.
        self._log.info("Waiting to receive on port {}".format(self._port))
        for heap in self._stream:
            # Extract data from the heap into a dictionary.
            data = {}
            items = item_group.update(heap)
            for item in items.values():
                data[item.name] = item.value

            # Read the header and create the Measurement Set.
            if 'num_channels' in data:
                self._header = {
                    'freq_start_hz':        data['freq_start_hz'],
                    'freq_inc_hz':          data['freq_inc_hz'],
                    'num_baselines':        data['num_baselines'],
                    'num_channels':         data['num_channels'],
                    'num_pols':             data['num_pols'],
                    'num_stations':         data['num_stations'],
                    'phase_centre_ra_deg':  data['phase_centre_ra_deg'],
                    'phase_centre_dec_deg': data['phase_centre_dec_deg'],
                    'time_average_sec':     data['time_average_sec'],
                    'time_inc_sec':         data['time_inc_sec'],
                    'time_start_mjd_utc':   data['time_start_mjd_utc']
                }
                self._log.info(
                    "Receiving {} channel(s) starting at {} MHz.".format(
                        data['num_channels'], data['freq_start_hz'] / 1e6))
                if self._measurement_set is None:
                    self._measurement_set = oskar.MeasurementSet.create(
                        self._file_name, data['num_stations'],
                        data['num_channels'], data['num_pols'],
                        data['freq_start_hz'], data['freq_inc_hz'])
                    self._measurement_set.set_phase_centre(
                        math.radians(data['phase_centre_ra_deg']),
                        math.radians(data['phase_centre_dec_deg']))

            # Write visibility data from the SPEAD heap.
            if 'vis' in data:
                vis = data['vis']
                time_inc_sec = self._header['time_inc_sec']
                if data['channel_index'] == 0:
                    self._measurement_set.write_coords(
                        self._header['num_baselines'] * data['time_index'],
                        self._header['num_baselines'],
                        vis['uu'], vis['vv'], vis['ww'],
                        self._header['time_average_sec'], time_inc_sec,
                        self._header['time_start_mjd_utc'] * 86400 +
                        time_inc_sec * (data['time_index'] + 0.5))
                self._measurement_set.write_vis(
                    self._header['num_baselines'] * data['time_index'],
                    data['channel_index'], 1,
                    self._header['num_baselines'], vis['amp'])

        # Stop the stream when there are no more heaps.
        self._stream.stop()


def main():
    """Main function for OSKAR SPEAD receiver module."""
    # Check command line arguments.
    if len(sys.argv) < 3:
        raise RuntimeError('Usage: python spead_recv.py '
                           '<port> <output_root_name>')

    # Get logger.
    log = logging.getLogger()
    log.addHandler(logging.StreamHandler(stream=sys.stdout))
    log.setLevel(logging.DEBUG)

    # Get socket port number.
    port = int(sys.argv[-2])

    # Append the port number to the output file root path.
    file_name = sys.argv[-1] + "_" + str(port) + ".ms"

    # Set up the SPEAD receiver and run it (see method, above).
    receiver = SpeadReceiver(log, port, file_name)
    receiver.run()


if __name__ == '__main__':
    main()
