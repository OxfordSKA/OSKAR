# -*- coding: utf-8 -*-
"""Receives visibility data using SPEAD and images it using OSKAR.

To be used in conjunction with spead_send.py in `python/examples/spead/sender`.
Launch this script before the sender to avoid missing any data.

Usage: python spead_recv_imager.py <port> <oskar_imager.ini>

The command line arguments are:
    - port:             The UDP port number on which to listen.
    - oskar_imager.ini: Path to a settings file for the oskar_imager app.
"""

from __future__ import division, print_function
import logging
import sys

import oskar
import spead2
import spead2.recv


class SpeadReceiver(oskar.Imager):
    """Receives visibility data using SPEAD and images it using OSKAR.

    Inherits oskar.Imager to receive data in the run() method.
    """
    def __init__(self, log, port, precision=None, oskar_settings=None):
        oskar.Imager.__init__(self, precision, oskar_settings)
        self._log = log
        self._port = port
        self._stream = spead2.recv.Stream(spead2.ThreadPool(), 0)
        self._stream.add_udp_reader(port)
        self._header = {}

    def run(self):
        """Runs the receiver."""
        self._log.info("Initialising...")
        self.reset_cache()
        self.check_init()
        item_group = spead2.ItemGroup()

        # Iterate over all heaps in the stream.
        self._log.info("Waiting to receive on port {}".format(self._port))
        for heap in self._stream:
            # Extract data from the heap into a dictionary.
            data = {}
            items = item_group.update(heap)
            for item in items.values():
                data[item.name] = item.value

            # Read the header and set imager visibility meta-data.
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
                self.set_vis_frequency(
                    data['freq_start_hz'], data['freq_inc_hz'],
                    data['num_channels'])
                self.set_vis_phase_centre(
                    data['phase_centre_ra_deg'], data['phase_centre_dec_deg'])

            # Update the imager with visibility data from the SPEAD heap.
            if 'vis' in data:
                vis = data['vis']
                self.update(
                    vis['uu'], vis['vv'], vis['ww'], vis['amp'],
                    start_channel=data['channel_index'],
                    end_channel=data['channel_index'],
                    num_pols=self._header['num_pols'])

        # Stop the stream when there are no more heaps, and finalise the image.
        self._stream.stop()
        self.finalise()


def main():
    """Main function for OSKAR SPEAD receiver module."""
    # Check command line arguments.
    if len(sys.argv) < 3:
        raise RuntimeError('Usage: python spead_recv_imager.py '
                           '<port> <oskar_imager.ini>')

    # Get logger.
    log = logging.getLogger()
    log.addHandler(logging.StreamHandler(stream=sys.stdout))
    log.setLevel(logging.DEBUG)

    # Get socket port number.
    port = int(sys.argv[-2])

    # Load the OSKAR settings INI file for the application.
    settings = oskar.SettingsTree('oskar_imager', sys.argv[-1])

    # Check that illegal options are not selected.
    if settings['image/weighting'] == 'Uniform':
        raise RuntimeError(
            'Cannot use uniform weighting in streaming mode '
            'as this requires baseline coordinates to be known in advance.')
    if settings['image/algorithm'] == 'W-projection' and \
            settings['image/wproj/num_w_planes'] == '0':
        raise RuntimeError(
            'Cannot auto-determine required number of W-projection planes '
            'as this requires baseline coordinates to be known in advance.')

    # Append the port number to the output file root path.
    key = 'image/root_path'
    settings.set_value(key, settings[key] + "_" + str(port), False)

    # Set up the SPEAD receiver and run it (see method, above).
    receiver = SpeadReceiver(log, port, oskar_settings=settings)
    receiver.run()


if __name__ == '__main__':
    main()
