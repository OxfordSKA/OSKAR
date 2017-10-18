# -*- coding: utf-8 -*-
"""Simulates visibilities using OSKAR and corrupts them.

Usage: python corruptor.py <oskar_sim_interferometer.ini>

The command line arguments are:
    - oskar_sim_interferometer.ini: Path to a settings file for the
                                    oskar_sim_interferometer app.
"""
from __future__ import print_function
import sys

import oskar


class Corruptor(oskar.Interferometer):
    """Corrupts visibilities on-the-fly from OSKAR.
    """
    def __init__(self, precision=None, oskar_settings=None):
        oskar.Interferometer.__init__(self, precision, oskar_settings)

        # Do any other initialisation here...
        print("Initialising...")

    def finalise(self):
        """Called automatically by the base class at the end of run()."""
        oskar.Interferometer.finalise(self)

        # Do any other finalisation here...
        print("Finalising...")

    def process_block(self, block, block_index):
        """Corrupts the visibility block amplitude data.

        Args:
            block (oskar.VisBlock): The block to be processed.
            block_index (int):      The index of the visibility block.
        """
        # Get handles to visibility block data as numpy arrays.
        uu = block.baseline_uu_metres()
        vv = block.baseline_vv_metres()
        ww = block.baseline_ww_metres()
        amp = block.cross_correlations()

        # Corrupt visibility amplitudes in the block here as needed
        # by messing with amp array.
        # uu, vv, ww have dimensions (num_times,num_baselines)
        # amp has dimensions (num_times,num_channels,num_baselines,num_pols)
        print("Processing block {}/{} (time index {}-{})...".
              format(block_index + 1,
                     self.num_vis_blocks,
                     block.start_time_index,
                     block.start_time_index + block.num_times - 1))

        # Simplest example: amp *= 2.0
        amp *= 2.0

        # Write corrupted visibilities in the block to file(s).
        self.write_block(block, block_index)


def main():
    """Main function for visibility corruptor."""
    # Check command line arguments.
    if len(sys.argv) < 2:
        raise RuntimeError('Usage: python corruptor.py '
                           '<oskar_sim_interferometer.ini>')

    # Load the OSKAR settings INI file for the application.
    settings = oskar.SettingsTree('oskar_sim_interferometer', sys.argv[-1])

    # Set up the corruptor and run it (see method, above).
    corruptor = Corruptor(oskar_settings=settings)
    corruptor.run()


if __name__ == '__main__':
    main()
