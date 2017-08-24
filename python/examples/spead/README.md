# SPEAD send and receive examples

Ensure the OSKAR Python interface has been installed by following the
instructions in the top-level `python` folder.

These examples also required the `spead2` Python library
(see https://spead2.readthedocs.io) to be installed.

## To launch a receiver

    cd receiver
    python spead_recv.py 41000 example

## To launch a sender

    cd sender
    python spead_send.py spead_send.json oskar_sim_interferometer.ini

Further information is in the Python module docstrings of each file.

# Notes

You may see messages from `spead2` such as this:

    requested socket buffer size 524288 but only received 212992:
    refer to documentation for details on increasing buffer size

On Ubuntu Linux, the default socket buffer sizes can be changed by
adding lines to the end of `/etc/sysctl.conf` as follows:

    net.core.rmem_max = 33554432
    net.core.rmem_default = 8388608
    net.core.wmem_max = 524288
    net.core.wmem_default = 524288

Reboot the machine after editing the file for the changes to come into effect.
