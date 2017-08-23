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
