.. _faq:

**************************
Frequently Asked Questions
**************************

This page lists some questions about OSKAR and their solutions or workarounds,
if known.

1. `Why are my visibilities (or images) all zero or NaN?`_
2. `I've checked that the beam is far above the horizon, but I still see an empty or bizarre image`_
3. `I have a strange error: code 108 (unrecognized error code)`_


Why are my visibilities (or images) all zero or NaN?
----------------------------------------------------
The most common reason for this is that the beam is pointing below the horizon
for part or all of the observation (recent versions of OSKAR will now issue a
warning to the log if this is the case).
Check that the target field is visible from the telescope location for the
whole duration of the observation.


I've checked that the beam is far above the horizon, but I still see an empty or bizarre image
----------------------------------------------------------------------------------------------
Check also that the sky model contains sources where they should be, and that
the phase centre is set appropriately. Sky models used by OSKAR are independent
of the observation parameters, so if the phase centre is not where the sources
are, the simulated data may be very hard to interpret.


I have a strange error: code 108 (unrecognized error code)
----------------------------------------------------------
Any errors from CUDA function calls should be reported, but error codes
from CFITSIO may appear as unrecognised errors.
Check the `CFITSIO error codes here <https://heasarc.gsfc.nasa.gov/docs/software/fitsio/c/c_user/node128.html>`_.
