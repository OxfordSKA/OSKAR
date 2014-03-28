# 
#  This file is part of OSKAR.
# 
# !!TODO LICENCE GOES HERE!!
#
# 
"""OSKAR is a package to simulate interferometric visibility data ....
"""

import warnings

try:
    ImportWarning
except NameError:
    class ImportWarning(Warning):
        pass

from version import __version__

#import image
import image
import mem
from mem import (mloc, mtype)

