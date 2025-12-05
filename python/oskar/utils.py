# -*- coding: utf-8 -*-
#
# Copyright (c) 2016-2025, The OSKAR Developers.
# See the LICENSE file at the top-level directory of this distribution.
#

"""Interfaces to OSKAR utility functions."""

from __future__ import absolute_import
try:
    from . import _utils
except ImportError as exc:
    ERROR_MSG = exc.msg
    _utils = None


def oskar_version_string():
    """Returns the version of the OSKAR library in use."""
    if _utils is None:
        raise RuntimeError(f"OSKAR library not found ({ERROR_MSG})")
    return _utils.version_string()
