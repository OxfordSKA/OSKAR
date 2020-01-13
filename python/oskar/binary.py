# -*- coding: utf-8 -*-
#
# Copyright (c) 2017-2020, The University of Oxford
# All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#  3. Neither the name of the University of Oxford nor the names of its
#     contributors may be used to endorse or promote products derived from this
#     software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#

"""Interfaces to OSKAR binary files."""

from __future__ import absolute_import, print_function
try:
    from . import _binary_lib
except ImportError:
    _binary_lib = None

# pylint: disable=useless-object-inheritance
class Binary(object):
    """This class provides a Python interface to an OSKAR binary data file.

    The :class:`oskar.Binary` class provides a low-level interface to allow
    binary data files written by OSKAR to be read from Python using its
    :meth:`read() <oskar.Binary.read>` method.

    See the
    `binary file documentation <https://github.com/OxfordSKA/OSKAR/releases>`_
    for details on the binary file format used by OSKAR if you need it.
    To read visibility data into Python, it is recommended to use the (much)
    more convenient :meth:`oskar.VisBlock.read` method instead,
    and this page can be safely ignored!

    """

    def __init__(self, filename, mode=b'r'):
        """Constructs a handle to a binary data file.

        Args:
            filename (str): Path of the file to open.
            mode (Optional[char]): Open mode: 'r' for read, 'w' for write.
        """
        if _binary_lib is None:
            raise RuntimeError("OSKAR library not found.")
        self._filename = filename
        self._mode = mode
        self._capsule = None

    def capsule_ensure(self):
        """Ensures the C capsule exists."""
        if self._capsule is None:
            self._capsule = _binary_lib.create(self._filename, self._mode)

    def capsule_get(self):
        """Returns the C capsule wrapped by the class."""
        self.capsule_ensure()
        return self._capsule

    def capsule_set(self, new_capsule):
        """Sets the C capsule wrapped by the class.

        Args:
            new_capsule (capsule): The new capsule to set.
        """
        if _binary_lib.capsule_name(new_capsule) == 'oskar_Binary':
            del self._capsule
            self._capsule = new_capsule
        else:
            raise RuntimeError("Capsule is not of type oskar_Binary.")

    def get_num_tags(self):
        """Returns the number of data tags in the file."""
        self.capsule_ensure()
        return _binary_lib.num_tags(self._capsule)

    def read(self, group, tag, user_index=0, data_type=None):
        """Returns data for the specified chunk in the file.

        The returned data will be either an array, a single value, or a string,
        depending on the contents of the data block.

        Args:
            group (int or str):
                The chunk group ID to match.
            tag (int or str):
                The chunk tag ID to match within the group.
            user_index (Optional[int]):
                The user-specified index to read. Defaults to 0.
            data_type (Optional[int]):
                The enumerated OSKAR data type to read.
                If not given, the first type found matching the other filters
                is returned.
        """
        self.capsule_ensure()
        return _binary_lib.read_data(self._capsule, group, tag, user_index,
                                     data_type)

    def set_query_search_start(self, start):
        """Sets the tag index at which to start the search query.

        Args:
            start (int): Index at which to start search query.
        """
        self.capsule_ensure()
        _binary_lib.set_query_search_start(self._capsule, start)

    # Properties
    capsule = property(capsule_get, capsule_set)
    num_tags = property(get_num_tags)
