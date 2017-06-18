#
#  This file is part of OSKAR.
#
# Copyright (c) 2017, The University of Oxford
# All rights reserved.
#
#  This file is part of the OSKAR package.
#  Contact: oskar at oerc.ox.ac.uk
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

from __future__ import absolute_import
try:
    from . import _binary_lib
except ImportError:
    _binary_lib = None


class Binary(object):
    """This class provides a Python interface to an OSKAR binary data file."""

    def __init__(self, filename, mode=b'r'):
        """Constructs a handle to a binary data file.

        Args:
            filename (str): Path of the file to open.
            mode (Optional[char]): Open mode: 'r' for read, 'w' for write.
        """
        if _binary_lib is None:
            raise RuntimeError("OSKAR library not found.")
        self._capsule = _binary_lib.create(filename, mode)

    def get_num_tags(self):
        """Returns the number of data tags in the file."""
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
        return _binary_lib.read_data(self._capsule, group, tag, user_index,
                                     data_type)

    def set_query_search_start(self, start):
        """Sets the tag index at which to start the search query.

        Args:
            start (int): Index at which to start search query.
        """
        _binary_lib.set_query_search_start(self._capsule, start)

    # Properties
    num_tags = property(get_num_tags)
