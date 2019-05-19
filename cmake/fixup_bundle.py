#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017-2019, The University of Oxford
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
from __future__ import print_function
import os
import shutil
import stat
import struct
import subprocess
import sys


def get_dependencies(file_name):
    """Gets dylib dependencies and rpaths in a Mach-O file."""
    # Define the Mach-O constants that we care about.
    MH_MAGIC = 0xfeedface
    MH_MAGIC_64 = 0xfeedfacf
    LC_LOAD_DYLIB = 0xc
    LC_RPATH = (0x1c | 0x80000000)
    file_header = struct.Struct('=IiiIIII')
    cmd_header = struct.Struct('=II')

    # Open the file and try to read the Mach-O header.
    if not os.path.exists(file_name) or os.path.islink(file_name):
        return None
    with open(file_name, 'rb') as file_handle:
        data = file_handle.read(file_header.size)
        if not data:
            return None
        (magic, _, _, _, num_cmds, _, _) = file_header.unpack(data)

        # Check the magic number.
        if magic == MH_MAGIC:
            pass
        elif magic == MH_MAGIC_64:
            data = file_handle.read(4)  # Read 4-byte padding.
        else:
            return None

        # Iterate load commands, storing dylib and rpath entries.
        deps = []
        rpath = []
        for _ in range(num_cmds):
            # Read the load command header.
            data = file_handle.read(cmd_header.size)
            (cmd_type, cmd_size) = cmd_header.unpack(data)

            # Read the load command payload.
            data = file_handle.read(cmd_size - cmd_header.size)
            # print(list('%2x'%b for b in data))

            # Record non-system LC_LOAD_DYLIB and all LC_RPATH entries.
            if cmd_type == LC_LOAD_DYLIB:
                offset = struct.unpack_from('=I', data[0:4])[0]
                path = data[(offset - cmd_header.size):]
                path = path[:path.find(b'\x00')]
                if path.startswith(b'/usr/lib') or \
                        path.startswith(b'/System'):
                    continue
                deps.append(path.decode('utf-8'))
            elif cmd_type == LC_RPATH:
                offset = struct.unpack_from('=I', data[0:4])[0]
                path = data[(offset - cmd_header.size):]
                path = path[:path.find(b'\x00')]
                rpath.append(path.decode('utf-8'))
        return {'deps': deps, 'rpaths': rpath}


def get_copy_path(file_path):
    """Returns the path we would need to copy from, allowing for frameworks."""
    copy_path = file_path
    ext = '.framework'
    if ext in copy_path:
        copy_path = copy_path[:copy_path.find(ext)] + ext
    return copy_path


def fixup(bundle_info, bundle_root, file_path, output_dir=None, depth=""):
    """Copies dependencies into the bundle, fixing install names and RPATHs."""
    if 'include' in file_path:
        return
    file_path = os.path.normpath(file_path)
    # print(depth + "Checking item %s" % file_path)
    if not output_dir:
        output_dir = os.path.join(bundle_root, 'Contents', 'Frameworks')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    local_path = file_path

    # Check if item is not in the bundle.
    if bundle_root not in file_path:
        # Copy the item into the bundle if it isn't already there.
        copy_path = get_copy_path(file_path)
        local_path = file_path.replace(os.path.dirname(copy_path), output_dir)
        if not os.path.exists(local_path):
            print(depth + "Bundling %s" % copy_path)
            if os.path.isdir(copy_path):
                shutil.copytree(
                    copy_path,
                    os.path.join(output_dir, os.path.basename(copy_path)),
                    symlinks=True,
                    ignore=shutil.ignore_patterns('Headers', '*_debug*'))
            else:
                shutil.copy(copy_path, output_dir)
            os.chmod(local_path,
                     stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)

    # Get dependencies of the item.
    if local_path not in bundle_info:
        bundle_info[local_path] = get_dependencies(file_path)
    if bundle_info[local_path] is None:
        return

    # Iterate dependencies of the item.
    loader_path = os.path.dirname(file_path)
    for dep in bundle_info[local_path]['deps']:
        # Get the full path of the dependency.
        dep_path = dep
        if dep.startswith('@rpath'):
            for rpath in bundle_info[local_path]['rpaths']:
                dep_path = dep.replace('@rpath', rpath)
                dep_path = dep_path.replace('@loader_path', loader_path)
                dep_path = dep_path.replace('@executable_path', loader_path)
                if os.path.exists(dep_path):
                    break
        dep_path = dep_path.replace('@loader_path', loader_path)
        dep_path = dep_path.replace('@executable_path', loader_path)

        # Update the item's RPATH so it can find its dependency.
        rel_path = os.path.relpath(output_dir, os.path.dirname(local_path))
        rel_path = '' if rel_path == '.' else os.path.join(rel_path, '')
        rpath = '@loader_path/' + rel_path
        if rpath not in bundle_info[local_path]['rpaths']:
            bundle_info[local_path]['rpaths'].append(rpath)
            print(depth + "In %s, adding %s to RPATH" %
                  (os.path.basename(local_path), rpath))
            subprocess.call(['install_name_tool', '-add_rpath',
                             rpath, local_path])

        # Update the install_name for this item if necessary.
        if not dep.startswith('@rpath'):
            new_name = dep_path.replace(
                os.path.dirname(get_copy_path(dep_path)), '@rpath')
            if new_name not in bundle_info[local_path]['deps']:
                bundle_info[local_path]['deps'][:] = [
                    new_name if d == dep else d
                    for d in bundle_info[local_path]['deps']]
                print(depth + "In %s, changing install name %s to %s" %
                      (os.path.basename(local_path), dep, new_name))
                subprocess.call(['install_name_tool', '-change',
                                 dep, new_name, local_path])

        # Recursive call.
        fixup(bundle_info, bundle_root, dep_path, depth=depth+"    ")

    # Delete /usr/local or @executable_path RPATHs from the bundled item.
    for rpath in bundle_info[local_path]['rpaths']:
        if 'libcudart' not in local_path and (
                rpath.startswith('/usr/local') or
                rpath.startswith('@executable_path')):
            bundle_info[local_path]['rpaths'].remove(rpath)
            print(depth + "In %s, removing %s from RPATH" %
                  (os.path.basename(local_path), rpath))
            subprocess.call(['install_name_tool', '-delete_rpath',
                             rpath, local_path])


def main():
    """Main for fixup_bundle module."""
    # Check command line arguments.
    if len(sys.argv) < 2:
        raise RuntimeError('Usage: python fixup_bundle.py <bundle.app>')

    # Create empty dictionary to store information about items in the bundle.
    bundle_info = {}

    # Get the bundle root path.
    bundle_root = sys.argv[1]

    # Get list of items in the bundle.
    items = []
    for root, _, files in os.walk(bundle_root):
        for name in files:
            items.append(os.path.join(root, name))

    # Fix-up each item in the list.
    for item in items:
        fixup(bundle_info, bundle_root, item)

    # Create symbolic link to GUI.
    link_dir = os.path.join(bundle_root, 'Contents', 'MacOS')
    if not os.path.isdir(link_dir):
        os.makedirs(link_dir)
    os.symlink("../Resources/bin/oskar", os.path.join(link_dir, 'oskar_gui'))

    # Bundle and fix up Qt plugins.
    plugin_dir = os.path.join(bundle_root, 'Contents', 'PlugIns')
    if len(sys.argv) > 2:
        fixup(bundle_info, bundle_root, sys.argv[2],
              os.path.join(plugin_dir, 'platforms'))
    if len(sys.argv) > 3:
        fixup(bundle_info, bundle_root, sys.argv[3],
              os.path.join(plugin_dir, 'styles'))


if __name__ == '__main__':
    main()
