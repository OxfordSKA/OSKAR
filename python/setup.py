#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Installs the OSKAR Python bindings."""

from os.path import join, isfile, dirname
import os
import platform
from numpy import get_include
try:
    from setuptools import setup, Extension
    from setuptools.command.build_ext import build_ext
except ImportError:
    from distutils.core import setup, Extension
    from distutils.command.build_ext import build_ext

# Define the version of OSKAR this is compatible with.
OSKAR_COMPATIBILITY_VERSION = '2.7'

# Define the extension modules to build.
MODULES = [
    ('_apps_lib', 'oskar_apps_lib.cpp'),
    ('_binary_lib', 'oskar_binary_lib.c'),
    ('_imager_lib', 'oskar_imager_lib.c'),
    ('_measurement_set_lib', 'oskar_measurement_set_lib.c'),
    ('_interferometer_lib', 'oskar_interferometer_lib.c'),
    ('_settings_lib', 'oskar_settings_lib.cpp'),
    ('_sky_lib', 'oskar_sky_lib.c'),
    ('_telescope_lib', 'oskar_telescope_lib.c'),
    ('_utils', 'oskar_utils.c'),
    ('_vis_block_lib', 'oskar_vis_block_lib.c'),
    ('_vis_header_lib', 'oskar_vis_header_lib.c'),
    ('_bda_utils', 'oskar_bda_utils.c')
]


class BuildExt(build_ext):
    """Class used to build OSKAR Python extensions. Inherits build_ext."""
    @staticmethod
    def find_file(name, dir_paths):
        """Returns path of given file if it exists in the list of directories.

        Args:
            name (str):                   The name of the file to find.
            dir_paths (array-like, str):  List of directories to search.
        """
        for directory in dir_paths:
            if os.path.exists(directory):
                test_path = join(directory, name)
                if isfile(test_path):
                    return test_path
        return None

    @staticmethod
    def dir_contains(name, dir_paths):
        """Returns directory if name fragment is part of a directory listing.

        Args:
            name (str):                   The name fragment to search for.
            dir_paths (array-like, str):  List of directories to search.
        """
        for directory in dir_paths:
            if os.path.exists(directory):
                dir_contents = os.listdir(directory)
                for item in dir_contents:
                    if name in item:
                        return directory
        return None

    @staticmethod
    def get_oskar_version(version_file):
        """Returns the version of OSKAR found on the system."""
        with open(version_file) as file_handle:
            for line in file_handle:
                if 'define OSKAR_VERSION_STR' in line:
                    return (line.split()[2]).replace('"', '')
        return None

    def run(self):
        """Overridden method. Runs the build.
        Library directories and include directories are checked here, first.
        """
        # Check we can find the OSKAR library.
        directory = self.dir_contains('oskar.', self.library_dirs)
        if not directory:
            raise RuntimeError(
                "Could not find OSKAR library. "
                "Check that OSKAR has already been installed on this system, "
                "and set the library path to build_ext "
                "using -L or --library-dirs")
        self.rpath.append(directory)
        self.libraries.append('oskar')
        self.libraries.append('oskar_apps')
        self.libraries.append('oskar_binary')
        self.libraries.append('oskar_settings')
        if self.dir_contains('oskar_ms.', self.library_dirs):
            self.libraries.append('oskar_ms')

        # Check we can find the OSKAR headers.
        header = self.find_file(
            join('oskar', 'oskar_version.h'), self.include_dirs)
        if not header:
            raise RuntimeError(
                "Could not find oskar/oskar_version.h. "
                "Check that OSKAR has already been installed on this system, "
                "and set the include path to build_ext "
                "using -I or --include-dirs")
        self.include_dirs.insert(0, dirname(header))
        self.include_dirs.insert(0, get_include())

        # Check the version of OSKAR is compatible.
        version = self.get_oskar_version(header)
        if not version.startswith(OSKAR_COMPATIBILITY_VERSION):
            raise RuntimeError(
                "The version of OSKAR found is not compatible with oskarpy. "
                "Found OSKAR %s, but require OSKAR %s." % (
                    version, OSKAR_COMPATIBILITY_VERSION)
            )
        build_ext.run(self)

    def build_extension(self, ext):
        """Overridden method. Builds each Extension."""
        ext.runtime_library_dirs = self.rpath

        # Unfortunately things don't work as they should on the Mac...
        if platform.system() == 'Darwin':
            for rpath in self.rpath:
                ext.extra_link_args.append('-Wl,-rpath,'+rpath)

        # Don't try to build MS extension if liboskar_ms is not found.
        if 'measurement_set' in ext.name:
            if not self.dir_contains('oskar_ms.', self.library_dirs):
                return
        build_ext.build_extension(self, ext)


def get_oskarpy_version():
    """Get the version of oskarpy from the version file."""
    globals_ = {}
    with open(join(dirname(__file__), 'oskar', '_version.py')) as file_handle:
        code = file_handle.read()
    # pylint: disable=exec-used
    exec(code, globals_)
    return globals_['__version__']


# Call setup() with list of extensions to build.
EXTENSIONS = []
for module in MODULES:
    EXTENSIONS.append(Extension(
        'oskar.' + module[0], sources=[join('oskar', 'src', module[1])],
        extra_compile_args=["-std=c99"]))
setup(
    name='oskarpy',
    version=get_oskarpy_version(),
    description='Radio interferometer simulation package (Python bindings)',
    packages=['oskar'],
    ext_modules=EXTENSIONS,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX',
        'Programming Language :: C',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3'
    ],
    author='University of Oxford',
    author_email='oskar@oerc.ox.ac.uk',
    url='http://oskar.oerc.ox.ac.uk',
    license='BSD',
    install_requires=['numpy'],
    setup_requires=['numpy'],
    cmdclass={'build_ext': BuildExt}
    )
