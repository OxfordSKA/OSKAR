# - this module looks for Matlab
#
# Defines:
#  MATLAB_INCLUDE_DIR: include path for mex.h, engine.h
#  MATLAB_LIBRARIES:   required libraries: libmex, etc
#
#  MATLAB_BINARY_DIR: path to matlab binaries
#
#  MATLAB_MEX_LIBRARY: filename path to mex library
#  MATLAB_MAT_LIBRARY: filename path to mat library
#  MATLAB_MX_LIBRARY:  filename path to mx library
#  MATLAB_ENG_LIBRARY: filename path to eng library
#
#  MATLAB_QT_QTCORE_LIBRARY
#  MATLAB_QT_QTGUI_LIBRARY
#  MATLAB_QT_QTXML_LIBRARY
#
#=============================================================================
# Copyright 2005-2009 Kitware, Inc.
#
# Distributed under the OSI-approved BSD License (the "License");
# see accompanying file Copyright.txt for details.
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================
#
# edited to reflect some of the changes from:
#   http://public.kitware.com/Bug/view.php?id=8207
# Cleanup with support for definition of MATLAB_ROOT
#

set(LIB_NAMES mex mat mx)
set(MATLAB_ROOT_HINTS
    ${MATLAB_ROOT}
    $ENV{MATLAB_ROOT}
)
set(MATLAB_ROOT_PATHS
    /Applications/*
    /usr/local/MATLAB/*
    /data/MATLAB/*
    /opt/*/*
)
set(MATLAB_LIB_SUFFIXES
    /bin/glnxa64/
    /bin/glnx86/
    /bin/maci64/
)

SET(MATLAB_FOUND FALSE)
if (WIN32)
    if(${CMAKE_GENERATOR} MATCHES "Visual Studio .*" OR ${CMAKE_GENERATOR} MATCHES "NMake Makefiles")
        set(MATLAB_ROOT "[HKEY_LOCAL_MACHINE\\SOFTWARE\\MathWorks\\MATLAB\\7.0;MATLABROOT]/extern/lib/win32/microsoft/")
    else(${CMAKE_GENERATOR} MATCHES "Visual Studio .*" OR ${CMAKE_GENERATOR} MATCHES "NMake Makefiles")
        if(${CMAKE_GENERATOR} MATCHES "Borland")
            # Same here, there are also: bcc50 and bcc51 directories
            set(MATLAB_ROOT "[HKEY_LOCAL_MACHINE\\SOFTWARE\\MathWorks\\MATLAB\\7.0;MATLABROOT]/extern/lib/win32/microsoft/bcc54")
        else(${CMAKE_GENERATOR} MATCHES "Borland")
            message(FATAL_ERROR "Generator not compatible: ${CMAKE_GENERATOR}")
        endif(${CMAKE_GENERATOR} MATCHES "Borland")
    endif(${CMAKE_GENERATOR} MATCHES "Visual Studio .*" OR ${CMAKE_GENERATOR} MATCHES "NMake Makefiles")
    find_library(MATLAB_MEX_LIBRARY libmex ${MATLAB_ROOT})
    find_library(MATLAB_MX_LIBRARY libmx ${MATLAB_ROOT})
    find_library(MATLAB_ENG_LIBRARY libeng ${MATLAB_ROOT})
    find_library(MATLAB_MAT_LIBRARY libmat ${MATLAB_ROOT})
    find_path(MATLAB_INCLUDE_DIR "mex.h"
    "[HKEY_LOCAL_MACHINE\\SOFTWARE\\MathWorks\\MATLAB\\7.0;MATLABROOT]/extern/include")
else (WIN32)

    foreach (lib ${LIB_NAMES})
        string(TOUPPER ${lib} _LIB)
        find_library(MATLAB_${_LIB}_LIBRARY ${lib}
            HINTS ${MATLAB_ROOT_HINTS}
            PATHS ${MATLAB_ROOT_PATHS}
            PATH_SUFFIXES ${MATLAB_LIB_SUFFIXES}
            NO_DEFAULT_PATH)
    endforeach ()

    # HACK: find_library doesn't seem to be able to find versioned libraries... :(
    set(MATLAB_QT_LIBS QtCore QtXml QtGui)
    if (APPLE)
        foreach (lib ${MATLAB_QT_LIBS})
            string(TOUPPER ${lib} _LIB)
            find_library(MATLAB_QT_${_LIB}_LIBRARY ${lib}
                HINTS ${MATLAB_ROOT_HINTS}
                PATHS ${MATLAB_ROOT_PATHS}
                PATH_SUFFIXES ${MATLAB_LIB_SUFFIXES}
                NO_DEFAULT_PATH)
        endforeach ()
    else()
        foreach (lib ${MATLAB_QT_LIBS})
            string(TOUPPER ${lib} _LIB)
            find_file(MATLAB_QT_${_LIB}_LIBRARY ${lib}
                NAMES lib${lib}.so lib${lib}.so.4 lib${lib}.dylib lib${lib}.dylib.4
                HINTS ${MATLAB_ROOT_HINTS}
                PATHS ${MATLAB_ROOT_PATHS}
                PATH_SUFFIXES ${MATLAB_LIB_SUFFIXES}
                NO_DEFAULT_PATH)
        endforeach ()
        # HACK TO MAKE R2013b work - this seems to have dropped Qt inside the
        # matlab folder so need to use the system one instead.
        if (    NOT MATLAB_QT_QTCORE_LIBRARY_FOUND AND 
                NOT MATLAB_QT_QTXML_LIBRARY_FOUND AND 
                NOT MATLAB_QT_QTGUI_LIBRARY_FOUND
            )
            if (QT4_FOUND)
                #find_package(Qt4 4.6 QUIET)
                set(MATLAB_QT_QTCORE_LIBRARY ${QT_QTCORE_LIBRARY})
                set(MATLAB_QT_QTXML_LIBRARY ${QT_QTXML_LIBRARY})
                set(MATLAB_QT_QTGUI_LIBRARY ${QT_QTGUI_LIBRARY})
            endif ()
        endif()
    endif()

    find_path(MATLAB_INCLUDE_DIR "mex.h"
        HINTS ${MATLAB_ROOT_HINTS}
        PATHS ${MATLAB_ROOT_PATHS}
        PATH_SUFFIXES /extern/include/
        NO_DEFAULT_PATH)
endif(WIN32)

# This is common to UNIX and Win32:
foreach (lib ${LIB_NAMES})
    string(TOUPPER ${lib} _LIB)
    list(APPEND MATLAB_LIBRARIES ${MATLAB_${_LIB}_LIBRARY})
endforeach ()

# Find the MATLAB binary path
find_path(MATLAB_BINARY_DIR matlab
    HINTS ${MATLAB_ROOT_HINTS}
    PATHS ${MATLAB_ROOT_PATHS}
    PATH_SUFFIXES /bin
    NO_DEFAULT_PATH)

if (MATLAB_INCLUDE_DIR AND MATLAB_LIBRARIES AND MATLAB_BINARY_DIR)
    set(MATLAB_FOUND TRUE)
endif(MATLAB_INCLUDE_DIR AND MATLAB_LIBRARIES AND MATLAB_BINARY_DIR)

if (MATLAB_FOUND)
    message(STATUS "MATLAB_LIBRARIES: ${MATLAB_LIBRARIES}")
endif ()

mark_as_advanced(
  MATLAB_LIBRARIES
  MATLAB_MEX_LIBRARY
  MATLAB_MX_LIBRARY
  MATLAB_ENG_LIBRARY
  MATLAB_INCLUDE_DIR
  MATLAB_FOUND
  MATLAB_ROOT
)
