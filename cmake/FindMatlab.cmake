# - this module looks for Matlab
#
# Defines:
#  MATLAB_INCLUDE_DIR: include path for mex.h, engine.h
#  MATLAB_LIBRARIES:   required libraries: libmex, etc
#
#  MATLAB_BINARY_DIR: path to matlab binaries
#
#  MATLAB_MEX_LIBRARY: filename path to mex library
#  MATLAB_MX_LIBRARY:  filename path to mx library
#  MATLAB_ENG_LIBRARY: filename path to end library
#
#  MATLAB_QT_QTCORE_LIBRARY
#  MATLAB_QT_QTGUI_LIBRARY
#  MATLAB_QT_QTXML_LIBRARY

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
#

set(LIB_NAMES mex mat mx eng) 

SET(MATLAB_FOUND FALSE)
IF(WIN32)
  IF(${CMAKE_GENERATOR} MATCHES "Visual Studio .*" OR ${CMAKE_GENERATOR} MATCHES "NMake Makefiles")
    SET(MATLAB_ROOT "[HKEY_LOCAL_MACHINE\\SOFTWARE\\MathWorks\\MATLAB\\7.0;MATLABROOT]/extern/lib/win32/microsoft/")
  ELSE(${CMAKE_GENERATOR} MATCHES "Visual Studio .*" OR ${CMAKE_GENERATOR} MATCHES "NMake Makefiles")
      IF(${CMAKE_GENERATOR} MATCHES "Borland")
        # Same here, there are also: bcc50 and bcc51 directories
        SET(MATLAB_ROOT "[HKEY_LOCAL_MACHINE\\SOFTWARE\\MathWorks\\MATLAB\\7.0;MATLABROOT]/extern/lib/win32/microsoft/bcc54")
      ELSE(${CMAKE_GENERATOR} MATCHES "Borland")
        MESSAGE(FATAL_ERROR "Generator not compatible: ${CMAKE_GENERATOR}")
      ENDIF(${CMAKE_GENERATOR} MATCHES "Borland")
  ENDIF(${CMAKE_GENERATOR} MATCHES "Visual Studio .*" OR ${CMAKE_GENERATOR} MATCHES "NMake Makefiles")
  FIND_LIBRARY(MATLAB_MEX_LIBRARY
    libmex
    ${MATLAB_ROOT}
    )
  FIND_LIBRARY(MATLAB_MX_LIBRARY
    libmx
    ${MATLAB_ROOT}
    )
  FIND_LIBRARY(MATLAB_ENG_LIBRARY
    libeng
    ${MATLAB_ROOT}
    )
  FIND_LIBRARY(MATLAB_MAT_LIBRARY
    libmat
    ${MATLAB_ROOT}
    )

  FIND_PATH(MATLAB_INCLUDE_DIR
    "mex.h"
    "[HKEY_LOCAL_MACHINE\\SOFTWARE\\MathWorks\\MATLAB\\7.0;MATLABROOT]/extern/include"
    )
ELSE( WIN32 )

  IF(NOT MATLAB_ROOT)
    IF($ENV{MATLAB_ROOT})
      SET(MATLAB_ROOT $ENV{MATLAB_ROOT})
    ELSE($ENV{MATLAB_ROOT})
      SET(MATLAB_ROOT /opt/matlab)
    ENDIF($ENV{MATLAB_ROOT})
  ENDIF(NOT MATLAB_ROOT)
  
  IF(CMAKE_SIZEOF_VOID_P EQUAL 4)

    # ==== x86 ====
    set(MATLAB_LIB_PATHS
      /usr/local/MATLAB/R2010b/bin/glnx86
      /usr/local/MATLAB/R2011a/bin/glnx86
      /usr/local/MATLAB/R2011b/bin/glnx86
      /usr/local/MATLAB/R2012a/bin/glnx86
    )
  
    foreach (lib ${LIB_NAMES})
        string(TOUPPER ${lib} _LIB)
        find_library(MATLAB_${_LIB}_LIBRARY ${lib} 
            HINTS $ENV{MATLAB_ROOT}/bin/glnx86/
            PATHS ${MATLAB_LIB_PATHS}
        )
    endforeach ()
      
  ELSE(CMAKE_SIZEOF_VOID_P EQUAL 4)
  
    # ==== x86_64 ====
    set(MATLAB_LIB_PATHS
      /usr/local/MATLAB/R2010b/bin/glnxa64
      /usr/local/MATLAB/R2011a/bin/glnxa64
      /usr/local/MATLAB/R2011b/bin/glnxa64
      /usr/local/MATLAB/R2012a/bin/glnxa64
      /usr/local/matlab/bin/glnxa64
      /data/MATLAB/R2012a/bin/glnxa64
      /Applications/MATLAB_R2011a.app/bin/maci64
      /Applications/MATLAB_R2012a.app/bin/maci64
      /Applications/MATLAB_R2012b.app/bin/maci64
    )

    foreach (lib ${LIB_NAMES})
        string(TOUPPER ${lib} _LIB)
        find_library(MATLAB_${_LIB}_LIBRARY ${lib} 
            HINTS $ENV{MATLAB_ROOT}/bin/glnxa64/
            PATHS ${MATLAB_LIB_PATHS}
        )
    endforeach ()
    
  ENDIF(CMAKE_SIZEOF_VOID_P EQUAL 4)

  # HACK: find_library doesnt seem to be able to find versioned libraries... :(
  if (NOT APPLE)    
    find_file(MATLAB_QT_QTCORE_LIBRARY libQtCore.so.4
        HINTS $ENV{MATLAB_ROOT}/bin/glnxa64/
        PATHS ${MATLAB_LIB_PATHS} 
        NO_DEFAULT_PATH)
  else ()
    find_package(Qt4 4.6 QUIET)
    set(MATLAB_QT_CORE_LIBRARY ${QT_QTCORE_LIBRARY})
  endif()

  find_path(MATLAB_INCLUDE_DIR
    "mex.h"
    HINTS
    $ENV{MATLAB_ROOT}/extern/include
    PATHS
    /usr/local/MATLAB/R2010b/extern/include
    /usr/local/MATLAB/R2011a/extern/include
    /usr/local/MATLAB/R2011b/extern/include
    /usr/local/MATLAB/R2012a/extern/include
    /usr/local/matlab/extern/include
    /data/MATLAB/R2011b/extern/include
    /data/MATLAB/R2012a/extern/include
    /Applications/MATLAB_R2011a.app/extern/include
    /Applications/MATLAB_R2012a.app/extern/include
    /Applications/MATLAB_R2012b.app/extern/include
  )
ENDIF(WIN32)


# This is common to UNIX and Win32:
foreach (lib ${LIB_NAMES})
    string(TOUPPER ${lib} _LIB)
    list(APPEND MATLAB_LIBRARIES ${MATLAB_${_LIB}_LIBRARY})
endforeach ()

IF(MATLAB_INCLUDE_DIR AND MATLAB_LIBRARIES)
  SET(MATLAB_FOUND TRUE)
ENDIF(MATLAB_INCLUDE_DIR AND MATLAB_LIBRARIES)

# Find the MATLAB binary path
find_path(MATLAB_BINARY_DIR mex 
    HINTS
    $ENV{MATLAB_ROOT}
    PATHS
    /usr/local/MATLAB/2010b/bin
    /usr/local/MATLAB/2011a/bin
    /usr/local/MATLAB/2011b/bin
    /usr/local/MATLAB/2012a/bin
    /data/MATLAB/R2012a/bin
    /Applications/MATLAB_R2011a.app/bin
    /Applications/MATLAB_R2012a.app/bin
    /Applications/MATLAB_R2012b.app/bin
)

MARK_AS_ADVANCED(
  MATLAB_LIBRARIES
  MATLAB_MEX_LIBRARY
  MATLAB_MX_LIBRARY
  MATLAB_ENG_LIBRARY
  MATLAB_INCLUDE_DIR
  MATLAB_FOUND
  MATLAB_ROOT
)

