# - this module looks for Matlab
# Defines:
#  MATLAB_INCLUDE_DIR: include path for mex.h, engine.h
#  MATLAB_LIBRARIES:   required libraries: libmex, etc
#  MATLAB_MEX_LIBRARY: path to libmex.lib
#  MATLAB_MX_LIBRARY:  path to libmx.lib
#  MATLAB_ENG_LIBRARY: path to libeng.lib
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
#
#



SET(MATLAB_FOUND 0)
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
  # Regular x86
  IF(CMAKE_SIZEOF_VOID_P EQUAL 4)
    SET(MATLAB_SYS
      $ENV{MATLAB_ROOT}/bin/glnx86/
      /usr/local/MATLAB/R2010b/bin/glnx86/
      /usr/local/MATLAB/R2011a/bin/glnx86/
      /usr/local/MATLAB/R2011b/bin/glnx86/
      /usr/local/MATLAB/R2012a/bin/glnx86/
      /usr/local/matlab/bin/glnx86/
      )
  # AMD64:
  ELSE(CMAKE_SIZEOF_VOID_P EQUAL 4)
    SET(MATLAB_SYS
      $ENV{MATLAB_ROOT}/bin/glnxa64/
      /usr/local/MATLAB/R2010b/bin/glnxa64/
      /usr/local/MATLAB/R2011a/bin/glnxa64/
      /usr/local/MATLAB/R2011b/bin/glnxa64/
      /usr/local/MATLAB/R2012a/bin/glnxa64/
      /usr/local/matlab/bin/glnxa64/
      /data/MATLAB/R2011b/bin/glnxa64/
      /Applications/MATLAB_R2011a.app/bin/maci64
      )
  ENDIF(CMAKE_SIZEOF_VOID_P EQUAL 4)

  find_library(MATLAB_MEX_LIBRARY mex  PATHS ${MATLAB_SYS} NO_DEFAULT_PATH)
  find_library(MATLAB_MAT_LIBRARY mat  PATHS ${MATLAB_SYS} NO_DEFAULT_PATH)
  find_library(MATLAB_MX_LIBRARY  mx   PATHS ${MATLAB_SYS} NO_DEFAULT_PATH)
  find_library(MATLAB_ENG_LIBRARY eng  PATHS ${MATLAB_SYS} NO_DEFAULT_PATH)
  # HACK: find_library doesnt seem to be able to find versioned libraries... :(
  if (NOT APPLE)    
    find_file(MATLAB_QT_QTCORE_LIBRARY libQtCore.so.4 
      PATHS ${MATLAB_SYS} NO_DEFAULT_PATH)
  else ()
    find_package(Qt4 4.6 QUIET)
    set(MATLAB_QT_CORE_LIBRARY ${QT_QTCORE_LIBRARY})
  endif()

  find_path(MATLAB_INCLUDE_DIR
    "mex.h"
    PATHS
    $ENV{MATLAB_ROOT}/extern/include/
    /usr/local/MATLAB/R2010b/extern/include
    /usr/local/MATLAB/R2011a/extern/include
    /usr/local/MATLAB/R2011b/extern/include
    /usr/local/MATLAB/R2012a/extern/include
    /usr/local/matlab/extern/include/
    /data/MATLAB/R2011b/extern/include/
    /Applications/MATLAB_R2011a.app/extern/include/
    )
ENDIF(WIN32)

# This is common to UNIX and Win32:
SET(MATLAB_LIBRARIES
  ${MATLAB_MEX_LIBRARY}
  ${MATLAB_MAT_LIBRARY}
  ${MATLAB_MX_LIBRARY}
  ${MATLAB_ENG_LIBRARY}
)

IF(MATLAB_INCLUDE_DIR AND MATLAB_LIBRARIES)
  SET(MATLAB_FOUND 1)
ENDIF(MATLAB_INCLUDE_DIR AND MATLAB_LIBRARIES)

MARK_AS_ADVANCED(
  MATLAB_LIBRARIES
  MATLAB_MEX_LIBRARY
  MATLAB_MX_LIBRARY
  MATLAB_ENG_LIBRARY
  MATLAB_INCLUDE_DIR
  MATLAB_FOUND
  MATLAB_ROOT
)

