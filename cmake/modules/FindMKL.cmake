#
# Find Intel MKL.
#
# This script defines the following variables:
#  MKL_FOUND:        True if MKL is found.
#  MKL_INCLUDE_DIR:  MKL include directory.
#  MKL_LIBRARIES:    MKL libraries to link against.
#


# Find the include directory.
# ==============================================================================
find_path(MKL_INCLUDE_DIR mkl.h PATHS)

# Set the architecture specfic interface layer library name to look for.
# ==============================================================================
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(mkl_lib_names mkl_intel_lp64)
    set(intel_64 true)
else(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(mkl_lib_names mkl_intel)
    set(intel_64 false)
endif(CMAKE_SIZEOF_VOID_P EQUAL 8)


# Set the computation layer library name to look for. (see http://bit.ly/bMCczV)
# ==============================================================================
list(APPEND mkl_lib_names mkl_core)

# Set the threading model library name to look for.
# ==============================================================================
set(use_threaded_mkl false) # Turns of use of threaded mkl (default for pelican)

if(use_threaded_mkl)
    if(CMAKE_COMPILER_IS_GNUCXX)
        list(APPEND mkl_lib_names mkl_gnu_thread)
    else(CMAKE_COMPILER_IS_GNUCXX)
        list(APPEND mkl_lib_names mkl_intel_thread)
    endif(CMAKE_COMPILER_IS_GNUCXX)
else(use_threaded_mkl)
    list(APPEND mkl_lib_names mkl_sequential)
endif(use_threaded_mkl)


# Loop over required library names adding to MKL_LIBRARIES.
# ==============================================================================
foreach(mkl_lib ${mkl_lib_names})
    if (intel_64)
        find_library(${mkl_lib}_LIBRARY
            NAMES ${mkl_lib}
            HINTS ${MKL_INCLUDE_DIR}/../lib/intel64
            PATHS)
    else (intel_64)
        find_library(${mkl_lib}_LIBRARY
            NAMES ${mkl_lib}
            PATHS)
    endif (intel_64)
    set(tmp_library ${${mkl_lib}_LIBRARY})
    if (tmp_library)
        list(APPEND MKL_LIBRARIES ${tmp_library})
    endif(tmp_library)
endforeach(mkl_lib ${mkl_lib_names})


# Handle the QUIETLY and REQUIRED arguments.
# ==============================================================================
FIND_PACKAGE_HANDLE_STANDARD_ARGS(MKL DEFAULT_MSG MKL_LIBRARIES)

# Put variables in advanced section of cmake cache
# ==============================================================================
mark_as_advanced(MKL_LIBRARIES tmp_library)
