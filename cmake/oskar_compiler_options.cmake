if (NOT CHECKED_DEPENDENCIES)
    message(FATAL_ERROR "Please include oskar_dependencies.cmake before this script!")
endif ()

# Set the build type to release if not otherwise specified.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE release)
endif()

message("===============================================================================")
if (CMAKE_BUILD_TYPE MATCHES RELEASE|[Rr]elease)
    message("-- INFO: Building in release mode.")
else ()
    message("-- INFO: Building in debug mode.")
endif()
message("===============================================================================")

set(BUILD_SHARED_LIBS ON)

# Set general compiler flags.
# ------------------------------------------------------------------------------
if (NOT WIN32)
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -fPIC -DNDEBUG -DQT_NO_DEBUG -DQT_NO_DEBUG_OUTPUT")
    set(CMAKE_C_FLAGS_RELEASE   "-O2 -fPIC -DNDEBUG")

    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -fPIC -g -Wall")
    set(CMAKE_C_FLAGS_DEBUG   "-O0 -fPIC -g -Wall")

    if (CMAKE_COMPILER_IS_GNUCC)
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wextra")
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -pedantic")
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wcast-align")
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wcast-qual")
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wno-long-long")
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wno-variadic-macros")
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wno-unused-function")
    endif ()

    if (CMAKE_COMPILER_IS_GNUCXX)
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wextra")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -pedantic")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wcast-align")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wcast-qual")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wno-long-long")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wno-variadic-macros")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wno-unused-function")
    endif ()

    if (${CMAKE_C_COMPILER} MATCHES "icc.*$")
        set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG}   -wd2259")
        set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG}   -wd1125")
    endif ()

    if (${CMAKE_CXX_COMPILER} MATCHES "icpc.*$")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -wd2259")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -wd1125")
    endif ()
elseif (MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /D QT_NO_DEBUG /D QT_NO_DEBUG_OUTPUT")

    # Disable warning about loss of precision converting double to float.
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /wd4244")
    set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE}   /wd4244")
    set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG}   /wd4244")
    set(CMAKE_C_FLAGS_DEBUG     "${CMAKE_C_FLAGS_DEBUG}     /wd4244")

    # Disable nonsensical warning about fopen.
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /wd4996")
    set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE}   /wd4996")
    set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG}   /wd4996")
    set(CMAKE_C_FLAGS_DEBUG     "${CMAKE_C_FLAGS_DEBUG}     /wd4996")
endif ()


# Set CUDA releated compiler flags.
# ------------------------------------------------------------------------------
if (CUDA_FOUND)
    if (NOT WIN32)
        set(CUDA_PROPAGATE_HOST_FLAGS OFF)
        set(CUDA_VERBOSE_BUILD OFF)

        # General NVCC compiler options.
        list(APPEND CUDA_NVCC_FLAGS_RELEASE -O2;)
        list(APPEND CUDA_NVCC_FLAGS_DEBUG -O0;)
        list(APPEND CUDA_NVCC_FLAGS_DEBUG -g;)

        # Options passed to the compiler NVCC encapsulates.
        list(APPEND CUDA_NVCC_FLAGS_RELEASE --compiler-options;-O2;)
        list(APPEND CUDA_NVCC_FLAGS_DEBUG --compiler-options;-O0;)
        list(APPEND CUDA_NVCC_FLAGS_DEBUG --compiler-options;-g;)

        if (CMAKE_COMPILER_IS_GNUCC)
            list(APPEND CUDA_NVCC_FLAGS_RELEASE --compiler-options;-fPIC;)
            list(APPEND CUDA_NVCC_FLAGS_DEBUG --compiler-options;-fPIC;)
            list(APPEND CUDA_NVCC_FLAGS_DEBUG --compiler-options;-Wall;)
            list(APPEND CUDA_NVCC_FLAGS_DEBUG --compiler-options;-Wextra;)
            list(APPEND CUDA_NVCC_FLAGS_DEBUG --compiler-options;-Wno-unused-parameter;)
            list(APPEND CUDA_NVCC_FLAGS_DEBUG --compiler-options;-Wno-variadic-macros;)
            list(APPEND CUDA_NVCC_FLAGS_DEBUG --compiler-options;-Wno-long-long;)
            # Disable warning about missing initializers (for CUDA Thrust).
            list(APPEND CUDA_NVCC_FLAGS_DEBUG --compiler-options;-Wno-missing-field-initializers;)
            # Disable warning about "unsigned int* __get_precalculated_matrix(int) defined but not used".
            list(APPEND CUDA_NVCC_FLAGS_DEBUG --compiler-options;-Wno-unused-function;)
            # Ignore warnings from CUDA headers by specifying them as system headers.
            set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -isystem ${CUDA_INCLUDE_DIRS}")
            set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG} -isystem ${CUDA_INCLUDE_DIRS}")
            if (NOT APPLE)
                # Disable warning about "variable '__f' set but not used".
                list(APPEND CUDA_NVCC_FLAGS_DEBUG --compiler-options;-Wno-unused-but-set-variable;)
            endif ()
        endif()

        # PTX compiler options
        #list(APPEND CUDA_NVCC_FLAGS_DEBUG --ptxas-options=-v;)
    endif ()

    message("===============================================================================")
    if (NOT DEFINED CUDA_ARCH)
        message("-- INFO: Building CUDA device code for architecture 2.0.")
        message("-- INFO: The target CUDA architecture can be changed by using the option:")
        message("-- INFO:   -DCUDA_ARCH=<arch>")
        message("-- INFO: where <arch> is one of:")
        message("-- INFO:   1.3, 2.0, 2.1, 3.0, or ALL.")
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
    elseif (CUDA_ARCH MATCHES 1.1)
        message("-- INFO: Building CUDA device code for architecture 1.1")
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_11,code=sm_11)
    elseif (CUDA_ARCH MATCHES 1.3)
        message("-- INFO: Building CUDA device code for architecture 1.3")
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_13,code=sm_13)
    elseif (CUDA_ARCH MATCHES 2.0)
        message("-- INFO: Building CUDA device code for architecture 2.0")
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
    elseif (CUDA_ARCH MATCHES 2.1)
        message("-- INFO: Building CUDA device code for architecture 2.1")
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_21)
    elseif (CUDA_ARCH MATCHES 3.0)
        message("-- INFO: Building CUDA device code for architecture 3.0")
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
    elseif (CUDA_ARCH MATCHES ALL|[Aa]ll)
        message("-- INFO: Building CUDA device code all supported architectures")
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_11,code=sm_11)
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_13,code=sm_13)
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_21)
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
    else()
        message(FATAL_ERROR "-- CUDA_ARCH ${CUDA_ARCH} not recognised!")
    endif()
    message("===============================================================================")
    add_definitions(-DCUDA_ARCH=${CUDA_ARCH})
endif (CUDA_FOUND)


# Set MATLAB mex function compiler flags.
# ------------------------------------------------------------------------------
if (MATLAB_FOUND)
    if (APPLE)
        set(MATLAB_MEXFILE_EXT mexmaci64)
        link_directories(/usr/local/cuda/lib/)
        link_directories(/Applications/MATLAB_R2011a.app/bin/maci64/)
        set(MATLAB_CXX_FLAGS "-DMATLAB_MEX_FILE -DMX_COMPAT_32 -pthread -flat_namespace -undefined suppress")
    else ()
        set(MATLAB_MEXFILE_EXT mexa64)
        #set(MATLAB_CXX_FLAGS "-DMATLAB_MEX_FILE -DMX_COMPAT_32 -pthread")
        set(MATLAB_CXX_FLAGS "-DMATLAB_MEX_FILE -pthread")
    endif ()
endif ()

