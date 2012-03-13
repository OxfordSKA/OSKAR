if (NOT CHECKED_DEPENDENCIES)
    message(FATAL_ERROR "Please include oskar_dependencies.cmake before this script!")
endif ()

# === Set the build type to release if not otherwise specified.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE release)
endif()

message("================================================================================")
if (CMAKE_BUILD_TYPE MATCHES RELEASE|[Rr]elease)
    message("-- INFO: Building in release mode.")
else ()
    message("-- INFO: Building in debug mode.")
endif()
message("================================================================================")

set(BUILD_SHARED_LIBS ON)

# === GNU compiler.
if (CMAKE_COMPILER_IS_GNUCC)
    # Enable warnings.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} -Wall -Wextra -pedantic")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wcast-align")
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} -Wcast-align")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wcast-qual")
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} -Wcast-qual")
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wconversion")
    #set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} -Wconversion")
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfloat-equal")
    #set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} -Wfloat-equal")

    # Disable specified warnings.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-long-long")
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} -Wno-long-long")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-variadic-macros")
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} -Wno-variadic-macros")

    # Ignore warnings from CUDA headers by specifying them as system headers.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem ${CUDA_INCLUDE_DIRS}")
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} -isystem ${CUDA_INCLUDE_DIRS}")

    # Add release and debug flags.
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -DQT_NO_DEBUG -DQT_NO_DEBUG_OUTPUT")
    set(CMAKE_C_FLAGS_RELEASE   "-O3 -DNDEBUG")
    set(CMAKE_Fortran_FLAGS_RELEASE "-O3 -DNDEBUG")
    set(CMAKE_Fortran_FLAGS_DEBUG "-g -O0")

# === Intel compiler.
elseif (NOT WIN32)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -DQT_NO_DEBUG -DQT_NO_DEBUG_OUTPUT")
    list(APPEND CMAKE_C_FLAGS "-fPIC -std=c99")
    list(APPEND CMAKE_CXX_FLAGS "-fPIC")
    add_definitions(-Wall)
    add_definitions(-Wcheck)
    add_definitions(-wd2259)
    add_definitions(-wd1125)

# === Microsoft visual studio compiler.
elseif (MSVC) # visual studio compiler.
    add_definitions(/wd4100) # NEED TO FIX ALL THESE!
    add_definitions(/wd4305) # NEED TO FIX ALL THESE!
    add_definitions(/wd4244) # NEED TO FIX ALL THESE!
    add_definitions(-DQWT_DLL)

# === No compiler found.
else ()
    message("-- INFO: Unknown compiler.")
endif ()

# === Cuda
if (CUDA_FOUND)
    # Use a separate set of flags for CUDA.
    if (NOT MSVC)
        set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    endif ()
    set(CUDA_VERBOSE_BUILD OFF)

    # Default flags.
    if (NOT WIN32)
        set(CUDA_NVCC_FLAGS --compiler-options;-Wall;)
        list(APPEND CUDA_NVCC_FLAGS --compiler-options;-Wextra;)
        list(APPEND CUDA_NVCC_FLAGS --compiler-options;-Wno-unused-parameter;)
        #list(APPEND CUDA_NVCC_FLAGS --compiler-options;-pedantic;)
        list(APPEND CUDA_NVCC_FLAGS --compiler-options;-Wno-variadic-macros;)
        list(APPEND CUDA_NVCC_FLAGS --compiler-options;-Wno-long-long;)
        # Disable warning about missing initializers (for CUDA Thrust).
        list(APPEND CUDA_NVCC_FLAGS --compiler-options;-Wno-missing-field-initializers;)
    endif ()

    # Build mode specific flags.
    if (CMAKE_BUILD_TYPE MATCHES RELEASE|[Rr]elease)
        if (MSVC)
            list(APPEND CUDA_NVCC_FLAGS --compiler-options;/wd4100;)
        else ()
            list(APPEND CUDA_NVCC_FLAGS --compiler-options;-O2;)
        endif ()
        list(APPEND CUDA_NVCC_FLAGS -O2;)
        list(APPEND CUDA_NVCC_FLAGS --compiler-options;-fPIC;)
        #list(APPEND CUDA_NVCC_FLAGS --ptxas-options=-v;)
        #list(APPEND CUDA_NVCC_FLAGS --ptxas-options=-dlcm=cg)
    else ()
        if (NOT MSVC)
            list(APPEND CUDA_NVCC_FLAGS --compiler-options;-O0;)
            list(APPEND CUDA_NVCC_FLAGS --compiler-options;-g;)
        endif ()
        list(APPEND CUDA_NVCC_FLAGS -g;)
        list(APPEND CUDA_NVCC_FLAGS -O0;)
        list(APPEND CUDA_NVCC_FLAGS --compiler-options;-fPIC;)
        list(APPEND CUDA_NVCC_FLAGS --ptxas-options=-v;)
    endif ()

    message("================================================================================")
    if (NOT DEFINED CUDA_ARCH)
        message("-- INFO: Building CUDA device code for architecture 2.0.")
        message("-- INFO: The target CUDA architecture can be changed by using the option:")
        message("-- INFO:   -DCUDA_ARCH=<arch>")
        message("-- INFO: where <arch> is one of:")
        message("-- INFO:   1.1, 1.3, 2.0, 2.1, or ALL.")
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
    elseif (CUDA_ARCH MATCHES "ALL")
        message("-- INFO: Building CUDA device code for 1.1, 1.3, 2.0 & 2.1")
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_11,code=sm_11)
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_13,code=sm_13)
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_21)
    else()
        message(FATAL_ERROR "-- CUDA_ARCH ${CUDA_ARCH} not recognised!")
    endif()
    message("================================================================================")
    add_definitions(-DCUDA_ARCH=${CUDA_ARCH})
endif (CUDA_FOUND)


# === MATLAB mex functions.
if (MATLAB_FOUND)
    if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        set(MATLAB_MEXFILE_EXT mexmaci64)
        link_directories(/usr/local/cuda/lib/)
        link_directories(/Applications/MATLAB_R2011a.app/bin/maci64/)
        set(MATLAB_CXX_FLAGS "-DMATLAB_MEX_FILE -DMX_COMPAT_32 -pthread -fPIC -flat_namespace -undefined suppress")
    else ()
        set(MATLAB_MEXFILE_EXT mexa64)
        #set(MATLAB_CXX_FLAGS "-DMATLAB_MEX_FILE -DMX_COMPAT_32 -pthread -fPIC")
        set(MATLAB_CXX_FLAGS "-DMATLAB_MEX_FILE -pthread -fPIC")
    endif ()
    include_directories(${MATLAB_INCLUDE_DIR})
endif ()


# === Set some include directories at the project level.
include_directories(${oskar-lib_SOURCE_DIR})
if (CFitsio_FOUND)
    include_directories(${CFITSIO_INCLUDE_DIR})
endif()
