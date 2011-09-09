if (NOT CHECKED_DEPENDENCIES)
    message(FATAL_ERROR "Please include dependencies.cmake before this script!")
endif ()

# === Set the build type to release if not otherwise specified.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE release)
endif()

message("*****************************************************************")
if (CMAKE_BUILD_TYPE MATCHES RELEASE|[Rr]elease)
    message("** NOTE: Building in release mode!")
else ()
    message("** NOTE: Building in debug mode!")
endif()
message("*****************************************************************")

set(BUILD_SHARED_LIBS true)

# === GNU C++ compiler.
if (CMAKE_COMPILER_IS_GNUCC) # || CMAKE_COMPILER_IS_GNUCXX ?!
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -DQT_NO_DEBUG -DQT_NO_DEBUG_OUTPUT")
    set(CMAKE_C_FLAGS_RELEASE "-O3 -std=c99 -DNDEBUG -DQT_NO_DEBUG -DQT_NO_DEBUG_OUTPUT")
#    add_definitions(-Wall -Wextra -pedantic -std=c++0x)
    add_definitions(-Wall -Wextra)
    add_definitions(-Wcast-align)
    add_definitions(-Wcast-qual)
    add_definitions(-Wdisabled-optimization)
    add_definitions(-Wstrict-aliasing)
    add_definitions(-Wunknown-pragmas)
    #add_definitions(-Wconversion)
    #add_definitions(-Wno-deprecated -Wno-unknown-pragmas)
    #add_definitions(-Wfloat-equal)

# === Intel compiler.
elseif (NOT WIN32)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -DQT_NO_DEBUG -DQT_NO_DEBUG_OUTPUT")
    add_definitions(-Wall -Wcheck)
    add_definitions(-wd1418) # External function with no prior declaration.
    add_definitions(-wd1419) # External declaration in primary source file.
    add_definitions(-wd383)  # Value copied to temporary, reference to temporary used.
    add_definitions(-wd444)  # Destructor for base class not virtual.
    add_definitions(-wd981)  # Operands are evaluated in unspecified order.
    add_definitions(-wd177)  # Variable declared by never referenced.
    add_definitions(-ww111)  # Promote remark 111 to warning.
    add_definitions(-ww1572) # Promote remark 1572 to warning.

# === Microsoft visual studio compiler.
elseif (MSVC) # visual studio compiler.
    add_definitions(/wd4100) # NEED TO FIX ALL THESE!
    add_definitions(/wd4305) # NEED TO FIX ALL THESE!
    add_definitions(/wd4244) # NEED TO FIX ALL THESE!
    add_definitions(-DQWT_DLL)

# === No compiler found.
else ()
    message("INFO: Unknown compiler...")
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
    endif ()

    # Build mode specific flags.
    if (CMAKE_BUILD_TYPE MATCHES RELEASE|[Rr]elease)
        if (MSVC)
            list(APPEND CUDA_NVCC_FLAGS --compiler-options;/wd4100;)
        else ()
            list(APPEND CUDA_NVCC_FLAGS --compiler-options;-O2;)
        endif ()
        list(APPEND CUDA_NVCC_FLAGS -O2;)
        #list(APPEND CUDA_NVCC_FLAGS --ptxas-options=-v;)

        #list(APPEND CUDA_NVCC_FLAGS --ptxas-options=-dlcm=cg)
    else ()
        if (NOT MSVC)
            list(APPEND CUDA_NVCC_FLAGS --compiler-options;-O0;)
            list(APPEND CUDA_NVCC_FLAGS --compiler-options;-g;)
        endif ()
        list(APPEND CUDA_NVCC_FLAGS -g;)
        list(APPEND CUDA_NVCC_FLAGS -O0;)
        #list(APPEND CUDA_NVCC_FLAGS --ptxas-options=-v')
    endif ()

    if (CUDA_ARCH MATCHES 1.1)
        message("== INFO: Building CUDA device code for architecture 1.1")
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_11,code=sm_11)
    elseif (CUDA_ARCH MATCHES 1.3)
        message("== INFO: Building CUDA device code for architecture 1.3")
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_13,code=sm_13)
    elseif (CUDA_ARCH MATCHES 2.0)
        message("== INFO: Building CUDA device code for architecture 2.0")
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
    elseif (CUDA_ARCH MATCHES 2.1)
        message("== INFO: Building CUDA device code for architecture 2.1 (default)")
        message("== INFO: Run cmake with -DCUDA_ARCH=<arch> to set a different target.")
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_21)
    else()
        message(WARNING "*** CUDA_ARCH ${CUDA_ARCH} not recognised! ***")
    endif()
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
        set(MATLAB_CXX_FLAGS "-DMATLAB_MEX_FILE -DMX_COMPAT_32 -pthread -fPIC")
    endif ()
    include_directories(${MATLAB_INCLUDE_DIR})
endif ()


# === Set some include directories at the project level.
include_directories(${oskar-lib_SOURCE_DIR})
if (CFitsio_FOUND)
    include_directories(${CFITSIO_INCLUDE_DIR})
endif()
