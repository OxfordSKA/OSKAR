# === Set the build type to release if not otherwise specified.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE release)
endif()

set(BUILD_SHARED_LIBS true)

# == C++ compiler.
if(CMAKE_COMPILER_IS_GNUCC) # || CMAKE_COMPILER_IS_GNUCXX ?!
	message("INFO: Setting compiler flags for gcc/g++.")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -DQT_NO_DEBUG -DQT_NO_DEBUG_OUTPUT")

    add_definitions(-Wall -Wextra -pedantic -std=c++0x)
    add_definitions(-Wcast-align)
    add_definitions(-Wcast-qual)
    add_definitions(-Wdisabled-optimization)
    add_definitions(-Wstrict-aliasing)
    add_definitions(-Wunknown-pragmas)
    #add_definitions(-Wconversion)
    #add_definitions(-Wno-deprecated -Wno-unknown-pragmas)
    #add_definitions(-Wfloat-equal)

elseif (NOT WIN32) # INTEL COMPILER
	message("INFO: Setting compiler flags for icc/icpc.")
    #set(CMAKE_CXX_FLAGS_RELEASE "-O3 -xHost -ipo -no-prec-div -DNDEBUG -DQT_NO_DEBUG -DQT_NO_DEBUG_OUTPUT")
    #set(CMAKE_CXX_FLAGS_RELEASE "-O2 -xHost -funroll-loops -ipo -no-prec-div -DNDEBUG -DQT_NO_DEBUG -DQT_NO_DEBUG_OUTPUT")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -DQT_NO_DEBUG -DQT_NO_DEBUG_OUTPUT")
    #add_definitions(-vec-report3)

    # Enable warning flags.
    # --------------------
    add_definitions(-Wall -Wcheck)
    #add_definitions(-Weffc++)

    # Suppress remarks and warnings.
    # -----------------------------
    add_definitions(-wd1418) # External function with no prior declaration.
    add_definitions(-wd1419) # External declaration in primary source file.
    add_definitions(-wd383)  # Value copied to temporary, reference to temporary used.
    add_definitions(-wd444)  # Destructor for base class not virtual.
    add_definitions(-wd981)  # Operands are evaluated in unspecified order.
    add_definitions(-wd177)  # Variable declared by never referenced.

    # Promote remarks to warnings.
    # ----------------------------
    add_definitions(-ww111)
    add_definitions(-ww1572)

elseif (MSVC) # visual studio compiler.
	message("INFO: Settings for compiler MSVC")
else ()
	message("INFO: Unknown compiler...")
endif ()

# == Cuda
if(CUDA_FOUND)
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    set(CUDA_BUILD_EMULATION OFF)
    if(CMAKE_BUILD_TYPE MATCHES RELEASE|[Rr]elease)
        set(CUDA_NVCC_FLAGS --compiler-options;-Wall;--compiler-options;-O2;-arch sm_13)
    else()
        set(CUDA_NVCC_FLAGS --compiler-options;-Wall;--compiler-options;-O0;--compiler-options;-g;-arch sm_13)
    endif()
endif()


# === Set some include directories at the project level.
include_directories(${oskar-lib_SOURCE_DIR} ${QT_INCLUDE_DIR})
if (CFitsio_FOUND)
    include_directories(${CFITSIO_INCLUDE_DIR})
endif()
