if (NOT CHECKED_DEPENDENCIES)
    message(FATAL_ERROR "Please include oskar_dependencies.cmake before this script!")
endif ()

# Set the build type to release if not otherwise specified.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE release)
endif()

message("===============================================================================")
message("-- INFO: Build type: ${CMAKE_BUILD_TYPE}")
message("===============================================================================")

set(BUILD_SHARED_LIBS ON)

# Set the include path to include the top-level folder and sub-folders for
# main oskar library.
# ------------------------------------------------------------------------------
include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/apps
    ${PROJECT_SOURCE_DIR}/apps/lib
    ${PROJECT_SOURCE_DIR}/correlate
    ${PROJECT_SOURCE_DIR}/extern/gtest-1.7.0/include
    ${PROJECT_SOURCE_DIR}/imaging
    ${PROJECT_SOURCE_DIR}/interferometry
    ${PROJECT_SOURCE_DIR}/jones
    ${PROJECT_SOURCE_DIR}/math
    ${PROJECT_SOURCE_DIR}/measures
    ${PROJECT_SOURCE_DIR}/settings
    ${PROJECT_SOURCE_DIR}/settings/load
    ${PROJECT_SOURCE_DIR}/settings/widgets
    ${PROJECT_SOURCE_DIR}/settings/struct
    ${PROJECT_SOURCE_DIR}/sky
    ${PROJECT_SOURCE_DIR}/splines
    ${PROJECT_SOURCE_DIR}/station
    ${PROJECT_SOURCE_DIR}/utility
    ${PROJECT_SOURCE_DIR}/utility/log
    ${PROJECT_SOURCE_DIR}/widgets
)



# Set general compiler flags.
# ------------------------------------------------------------------------------
if (NOT WIN32)
    if (NOT APPLE)
        set(CMAKE_CXX_FLAGS "-fPIC")
        set(CMAKE_C_FLAGS "-fPIC")
    endif ()

    if (CMAKE_COMPILER_IS_GNUCXX)    
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility-inlines-hidden")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-show-option")
    endif ()

    if (CMAKE_COMPILER_IS_GNUCC)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=hidden")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fdiagnostics-show-option")
    endif ()

    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG -DQT_NO_DEBUG -DQT_NO_DEBUG_OUTPUT")
    set(CMAKE_C_FLAGS_RELEASE   "-O2 -DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -Wall -Wno-unused-function")
    set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -g -Wall -Wno-unused-function")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall")
    set(CMAKE_C_FLAGS_DEBUG   "-O0 -g -Wall")
    set(CMAKE_CXX_FLAGS_MINSIZEREL "-O1 -DNDEBUG -DQT_NO_DEBUG -DQT_NO_DEBUG_OUTPUT")
    set(CMAKE_C_FLAGS_MINSIZEREL "-O1 -DNDEBUG -DQT_NO_DEBUG -DQT_NO_DEBUG_OUTPUT")

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
        set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG}   -Wcheck")
        set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG}   -wd2259")
        set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG}   -wd1125")
    endif ()

    if (${CMAKE_CXX_COMPILER} MATCHES "icpc.*$")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wcheck")
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

# Rpath settings for OS X
# ------------------------------------------------------------------------------
if (APPLE)
    set(CMAKE_INSTALL_NAME_DIR "@rpath")
endif (APPLE)

# Set CUDA releated compiler flags.
# --compiler-options or -Xcompiler: specify options directly to the compiler
#                                   that nvcc encapsulates.
# ------------------------------------------------------------------------------
if (CUDA_FOUND)
    if (NOT WIN32)
        set(CUDA_PROPAGATE_HOST_FLAGS OFF)
        set(CUDA_VERBOSE_BUILD OFF)

        # General NVCC compiler options.
        set(CUDA_NVCC_FLAGS_RELEASE "-O2")
        set(CUDA_NVCC_FLAGS_DEBUG "-O0 -g --generate-line-info")
        set(CUDA_NVCC_FLAGS_RELWIDTHDEBINFO "-02 -g --generate-line-info")
        set(CUDA_NVCC_FLAGS_MINSIZEREL -01)

        # Options passed to the compiler NVCC encapsulates.
        list(APPEND CUDA_NVCC_FLAGS_RELEASE -Xcompiler;-O2)
        list(APPEND CUDA_NVCC_FLAGS_DEBUG -Xcompiler;-O0)
        list(APPEND CUDA_NVCC_FLAGS_DEBUG -Xcompiler;-g)
        list(APPEND CUDA_NVCC_FLAGS_RELWIDTHDEBINFO -Xcompiler;-02)
        list(APPEND CUDA_NVCC_FLAGS_RELWIDTHDEBINFO -Xcompiler;-g)
        list(APPEND CUDA_NVCC_FLAGS_MINSIZEREL -Xcompiler;-01)

        if (CMAKE_COMPILER_IS_GNUCC)
            list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-fvisibility=hidden;)
            if (NOT APPLE)
                list(APPEND CUDA_NVCC_FLAGS_RELEASE -Xcompiler;-fPIC;)
                list(APPEND CUDA_NVCC_FLAGS_DEBUG -Xcompiler;-fPIC;)
            endif()
            list(APPEND CUDA_NVCC_FLAGS_DEBUG -Xcompiler;-Wall;)
            list(APPEND CUDA_NVCC_FLAGS_DEBUG -Xcompiler;-Wextra;)
            list(APPEND CUDA_NVCC_FLAGS_DEBUG -Xcompiler;-Wno-unused-parameter;)
            list(APPEND CUDA_NVCC_FLAGS_DEBUG -Xcompiler;-Wno-variadic-macros;)
            list(APPEND CUDA_NVCC_FLAGS_DEBUG -Xcompiler;-Wno-long-long;)
            # Disable warning about missing initializers (for CUDA Thrust).
            list(APPEND CUDA_NVCC_FLAGS_DEBUG -Xcompiler;-Wno-missing-field-initializers;)
            # Disable warning about "unsigned int* __get_precalculated_matrix(int) defined but not used".
            list(APPEND CUDA_NVCC_FLAGS_DEBUG -Xcompiler;-Wno-unused-function;)
            # Ignore warnings from CUDA headers by specifying them as system headers.
            set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -isystem ${CUDA_INCLUDE_DIRS}")
            set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG} -isystem ${CUDA_INCLUDE_DIRS}")
            if (NOT APPLE)
                # Disable warning about "variable '__f' set but not used".
                list(APPEND CUDA_NVCC_FLAGS_DEBUG -Xcompiler;-Wno-unused-but-set-variable;)
            endif ()
        endif()
        # PTX compiler options
        #list(APPEND CUDA_NVCC_FLAGS_RELEASE --ptxas-options=-v;)
    endif ()

    message("===============================================================================")
    if (NOT DEFINED CUDA_ARCH)
        # TODO change to a fail in the case device is not set?
        message("-- INFO: Building CUDA device code for architecture 2.0.")
        message("-- INFO: The target CUDA architecture can be changed by using the option:")
        message("-- INFO:   -DCUDA_ARCH=<arch>")
        message("-- INFO: where <arch> is one of:")
        message("-- INFO:   1.3, 2.0, 2.1, 3.0, 3.5, or ALL.")
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
    elseif (CUDA_ARCH MATCHES 3.5)
        message("-- INFO: Building CUDA device code for architecture 3.5")
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
    elseif (CUDA_ARCH MATCHES ALL|[Aa]ll)
        message("-- INFO: Building CUDA device code all supported architectures")
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_11,code=sm_11)
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_13,code=sm_13)
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_21)
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
    else()
        message(FATAL_ERROR "-- CUDA_ARCH ${CUDA_ARCH} not recognised!")
    endif()
    message("===============================================================================")
    add_definitions(-DCUDA_ARCH=${CUDA_ARCH})
endif (CUDA_FOUND)

# Configure MSVC runtime.
if (MSVC)
    # Default to dynamically-linked runtime.
    if ("${MSVC_RUNTIME}" STREQUAL "")
        set (MSVC_RUNTIME "dynamic")
    endif ()
    # Set compiler options.
    set(vars
        CMAKE_C_FLAGS_DEBUG
        CMAKE_C_FLAGS_MINSIZEREL
        CMAKE_C_FLAGS_RELEASE
        CMAKE_C_FLAGS_RELWITHDEBINFO
        CMAKE_CXX_FLAGS_DEBUG
        CMAKE_CXX_FLAGS_MINSIZEREL
        CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_RELWITHDEBINFO
    )
    if (${MSVC_RUNTIME} STREQUAL "static")
        message(STATUS "MSVC: Using statically-linked runtime.")
        foreach (var ${vars})
            if (${var} MATCHES "/MD")
                string(REGEX REPLACE "/MD" "/MT" ${var} "${${var}}")
            endif ()
        endforeach ()
    else ()
        message(STATUS "MSVC: Using dynamically-linked runtime.")
        foreach (var ${vars})
            if (${var} MATCHES "/MT")
                string(REGEX REPLACE "/MT" "/MD" ${var} "${${var}}")
            endif ()
        endforeach ()
    endif ()
endif ()

if (BUILD_INFO)
    message(STATUS "")
    message(STATUS "****************************************************************************")
    message(STATUS "Compiler Options:")
    message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
    message(STATUS "  C++ compiler: ${CMAKE_CXX_COMPILER}")
    message(STATUS "  C compiler: ${CMAKE_C_COMPILER}")
    if (${CMAKE_BUILD_TYPE} MATCHES release)
        message(STATUS "  C++ flags: ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
        message(STATUS "  C flags: ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELEASE}")
        message(STATUS "  CUDA flags: ${CUDA_NVCC_FLAGS} ${CUDA_NVCC_FLAGS_RELEASE}")
    elseif (${CMAKE_BUILD_TYPE} MATCHES debug)
        message(STATUS "  C++ flags: ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
        message(STATUS "  C flags: ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG}")
        message(STATUS "  CUDA flags: ${CUDA_NVCC_FLAGS} ${CUDA_NVCC_FLAGS_DEBUG}")
    elseif (${CMAKE_BUILD_TYPE} MATCHES relwithdebinfo)
        message(STATUS "  C++ flags: ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
        message(STATUS "  C flags: ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELWITHDEBINFO}")
        message(STATUS "  CUDA flags: ${CUDA_NVCC_FLAGS} ${CUDA_NVCC_FLAGS_RELWITHDEBINFO}")
    elseif (${CMAKE_BUILD_TYPE} MATCHES minsizerel)
        message(STATUS "  C++ flags: ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_MINSIZEREL}")
        message(STATUS "  C flags: ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_MINSIZEREL}")
        message(STATUS "  CUDA flags: ${CUDA_NVCC_FLAGS} ${CUDA_NVCC_FLAGS_MINSIZEREL}")
    endif ()
    message(STATUS "****************************************************************************")
    message(STATUS "")
endif (BUILD_INFO)

