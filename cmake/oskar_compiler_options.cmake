
#! TODO remove the need to check this ....
if (NOT CHECKED_DEPENDENCIES)
    message(FATAL_ERROR "Please include oskar_dependencies.cmake before this script!")
endif ()

# Automatically set the build type if not specified.
if (NOT CMAKE_BUILD_TYPE)
    # Use debug mode if building in dbg or debug directory.
    get_filename_component(dirname ${CMAKE_BINARY_DIR} NAME)
    if (${dirname} MATCHES "dbg" OR ${dirname} MATCHES "debug")
        set(CMAKE_BUILD_TYPE Debug)
    else()
        set(CMAKE_BUILD_TYPE Release)
    endif()
    message(STATUS "INFO: Setting CMAKE_BUILD_TYPE to ${CMAKE_BUILD_TYPE}")
endif()

set(BUILD_SHARED_LIBS ON)

# Build the various version strings to be passed to the code.
if (CMAKE_VERSION VERSION_GREATER 2.8.11)
    string(TIMESTAMP OSKAR_BUILD_DATE "%Y-%m-%d %H:%M:%S")
endif()

macro(APPEND_FLAGS FLAG_VAR)
    foreach (flag ${ARGN})
        set(${FLAG_VAR} "${${FLAG_VAR}} ${flag}")
    endforeach()
endmacro()

# Set general compiler flags.
if (NOT WIN32)
    set(CMAKE_C_FLAGS "-fPIC -std=c99")
    set(CMAKE_C_FLAGS_RELEASE "-O2 -DNDEBUG")
    set(CMAKE_C_FLAGS_DEBUG "-O0 -g -Wall")
    set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -g -Wall")
    set(CMAKE_C_FLAGS_MINSIZEREL "-O1 -DNDEBUG -DQT_NO_DEBUG -DQT_NO_DEBUG_OUTPUT")
    set(CMAKE_CXX_FLAGS "-fPIC")
    if ("${CMAKE_CXX_COMPILER_ID}" MATCHES ".*Clang.*")
        # Tell Clang to use c++-11, required for casacore headers.
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++ -std=c++11")
    endif()
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG -DQT_NO_DEBUG -DQT_NO_DEBUG_OUTPUT")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -Wall")
    set(CMAKE_CXX_FLAGS_MINSIZEREL "-O1 -DNDEBUG -DQT_NO_DEBUG -DQT_NO_DEBUG_OUTPUT")

    if ("${CMAKE_C_COMPILER_ID}" MATCHES ".*Clang.*"
            OR "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
        # Treat external code as system headers.
        # This avoids a number of warning supression flags.
        append_flags(CMAKE_CXX_FLAGS
            -isystem ${GTEST_INCLUDE_DIR}
            -isystem ${GTEST_INCLUDE_DIR}/internal
            -isystem ${CASACORE_INCLUDE_DIR}/casacore
        )
        append_flags(CMAKE_C_FLAGS
            -fvisibility=hidden
            -fdiagnostics-show-option)

        # Note: long-long is required for cfitsio
        append_flags(CMAKE_C_FLAGS_DEBUG
            -Wextra -pedantic -Wcast-qual -Wcast-align
            -Wmissing-prototypes -Wno-long-long
            -Wno-variadic-macros -Wno-unused-function
         )
         # Additional test flags
#        append_flags(CMAKE_C_FLAGS_DEBUG
#            -Wbad-function-cast -Wstack-protector -Wpacked
#            -Wredundant-decls -Wshadow -Wwrite-strings
#            -Waggregate-return -Wstrict-prototypes
#            -Wstrict-aliasing -Wdeclaration-after-statement
#        )
        append_flags(CMAKE_C_FLAGS_RELWITHDEBINFO -Wno-unused-function)
        append_flags(CMAKE_CXX_FLAGS
            -fvisibility=hidden -fvisibility-inlines-hidden
            -fdiagnostics-show-option
        )
        append_flags(CMAKE_CXX_FLAGS_DEBUG
            -Wextra -pedantic -Wcast-qual -Wcast-align -Wno-long-long
            -Wno-variadic-macros -Wno-unused-function
        )
        append_flags(CMAKE_CXX_FLAGS_RELWITHDEBINFO -Wno-unused-function)

        if ("${CMAKE_CXX_COMPILER_ID}" MATCHES ".*Clang.*" AND FORCE_LIBSTDC++)
            # Tell Clang to use libstdc++ rather than libc++
            # This is required if any of the OSKAR dependencies are built
            # against libstdc++ due to ABI incompatibility with libc++.
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
        endif()

    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        # Using Intel compilers.
    endif()
else()
    if ("${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /D QT_NO_DEBUG /D QT_NO_DEBUG_OUTPUT")

        # Disable warning about loss of precision converting double to float.
        set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE}   /wd4244")
        set(CMAKE_C_FLAGS_DEBUG     "${CMAKE_C_FLAGS_DEBUG}     /wd4244")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /wd4244")
        set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG}   /wd4244")

        # Disable nonsensical warning about fopen.
        set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE}   /wd4996")
        set(CMAKE_C_FLAGS_DEBUG     "${CMAKE_C_FLAGS_DEBUG}     /wd4996")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /wd4996")
        set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG}   /wd4996")
    endif()
endif ()

# Rpath settings for macOS
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
        if ("${CMAKE_C_COMPILER_ID}" MATCHES ".*Clang.*"
                OR "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
            append_flags(CMAKE_C_FLAGS -isystem ${CUDA_INCLUDE_DIRS})
            append_flags(CMAKE_CXX_FLAGS -isystem ${CUDA_INCLUDE_DIRS})
        endif()
        set(CUDA_PROPAGATE_HOST_FLAGS OFF)
        set(CUDA_VERBOSE_BUILD OFF)

        # General NVCC compiler options.
        set(CUDA_NVCC_FLAGS_RELEASE "-O2")
        set(CUDA_NVCC_FLAGS_DEBUG "-O0 -g --generate-line-info")
        set(CUDA_NVCC_FLAGS_RELWIDTHDEBINFO "-02 -g --generate-line-info")
        set(CUDA_NVCC_FLAGS_MINSIZEREL -01)
        if (DEFINED NVCC_COMPILER_BINDIR)
            append_flags(CUDA_NVCC_FLAGS_RELEASE -ccbin ${NVCC_COMPILER_BINDIR})
            append_flags(CUDA_NVCC_FLAGS_DEBUG -ccbin ${NVCC_COMPILER_BINDIR})
            append_flags(CUDA_NVCC_FLAGS_RELWIDTHDEBINFO -ccbin ${NVCC_COMPILER_BINDIR})
            append_flags(CUDA_NVCC_FLAGS_MINSIZEREL -ccbin ${NVCC_COMPILER_BINDIR})
        endif()

        # Options passed to the compiler NVCC encapsulates.
        if (FORCE_LIBSTDC++)
            list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-stdlib=libstdc++;)
        endif()
        if (APPLE AND (DEFINED CMAKE_OSX_DEPLOYMENT_TARGET))
            list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET};)
        endif()
        list(APPEND CUDA_NVCC_FLAGS_RELEASE -Xcompiler;-O2)
        list(APPEND CUDA_NVCC_FLAGS_DEBUG -Xcompiler;-O0)
        list(APPEND CUDA_NVCC_FLAGS_DEBUG -Xcompiler;-g)
        list(APPEND CUDA_NVCC_FLAGS_RELWIDTHDEBINFO -Xcompiler;-02)
        list(APPEND CUDA_NVCC_FLAGS_RELWIDTHDEBINFO -Xcompiler;-g)
        list(APPEND CUDA_NVCC_FLAGS_MINSIZEREL -Xcompiler;-01)

        list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-fvisibility=hidden;)
        list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-fPIC;)
        list(APPEND CUDA_NVCC_FLAGS_DEBUG -Xcompiler;-Wall;)
        list(APPEND CUDA_NVCC_FLAGS_DEBUG -Xcompiler;-Wextra;)
        list(APPEND CUDA_NVCC_FLAGS_DEBUG -Xcompiler;-Wno-unused-private-field;)
        list(APPEND CUDA_NVCC_FLAGS_DEBUG -Xcompiler;-Wno-unused-parameter;)
        list(APPEND CUDA_NVCC_FLAGS_DEBUG -Xcompiler;-Wno-variadic-macros;)
        list(APPEND CUDA_NVCC_FLAGS_DEBUG -Xcompiler;-Wno-long-long;)
        # Disable warning about missing initializers (for CUDA Thrust).
        list(APPEND CUDA_NVCC_FLAGS_DEBUG -Xcompiler;-Wno-missing-field-initializers;)
        # Disable warning about "unsigned int* __get_precalculated_matrix(int) defined but not used".
        list(APPEND CUDA_NVCC_FLAGS_DEBUG -Xcompiler;-Wno-unused-function;)
        if (NOT "${CMAKE_CXX_COMPILER_ID}" MATCHES ".*Clang.*")
            list(APPEND CUDA_NVCC_FLAGS_DEBUG -Xcompiler;-Wno-unused-local-typedef;)
        endif()
        # PTX compiler options
        #list(APPEND CUDA_NVCC_FLAGS_RELEASE --ptxas-options=-v;)
    endif ()

    message("===============================================================================")
    if (NOT DEFINED CUDA_ARCH OR CUDA_ARCH MATCHES ALL|[Aa]ll)
        message("-- INFO: Building CUDA device code for Fermi, Kepler,")
        message("-- INFO: Maxwell and Pascal architectures")
        message("-- INFO: The target CUDA architecture can be specified by using the option:")
        message("-- INFO:   -DCUDA_ARCH=<arch>")
        message("-- INFO: where <arch> is one of:")
        message("-- INFO:   1.3, 2.0, 2.1, 3.0, 3.5, 3.7, 5.0, 5.2, 6.0, 6.1 or ALL.")
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_21)
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_37,code=sm_37)
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_50)
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_52,code=sm_52)
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_60,code=sm_60)
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_61,code=sm_61)
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
    elseif (CUDA_ARCH MATCHES 3.7)
        message("-- INFO: Building CUDA device code for architecture 3.7")
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_37,code=sm_37)
    elseif (CUDA_ARCH MATCHES 5.0)
        message("-- INFO: Building CUDA device code for architecture 5.0")
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_50)
    elseif (CUDA_ARCH MATCHES 5.2)
        message("-- INFO: Building CUDA device code for architecture 5.2")
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_52,code=sm_52)
    elseif (CUDA_ARCH MATCHES 6.0)
        message("-- INFO: Building CUDA device code for architecture 6.0")
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_60,code=sm_60)
    elseif (CUDA_ARCH MATCHES 6.1)
        message("-- INFO: Building CUDA device code for architecture 6.1")
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_61,code=sm_61)
    elseif (CUDA_ARCH MATCHES 6.2)
        message("-- INFO: Building CUDA device code for architecture 6.2")
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_62,code=sm_62)
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

message("===============================================================================")
message("-- INFO: OSKAR version  ${OSKAR_VERSION_STR} [${OSKAR_VERSION_ID}]")
if (CMAKE_VERSION VERSION_GREATER 2.8.11)
    message("-- INFO: Build date     ${OSKAR_BUILD_DATE}")
endif()
message("-- INFO: Build type     ${CMAKE_BUILD_TYPE}")
message("-- INFO: Compiler ID    ${CMAKE_C_COMPILER_ID}:${CMAKE_CXX_COMPILER_ID}")
message("===============================================================================")

if (BUILD_INFO)
    message("===============================================================================")
    message(STATUS "C++ compiler  : ${CMAKE_CXX_COMPILER}")
    message(STATUS "C compiler    : ${CMAKE_C_COMPILER}")
    if (DEFINED NVCC_COMPILER_BINDIR)
        message(STATUS "nvcc bindir   : ${NVCC_COMPILER_BINDIR}")
    endif()
    if (FORCE_LIBSTDC++)
        message(STATUS "Forcing linking with libstdc++")
    endif()
    if (${CMAKE_BUILD_TYPE} MATCHES [Rr]elease)
        message(STATUS "C++ flags     : ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
        message(STATUS "C flags       : ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELEASE}")
        message(STATUS "CUDA flags    : ${CUDA_NVCC_FLAGS} ${CUDA_NVCC_FLAGS_RELEASE}")
        message(STATUS "OpenMP flags  : ${OpenMP_CXX_FLAGS}")
    elseif (${CMAKE_BUILD_TYPE} MATCHES [Dd]ebug)
        message(STATUS "C++ flags     : ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
        message(STATUS "C flags       : ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG}")
        message(STATUS "CUDA flags    : ${CUDA_NVCC_FLAGS} ${CUDA_NVCC_FLAGS_DEBUG}")
        message(STATUS "OpenMP flags  : ${OpenMP_CXX_FLAGS}")
    elseif (${CMAKE_BUILD_TYPE} MATCHES [Rr]elWithDebInfo)
        message(STATUS "C++ flags     : ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
        message(STATUS "C flags       : ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELWITHDEBINFO}")
        message(STATUS "CUDA flags    : ${CUDA_NVCC_FLAGS} ${CUDA_NVCC_FLAGS_RELWITHDEBINFO}")
        message(STATUS "OpenMP flags  : ${OpenMP_CXX_FLAGS}")
    elseif (${CMAKE_BUILD_TYPE} MATCHES [Mm]inSizeRel)
        message(STATUS "C++ flags     : ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_MINSIZEREL}")
        message(STATUS "C flags       : ${CMAKE_C_FLAGS}$ {CMAKE_C_FLAGS_MINSIZEREL}")
        message(STATUS "CUDA flags    : ${CUDA_NVCC_FLAGS} ${CUDA_NVCC_FLAGS_MINSIZEREL}")
        message(STATUS "OpenMP flags  : ${OpenMP_CXX_FLAGS}")
    endif()
    message("===============================================================================")
endif (BUILD_INFO)

