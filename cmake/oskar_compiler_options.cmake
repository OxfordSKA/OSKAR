macro(APPEND_FLAGS FLAG_VAR)
    foreach (flag ${ARGN})
        set(${FLAG_VAR} "${${FLAG_VAR}} ${flag}")
    endforeach()
endmacro()

# Set general compiler flags.
set(BUILD_SHARED_LIBS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Using C11 for aligned_alloc.
set(CMAKE_C_STANDARD 11)

# Add -march=native if enabled and supported.
option(USE_NATIVE_CPU_ARCH "Build with -march=native" OFF)
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if (COMPILER_SUPPORTS_MARCH_NATIVE AND USE_NATIVE_CPU_ARCH)
    message("-- INFO: Adding -march=native flag")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

if (NOT WIN32)
    if ("${CMAKE_CXX_COMPILER_ID}" MATCHES ".*Clang.*")
        if (FORCE_LIBSTDC++)
            # Tell Clang to use libstdc++ rather than libc++
            # This is required if any of the OSKAR dependencies are built
            # against libstdc++ due to ABI incompatibility with libc++.
            append_flags(CMAKE_CXX_FLAGS -stdlib=libstdc++)
        else()
            append_flags(CMAKE_CXX_FLAGS -stdlib=libc++)
        endif()
    endif()

    if ("${CMAKE_C_COMPILER_ID}" MATCHES ".*Clang.*"
            OR "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
        # Note: long-long is required for cfitsio
        append_flags(CMAKE_CXX_FLAGS
            -Wall -Wextra -pedantic -Wcast-qual -Wcast-align -Wno-long-long
            -Wno-variadic-macros -Wno-unused-function
            -fvisibility=hidden -fvisibility-inlines-hidden
            -fdiagnostics-show-option)
        append_flags(CMAKE_C_FLAGS
            -Wall -Wextra -pedantic -Wcast-qual -Wcast-align -Wno-long-long
            -Wno-variadic-macros -Wno-unused-function -Wmissing-prototypes
            -fvisibility=hidden
            -fdiagnostics-show-option)

         # Additional test flags
#        append_flags(CMAKE_C_FLAGS
#            -Wbad-function-cast -Wstack-protector -Wpacked
#            -Wredundant-decls -Wshadow -Wwrite-strings
#            -Waggregate-return -Wstrict-prototypes
#            -Wstrict-aliasing -Wdeclaration-after-statement
#        )
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        # Using Intel compilers.
    endif()
else()
    if (MSVC)
        # Disable warning about loss of precision converting double to float.
        # Disable nonsensical warning about fopen.
        append_flags(CMAKE_C_FLAGS /wd4244 /wd4996)
        append_flags(CMAKE_CXX_FLAGS /wd4244 /wd4996)

        # Compile all files as C++.
        append_flags(CMAKE_C_FLAGS /TP)
        append_flags(CMAKE_CXX_FLAGS /TP)

        # Exception handling.
        append_flags(CMAKE_C_FLAGS /EHsc)
        append_flags(CMAKE_CXX_FLAGS /EHsc)

        # Default to dynamically-linked runtime.
        if ("${MSVC_RUNTIME}" STREQUAL "")
            set (MSVC_RUNTIME "dynamic")
        endif ()
        # Set compiler options.
        set(vars
            CMAKE_C_FLAGS
            CMAKE_C_FLAGS_DEBUG
            CMAKE_C_FLAGS_MINSIZEREL
            CMAKE_C_FLAGS_RELEASE
            CMAKE_C_FLAGS_RELWITHDEBINFO
            CMAKE_CXX_FLAGS
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
    endif()
endif ()

# RPATH settings.
# See https://gitlab.kitware.com/cmake/community/wikis/doc/cmake/RPATH-handling
# ------------------------------------------------------------------------------
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
if (APPLE)
    set(CMAKE_INSTALL_NAME_DIR "@rpath")
    list(APPEND CMAKE_INSTALL_RPATH
        "@loader_path/../${OSKAR_LIB_INSTALL_DIR}/")
elseif (NOT WIN32)
    list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES
            "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
    if ("${isSystemDir}" STREQUAL "-1")
       list(APPEND CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
    endif()
endif()

# Set CUDA releated compiler flags.
# --compiler-options or -Xcompiler: specify options directly to the compiler
#                                   that nvcc encapsulates.
# ------------------------------------------------------------------------------
if (CUDAToolkit_FOUND)
    if (MSVC)
        set(CUDA_PROPAGATE_HOST_FLAGS ON)
    else()
        set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    endif()
    set(CUDA_VERBOSE_BUILD OFF)

    # General NVCC compiler options.
    list(APPEND CMAKE_CUDA_FLAGS "--default-stream per-thread")
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
    set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -g --generate-line-info")
    set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O3 -g --generate-line-info")
    set(CMAKE_CUDA_FLAGS_MINSIZEREL "-O1")
    if (DEFINED NVCC_COMPILER_BINDIR)
        append_flags(CMAKE_CUDA_FLAGS -ccbin ${NVCC_COMPILER_BINDIR})
    endif()

    # Options passed to the compiler NVCC encapsulates.
    if (APPLE AND (DEFINED CMAKE_OSX_DEPLOYMENT_TARGET))
        list(APPEND CMAKE_CUDA_FLAGS -Xcompiler;-mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET};)
    endif()
    if (NOT WIN32)
        if (FORCE_LIBSTDC++)
            list(APPEND CMAKE_CUDA_FLAGS -Xcompiler;-stdlib=libstdc++;)
        endif()
        list(APPEND CMAKE_CUDA_FLAGS -Xcompiler;-fvisibility=hidden;)
        list(APPEND CMAKE_CUDA_FLAGS -Xcompiler;-Wall;)
        list(APPEND CMAKE_CUDA_FLAGS -Xcompiler;-Wextra;)
        list(APPEND CMAKE_CUDA_FLAGS -Xcompiler;-Wno-unused-private-field;)
        list(APPEND CMAKE_CUDA_FLAGS -Xcompiler;-Wno-unused-parameter;)
        list(APPEND CMAKE_CUDA_FLAGS -Xcompiler;-Wno-variadic-macros;)
        list(APPEND CMAKE_CUDA_FLAGS -Xcompiler;-Wno-long-long;)
        list(APPEND CMAKE_CUDA_FLAGS -Xcompiler;-Wno-unused-function;)
        if (NOT "${CMAKE_CXX_COMPILER_ID}" MATCHES ".*Clang.*")
            list(APPEND CMAKE_CUDA_FLAGS -Xcompiler;-Wno-unused-local-typedef;)
        endif()
        # PTX compiler options
        #list(APPEND CMAKE_CUDA_FLAGS_RELEASE --ptxas-options=-v;)
    endif ()

    # Replace semi-colons in CMAKE_CUDA_FLAGS list with spaces for nvcc.
    string(REPLACE ";" " " CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

    message("===============================================================================")
    if (NOT DEFINED CUDA_ARCH)
        set(CUDA_ARCH "ALL")
        message("-- INFO: Setting CUDA_ARCH to ALL.")
        message("-- INFO: The target CUDA architecture can be specified using:")
        message("-- INFO:   -DCUDA_ARCH=\"<arch>\"")
        message("-- INFO: where <arch> is one or more of:")
        message("-- INFO:   3.0, 3.2, 3.5, 3.7, 5.0, 5.2, 6.0, 6.1, 6.2,")
        message("-- INFO:   7.0, 7.5, 8.0, 8.6, 8.7, 8.9, 9.0, 10.0, 12.0,")
        message("-- INFO:   or ALL.")
        message("-- INFO: Note that ALL is currently most from 7.5 to 8.9.")
        message("-- INFO: Separate multiple architectures with semi-colons.")
    endif()
    foreach (ARCH ${CUDA_ARCH})
        if (ARCH MATCHES ALL|[Aa]ll)
            message("-- INFO: Building CUDA device code for architecture 7.5 (RTX 20xx-series; GTX 16xx-series; Titan RTX)")
            message("-- INFO: Building CUDA device code for architecture 8.0 (A100)")
            message("-- INFO: Building CUDA device code for architecture 8.6 (RTX 30xx-series)")
            message("-- INFO: Building CUDA device code for architecture 8.9 (RTX 40xx-series)")
            list(APPEND CMAKE_CUDA_ARCHITECTURES 75-real)
            list(APPEND CMAKE_CUDA_ARCHITECTURES 80-real)
            list(APPEND CMAKE_CUDA_ARCHITECTURES 86-real)
            list(APPEND CMAKE_CUDA_ARCHITECTURES 89-real)
        elseif (ARCH EQUAL 3.0)
            message("-- INFO: Building CUDA device code for architecture 3.0 (Early Kepler)")
            list(APPEND CMAKE_CUDA_ARCHITECTURES 30-real)
        elseif (ARCH EQUAL 3.2)
            message("-- INFO: Building CUDA device code for architecture 3.2")
            list(APPEND CMAKE_CUDA_ARCHITECTURES 32-real)
        elseif (ARCH EQUAL 3.5)
            message("-- INFO: Building CUDA device code for architecture 3.5 (K20; K40)")
            list(APPEND CMAKE_CUDA_ARCHITECTURES 35-real)
        elseif (ARCH EQUAL 3.7)
            message("-- INFO: Building CUDA device code for architecture 3.7 (K80)")
            list(APPEND CMAKE_CUDA_ARCHITECTURES 37-real)
        elseif (ARCH EQUAL 5.0)
            message("-- INFO: Building CUDA device code for architecture 5.0 (Early Maxwell)")
            list(APPEND CMAKE_CUDA_ARCHITECTURES 50-real)
        elseif (ARCH EQUAL 5.2)
            message("-- INFO: Building CUDA device code for architecture 5.2 (GTX 9xx-series)")
            list(APPEND CMAKE_CUDA_ARCHITECTURES 52-real)
        elseif (ARCH EQUAL 6.0)
            message("-- INFO: Building CUDA device code for architecture 6.0 (P100)")
            list(APPEND CMAKE_CUDA_ARCHITECTURES 60-real)
        elseif (ARCH EQUAL 6.1)
            message("-- INFO: Building CUDA device code for architecture 6.1 (GTX 10xx-series)")
            list(APPEND CMAKE_CUDA_ARCHITECTURES 61-real)
        elseif (ARCH EQUAL 6.2)
            message("-- INFO: Building CUDA device code for architecture 6.2")
            list(APPEND CMAKE_CUDA_ARCHITECTURES 62-real)
        elseif (ARCH EQUAL 7.0)
            message("-- INFO: Building CUDA device code for architecture 7.0 (V100; Titan V)")
            list(APPEND CMAKE_CUDA_ARCHITECTURES 70-real)
        elseif (ARCH EQUAL 7.5)
            message("-- INFO: Building CUDA device code for architecture 7.5 (RTX 20xx-series; GTX 16xx-series; Titan RTX)")
            list(APPEND CMAKE_CUDA_ARCHITECTURES 75-real)
        elseif (ARCH EQUAL 8.0)
            message("-- INFO: Building CUDA device code for architecture 8.0 (A100)")
            list(APPEND CMAKE_CUDA_ARCHITECTURES 80-real)
        elseif (ARCH EQUAL 8.6)
            message("-- INFO: Building CUDA device code for architecture 8.6 (RTX 30xx-series)")
            list(APPEND CMAKE_CUDA_ARCHITECTURES 86-real)
        elseif (ARCH EQUAL 8.7)
            message("-- INFO: Building CUDA device code for architecture 8.7")
            list(APPEND CMAKE_CUDA_ARCHITECTURES 87-real)
        elseif (ARCH EQUAL 8.9)
            message("-- INFO: Building CUDA device code for architecture 8.9 (RTX 40xx-series)")
            list(APPEND CMAKE_CUDA_ARCHITECTURES 89-real)
        elseif (ARCH EQUAL 9.0)
            message("-- INFO: Building CUDA device code for architecture 9.0 (H100; H200; GH200)")
            list(APPEND CMAKE_CUDA_ARCHITECTURES 90-real)
        elseif (ARCH EQUAL 10.0)
            message("-- INFO: Building CUDA device code for architecture 10.0 (B200; GB200)")
            list(APPEND CMAKE_CUDA_ARCHITECTURES 100-real)
        elseif (ARCH EQUAL 12.0)
            message("-- INFO: Building CUDA device code for architecture 12.0 (RTX 50xx-series)")
            list(APPEND CMAKE_CUDA_ARCHITECTURES 120-real)
        else()
            message(FATAL_ERROR "-- CUDA_ARCH ${ARCH} not recognised!")
        endif()
    endforeach()
endif (CUDAToolkit_FOUND)

# Printing only below this line.
message("===============================================================================")
message("-- INFO: OSKAR version  ${OSKAR_VERSION_STR} [${OSKAR_VERSION_ID}]")
message("-- INFO: Build type     ${CMAKE_BUILD_TYPE}")
message("-- INFO: Compiler ID    ${CMAKE_C_COMPILER_ID}:${CMAKE_CXX_COMPILER_ID}")
message("===============================================================================")
message("-- INFO: 'make install' will install OSKAR to:")
message("-- INFO:   - Libraries         ${CMAKE_INSTALL_PREFIX}/${OSKAR_LIB_INSTALL_DIR}")
message("-- INFO:   - Headers           ${CMAKE_INSTALL_PREFIX}/${OSKAR_INCLUDE_INSTALL_DIR}")
message("-- INFO:   - Applications      ${CMAKE_INSTALL_PREFIX}/${OSKAR_BIN_INSTALL_DIR}")
message("-- NOTE: These paths can be changed using: '-DCMAKE_INSTALL_PREFIX=<path>'")
message("===============================================================================")
if (NOT CUDAToolkit_FOUND)
    message("===============================================================================")
    message("-- WARNING: CUDA toolkit not found.")
    message("===============================================================================")
endif()
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
        message(STATUS "CUDA flags    : ${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_RELEASE}")
    elseif (${CMAKE_BUILD_TYPE} MATCHES [Dd]ebug)
        message(STATUS "C++ flags     : ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
        message(STATUS "C flags       : ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG}")
        message(STATUS "CUDA flags    : ${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_DEBUG}")
    elseif (${CMAKE_BUILD_TYPE} MATCHES [Rr]elWithDebInfo)
        message(STATUS "C++ flags     : ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
        message(STATUS "C flags       : ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELWITHDEBINFO}")
        message(STATUS "CUDA flags    : ${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_RELWITHDEBINFO}")
    elseif (${CMAKE_BUILD_TYPE} MATCHES [Mm]inSizeRel)
        message(STATUS "C++ flags     : ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_MINSIZEREL}")
        message(STATUS "C flags       : ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_MINSIZEREL}")
        message(STATUS "CUDA flags    : ${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_MINSIZEREL}")
    endif()
    message("===============================================================================")
endif (BUILD_INFO)
