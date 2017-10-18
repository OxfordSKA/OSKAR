macro(APPEND_FLAGS FLAG_VAR)
    foreach (flag ${ARGN})
        set(${FLAG_VAR} "${${FLAG_VAR}} ${flag}")
    endforeach()
endmacro()

# Set general compiler flags.
set(BUILD_SHARED_LIBS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set OpenMP compile and link flags for all targets, if defined.
if (OpenMP_C_FLAGS)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
endif()
if (OpenMP_CXX_FLAGS)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    if (NOT MSVC)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
endif()

if (NOT WIN32)
    if ("${CMAKE_CXX_COMPILER_ID}" MATCHES ".*Clang.*")
        # Tell Clang to use C++11, required for casacore headers.
        append_flags(CMAKE_CXX_FLAGS -std=c++11)
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
        # Specify C standard manually here.
        # (Using CMAKE_C_STANDARD sets it to GNU version instead,
        # which breaks strtok_r.)
        append_flags(CMAKE_C_FLAGS -std=c99)

        # Treat external code as system headers.
        # This avoids a number of warning supression flags.
        append_flags(CMAKE_CXX_FLAGS
            -isystem ${GTEST_INCLUDE_DIR}
            -isystem ${GTEST_INCLUDE_DIR}/internal
        )
        if (CASACORE_FOUND)
            append_flags(CMAKE_CXX_FLAGS -isystem ${CASACORE_INCLUDE_DIR}/casacore)
        endif()
        if (CUDA_FOUND)
            append_flags(CMAKE_C_FLAGS -isystem ${CUDA_INCLUDE_DIRS})
            append_flags(CMAKE_CXX_FLAGS -isystem ${CUDA_INCLUDE_DIRS})
        endif()
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

# RPATH settings for macOS
# See https://cmake.org/Wiki/CMake_RPATH_handling
# ------------------------------------------------------------------------------
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
if (APPLE)
    set(CMAKE_INSTALL_NAME_DIR "@rpath")
    list(APPEND CMAKE_INSTALL_RPATH
        "@loader_path/../${OSKAR_LIB_INSTALL_DIR}/")
endif (APPLE)

# Set CUDA releated compiler flags.
# --compiler-options or -Xcompiler: specify options directly to the compiler
#                                   that nvcc encapsulates.
# ------------------------------------------------------------------------------
if (CUDA_FOUND)
    if (MSVC)
        set(CUDA_PROPAGATE_HOST_FLAGS ON)
    else()
        set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    endif()
    set(CUDA_VERBOSE_BUILD OFF)

    # General NVCC compiler options.
    set(CUDA_NVCC_FLAGS_RELEASE "-O3")
    set(CUDA_NVCC_FLAGS_DEBUG "-O0 -g --generate-line-info")
    set(CUDA_NVCC_FLAGS_RELWIDTHDEBINFO "-03 -g --generate-line-info")
    set(CUDA_NVCC_FLAGS_MINSIZEREL "-01")
    if (DEFINED NVCC_COMPILER_BINDIR)
        append_flags(CUDA_NVCC_FLAGS -ccbin ${NVCC_COMPILER_BINDIR})
    endif()

    # Options passed to the compiler NVCC encapsulates.
    if (APPLE AND (DEFINED CMAKE_OSX_DEPLOYMENT_TARGET))
        list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET};)
    endif()
    if (NOT WIN32)
        if (FORCE_LIBSTDC++)
            list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-stdlib=libstdc++;)
        endif()
        list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-fvisibility=hidden;)
        list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-Wall;)
        list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-Wextra;)
        list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-Wno-unused-private-field;)
        list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-Wno-unused-parameter;)
        list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-Wno-variadic-macros;)
        list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-Wno-long-long;)
        list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-Wno-unused-function;)
        if (NOT "${CMAKE_CXX_COMPILER_ID}" MATCHES ".*Clang.*")
            list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-Wno-unused-local-typedef;)
        endif()
        # PTX compiler options
        #list(APPEND CUDA_NVCC_FLAGS_RELEASE --ptxas-options=-v;)
    endif ()

    message("===============================================================================")
    if (NOT DEFINED CUDA_ARCH)
        set(CUDA_ARCH "ALL")
        message("-- INFO: Setting CUDA_ARCH to ALL.")
        message("-- INFO: The target CUDA architecture can be specified using:")
        message("-- INFO:   -DCUDA_ARCH=\"<arch>\"")
        message("-- INFO: where <arch> is one or more of:")
        message("-- INFO:   1.3, 2.0, 2.1, 3.0, 3.2, 3.5, 3.7,")
        message("-- INFO:   5.0, 5.2, 6.0, 6.1, 6.2, 7.0 or ALL.")
        message("-- INFO: Separate multiple architectures with semi-colons.")
    endif()
    foreach (ARCH ${CUDA_ARCH})
        if (ARCH MATCHES ALL|[Aa]ll)
            message("-- INFO: Building CUDA device code for Kepler,")
            message("-- INFO: Maxwell and Pascal architectures")
            list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
            list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_32,code=sm_32)
            list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
            list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_37,code=sm_37)
            list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_50)
            list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_52,code=sm_52)
            list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_60,code=sm_60)
            list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_61,code=sm_61)
            list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_62,code=sm_62)
        elseif (ARCH MATCHES 1.3)
            message("-- INFO: Building CUDA device code for architecture 1.3")
            list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_13,code=sm_13)
        elseif (ARCH MATCHES 2.0)
            message("-- INFO: Building CUDA device code for architecture 2.0")
            list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
        elseif (ARCH MATCHES 2.1)
            message("-- INFO: Building CUDA device code for architecture 2.1")
            list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_21)
        elseif (ARCH MATCHES 3.0)
            message("-- INFO: Building CUDA device code for architecture 3.0")
            list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
        elseif (ARCH MATCHES 3.2)
            message("-- INFO: Building CUDA device code for architecture 3.2")
            list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_32,code=sm_32)
        elseif (ARCH MATCHES 3.5)
            message("-- INFO: Building CUDA device code for architecture 3.5")
            list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
        elseif (ARCH MATCHES 3.7)
            message("-- INFO: Building CUDA device code for architecture 3.7")
            list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_37,code=sm_37)
        elseif (ARCH MATCHES 5.0)
            message("-- INFO: Building CUDA device code for architecture 5.0")
            list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_50)
        elseif (ARCH MATCHES 5.2)
            message("-- INFO: Building CUDA device code for architecture 5.2")
            list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_52,code=sm_52)
        elseif (ARCH MATCHES 6.0)
            message("-- INFO: Building CUDA device code for architecture 6.0")
            list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_60,code=sm_60)
        elseif (ARCH MATCHES 6.1)
            message("-- INFO: Building CUDA device code for architecture 6.1")
            list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_61,code=sm_61)
        elseif (ARCH MATCHES 6.2)
            message("-- INFO: Building CUDA device code for architecture 6.2")
            list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_62,code=sm_62)
        elseif (ARCH MATCHES 7.0)
            message("-- INFO: Building CUDA device code for architecture 7.0")
            list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_70,code=sm_70)
        else()
            message(FATAL_ERROR "-- CUDA_ARCH ${ARCH} not recognised!")
        endif()
    endforeach()
endif (CUDA_FOUND)

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
if (NOT CUDA_FOUND)
    message("===============================================================================")
    message("-- WARNING: CUDA toolkit not found: Unable to use any GPUs.")
    message("===============================================================================")
endif()
if (NOT CASACORE_FOUND)
    message("===============================================================================")
    message("-- WARNING: CASACORE not found: Unable to use Measurement Sets.")
    message("===============================================================================")
endif()
if (BUILD_INFO)
    message("===============================================================================")
    if (CASACORE_FOUND)
        message(STATUS "CASACORE      : ${CASACORE_LIBRARIES}")
    endif()
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
