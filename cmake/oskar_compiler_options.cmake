
#! TODO remove the need to check this ....
if (NOT CHECKED_DEPENDENCIES)
    message(FATAL_ERROR "Please include oskar_dependencies.cmake before this script!")
endif ()

# Automatically set the build type to release or debug if not specified.
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

# Set the include path to include the top-level folder and sub-folders for
# main oskar library.
# ------------------------------------------------------------------------------
include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_BINARY_DIR}
    ${PROJECT_SOURCE_DIR}/apps
    ${PROJECT_SOURCE_DIR}/apps/lib
    ${PROJECT_SOURCE_DIR}/apps/log
    ${PROJECT_SOURCE_DIR}/convert
    ${PROJECT_SOURCE_DIR}/correlate
    ${PROJECT_SOURCE_DIR}/element
    ${PROJECT_SOURCE_DIR}/extern
    ${PROJECT_SOURCE_DIR}/extern/gtest-1.7.0/include
    ${PROJECT_SOURCE_DIR}/extern/rapidxml-1.13
    ${PROJECT_SOURCE_DIR}/extern/cfitsio-3.37
    ${PROJECT_SOURCE_DIR}/extern/Random123
    ${PROJECT_SOURCE_DIR}/extern/Random123/features
    ${PROJECT_SOURCE_DIR}/imaging
    ${PROJECT_SOURCE_DIR}/interferometry
    ${PROJECT_SOURCE_DIR}/jones
    ${PROJECT_SOURCE_DIR}/math
    ${PROJECT_SOURCE_DIR}/ms
    ${PROJECT_SOURCE_DIR}/settings
    ${PROJECT_SOURCE_DIR}/settings/list
    ${PROJECT_SOURCE_DIR}/settings/load
    ${PROJECT_SOURCE_DIR}/settings/struct
    ${PROJECT_SOURCE_DIR}/settings/types
    ${PROJECT_SOURCE_DIR}/settings/utility
    ${PROJECT_SOURCE_DIR}/settings/widgets
    ${PROJECT_SOURCE_DIR}/sky
    ${PROJECT_SOURCE_DIR}/splines
    ${PROJECT_SOURCE_DIR}/station
    ${PROJECT_SOURCE_DIR}/utility
    ${PROJECT_SOURCE_DIR}/utility/binary
)
set(GTEST_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/extern/gtest-1.7.0/include/gtest)
set(EZOPT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/extern/ezOptionParser-0.2.0)


# Build the various version strings to be passed to the code.
set(OSKAR_VERSION "${OSKAR_VERSION_MAJOR}.${OSKAR_VERSION_MINOR}.${OSKAR_VERSION_PATCH}")
set(OSKAR_VERSION_STR "${OSKAR_VERSION}")
if (OSKAR_VERSION_SUFFIX AND NOT OSKAR_VERSION_SUFFIX STREQUAL "")

    # Find the subversion revision.
    find_package(Subversion QUIET)
    if (SUBVERSION_FOUND)
        get_filename_component(SVN_PATH ${PROJECT_SOURCE_DIR} REALPATH)
        # Check that svn info returns a valid result.
        execute_process(COMMAND ${Subversion_SVN_EXECUTABLE} info ${SVN_PATH}
            OUTPUT_VARIABLE ${prefix}_WC_INFO
            ERROR_VARIABLE Subversion_svn_info_error
            RESULT_VARIABLE Subversion_svn_info_result
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        if (${Subversion_svn_info_result} EQUAL 0)
            Subversion_WC_INFO(${SVN_PATH} OSKAR_SVN)
            if (OSKAR_SVN_WC_REVISION)
                set(OSKAR_SVN_REVISION ${OSKAR_SVN_WC_REVISION})
            endif()
        endif()
    endif()

    set(OSKAR_VERSION_STR "${OSKAR_VERSION}-${OSKAR_VERSION_SUFFIX}")
    if (OSKAR_SVN_REVISION)
        set(OSKAR_VERSION_STR "${OSKAR_VERSION_STR} r${OSKAR_SVN_REVISION}")
    endif()
    if (CMAKE_BUILD_TYPE MATCHES Debug)
        set(OSKAR_VERSION_STR "${OSKAR_VERSION_STR} -- debug --")
    endif()
endif()
if (CMAKE_VERSION VERSION_GREATER 2.8.11)
    string(TIMESTAMP OSKAR_BUILD_DATE "%Y-%m-%d %H:%M:%S")
endif()

configure_file(${PROJECT_SOURCE_DIR}/cmake/oskar_version.h.in
    ${PROJECT_BINARY_DIR}/oskar_version.h @ONLY)

# Set general compiler flags.
if (NOT WIN32)
    # Common compiler options. Note C code is compiled as gnu89 in order to
    # allow for a number of non C89 compiler extensions such as sinf, powf,
    # strtok_r as well as gnu inline mode which is needed for CUDA Thurst with
    # some compilers.
    set(CMAKE_C_FLAGS "-fPIC -std=gnu89")
    set(CMAKE_C_FLAGS_RELEASE "-O2 -DNDEBUG")
    set(CMAKE_C_FLAGS_DEBUG "-O0 -g -Wall")
    set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -g -Wall")
    set(CMAKE_C_FLAGS_MINSIZEREL "-O1 -DNDEBUG -DQT_NO_DEBUG -DQT_NO_DEBUG_OUTPUT")
    set(CMAKE_CXX_FLAGS "-fPIC")
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG -DQT_NO_DEBUG -DQT_NO_DEBUG_OUTPUT")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -Wall")
    set(CMAKE_CXX_FLAGS_MINSIZEREL "-O1 -DNDEBUG -DQT_NO_DEBUG -DQT_NO_DEBUG_OUTPUT")

    if ("${CMAKE_C_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
        # Using Clang or GNU compilers.

        # Treat external code as system headers.
        # This avoids a number of warning supression flags.
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -isystem ${CUDA_INCLUDE_DIRS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem ${CUDA_INCLUDE_DIRS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem ${GTEST_INCLUDE_DIR}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem ${GTEST_INCLUDE_DIR}/internal")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem ${CASACORE_INCLUDE_DIR}/casacore")

        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=hidden")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fdiagnostics-show-option")
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wextra")
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -pedantic")
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wcast-qual")
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wcast-align")
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wmissing-prototypes")
        ############ TEST FLAGS ##########
        #set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wbad-function-cast")
#        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wstack-protector")
#        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wpacked")
#        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wredundant-decls")
#        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wshadow")
#        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wwrite-strings")
#        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Waggregate-return")
#        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wstrict-prototypes")
#        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wstrict-aliasing")
#        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wdeclaration-after-statement")
        #############
        # long-long is required for C as cfitsio headers pull this into OSKAR
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wno-long-long")
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wno-variadic-macros")
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wno-unused-function")
        set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -Wno-unused-function")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility-inlines-hidden")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-show-option")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wextra")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -pedantic")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wcast-qual")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wcast-align")
        # long-long is required for C++ as ezOptionParser, gTest, and Qt headers
        # all pull this into OSKAR.
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wno-long-long")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wno-variadic-macros")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wno-unused-function")
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -Wno-unused-function")
        if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
            # Tell Clang to use libstdc++ rather than libc++
            # This is required if any of the OSKAR dependencies are built
            # against libstdc++. libc++ seems not to be ABI compatible with
            # libstdc++ so this is currently required.
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
        endif()

        # Check GNU compiler version
        if (CMAKE_COMPILER_IS_GNUCC)
            execute_process(COMMAND ${CMAKE_C_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
#            if (GCC_VERSION VERSION_GREATER 4.9 OR GCC_VERSION VERSION_EQUAL 4.9)
#                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++98")
#            endif()
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
        if (APPLE)
            list(APPEND CUDA_NVCC_FLAGS -Xcompiler;-stdlib=libstdc++;)
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
        # PTX compiler options
        #list(APPEND CUDA_NVCC_FLAGS_RELEASE --ptxas-options=-v;)
    endif ()

    message("===============================================================================")
    if (NOT DEFINED CUDA_ARCH OR CUDA_ARCH MATCHES ALL|[Aa]ll)
        message("-- INFO: Building CUDA device code for all Fermi and Kepler architectures")
        message("-- INFO: The target CUDA architecture can be specified by using the option:")
        message("-- INFO:   -DCUDA_ARCH=<arch>")
        message("-- INFO: where <arch> is one of:")
        message("-- INFO:   1.3, 2.0, 2.1, 3.0, 3.5, or ALL.")
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_21)
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
        list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
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

#set(BUILD_INFO OFF) # Enable with -DBUILD_INFO=ON when running cmake
if (BUILD_INFO)
    message("===============================================================================")
    message(STATUS "C++ compiler  : ${CMAKE_CXX_COMPILER}")
    message(STATUS "C compiler    : ${CMAKE_C_COMPILER}")
    if (${CMAKE_BUILD_TYPE} MATCHES [Rr]elease)
        message(STATUS "C++ flags     : ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
        message(STATUS "C flags       : ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELEASE}")
        message(STATUS "CUDA flags    : ${CUDA_NVCC_FLAGS} ${CUDA_NVCC_FLAGS_RELEASE}")
    elseif (${CMAKE_BUILD_TYPE} MATCHES [Dd]ebug)
        message(STATUS "C++ flags     : ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
        message(STATUS "C flags       : ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG}")
        message(STATUS "CUDA flags    : ${CUDA_NVCC_FLAGS} ${CUDA_NVCC_FLAGS_DEBUG}")
    elseif (${CMAKE_BUILD_TYPE} MATCHES [Rr]elWithDebInfo)
        message(STATUS "C++ flags     : ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
        message(STATUS "C flags       : ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELWITHDEBINFO}")
        message(STATUS "CUDA flags    : ${CUDA_NVCC_FLAGS} ${CUDA_NVCC_FLAGS_RELWITHDEBINFO}")
    elseif (${CMAKE_BUILD_TYPE} MATCHES [Mm]inSizeRel)
        message(STATUS "C++ flags     : ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_MINSIZEREL}")
        message(STATUS "C flags       : ${CMAKE_C_FLAGS}$ {CMAKE_C_FLAGS_MINSIZEREL}")
        message(STATUS "CUDA flags    : ${CUDA_NVCC_FLAGS} ${CUDA_NVCC_FLAGS_MINSIZEREL}")
    endif()
    message("===============================================================================")
endif (BUILD_INFO)

