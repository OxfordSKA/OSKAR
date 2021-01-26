# CMake initialisation.
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

# Set policies to avoid CMake warnings.
if (POLICY CMP0042)
    cmake_policy(SET CMP0042 NEW)
endif()
if (POLICY CMP0045)
    cmake_policy(SET CMP0045 NEW)
endif()
if (POLICY CMP0068)
    cmake_policy(SET CMP0068 NEW)
endif()

# Declare the CMake project.
project(oskar_ms CXX)

# Set the name of the library.
set(libname ${PROJECT_NAME})

# Add local CMake module path.
list(INSERT CMAKE_MODULE_PATH 0 ${PROJECT_SOURCE_DIR}/cmake)

# If OSKAR_VERSION_STR is not defined, we are building a stand-alone library.
if (NOT OSKAR_VERSION_STR)
    set(STANDALONE_BUILD ON)
endif()

# Try to find dependencies.
find_package(CASACORE)

if (CASACORE_FOUND)
    # Set CMake cache variable so any parent CMakeLists file can use it.
    # (PARENT_SCOPE is no good here, as there may not be a parent.)
    set(CASACORE_FOUND TRUE CACHE BOOL "CASACORE_FOUND")

    # C++11 is now required for casacore headers.
    include(CheckCXXCompilerFlag)
    CHECK_CXX_COMPILER_FLAG(-std=c++11 COMPILER_SUPPORTS_CXX11)
    CHECK_CXX_COMPILER_FLAG(-std=c++0x COMPILER_SUPPORTS_CXX0X)
    if (COMPILER_SUPPORTS_CXX11)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    elseif (COMPILER_SUPPORTS_CXX0X)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    endif()

    include_directories(${CASACORE_INCLUDE_DIR})
    include_directories(${CASACORE_INCLUDE_DIR}/casacore)

    set(ms_SRC
        src/oskar_ms_accessors.cpp
        src/oskar_ms_add_history.cpp
        src/oskar_ms_close.cpp
        src/oskar_ms_create.cpp
        src/oskar_ms_open.cpp
        src/oskar_ms_read.cpp
        src/oskar_ms_write.cpp
    )

    if (STANDALONE_BUILD)
        # Build type detection.
        if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
            get_filename_component(dirname ${CMAKE_BINARY_DIR} NAME)
            if (${dirname} MATCHES "dbg" OR ${dirname} MATCHES "debug")
                set(CMAKE_BUILD_TYPE Debug)
            else()
                set(CMAKE_BUILD_TYPE Release)
            endif()
        endif()

        # Set install locations for standalone mode.
        if (NOT OSKAR_LIB_INSTALL_DIR)
            set(OSKAR_LIB_INSTALL_DIR "lib")
        endif()
        if (NOT OSKAR_INCLUDE_INSTALL_DIR)
            set(OSKAR_INCLUDE_INSTALL_DIR "include/oskar/ms")
        endif()
        set(OSKAR_VERSION "2")

        # Compiler options for standalone mode.
        set(BUILD_SHARED_LIBS ON)
        set(CMAKE_POSITION_INDEPENDENT_CODE ON)
        set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
        if (APPLE)
            set(CMAKE_OSX_DEPLOYMENT_TARGET "10.9" CACHE STRING "Minimum macOS version")
            set(CMAKE_INSTALL_NAME_DIR "@rpath")
            list(APPEND CMAKE_INSTALL_RPATH
                "@loader_path/../${OSKAR_LIB_INSTALL_DIR}/")
        else()
            list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES
                    "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
            if ("${isSystemDir}" STREQUAL "-1")
                list(APPEND CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
            endif()
        endif()
        if ("${CMAKE_CXX_COMPILER_ID}" MATCHES ".*Clang.*")
            if (FORCE_LIBSTDC++)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
            else()
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
            endif()
        endif()
        if ("${CMAKE_C_COMPILER_ID}" MATCHES ".*Clang.*"
                OR "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
        endif()
        include_directories("${PROJECT_SOURCE_DIR}/../")

        # Install the headers.
        file(GLOB ${libname}_H oskar*.h) # Public headers only.
        install(FILES ${${libname}_H}
            DESTINATION ${OSKAR_INCLUDE_INSTALL_DIR} COMPONENT headers)
    endif(STANDALONE_BUILD)

    # Build and install the library.
    add_library(${libname} ${ms_SRC})
    set_target_properties(${libname} PROPERTIES
        COMPILE_FLAGS "-Wno-cast-qual -fvisibility=default")
    target_link_libraries(${libname} ${CASACORE_LIBRARIES})
    set_target_properties(${libname} PROPERTIES
        SOVERSION ${OSKAR_VERSION}
        VERSION ${OSKAR_VERSION})
    install(TARGETS ${libname}
        ARCHIVE DESTINATION ${OSKAR_LIB_INSTALL_DIR} COMPONENT libraries
        LIBRARY DESTINATION ${OSKAR_LIB_INSTALL_DIR} COMPONENT libraries
        RUNTIME DESTINATION ${OSKAR_BIN_INSTALL_DIR} COMPONENT runtime
    )

    if (NOT STANDALONE_BUILD)
        # Add unit tests.
        # (Tricky if in standalone mode because of current GTest dependency.)
        if (BUILD_TESTING OR NOT DEFINED BUILD_TESTING)
            enable_testing()
            add_subdirectory(test)
        endif()
    endif()

    # Verbose printing.
    if (BUILD_INFO)
        message(STATUS "CASACORE      : ${CASACORE_LIBRARIES}")
    endif()
else()
    message("===============================================================================")
    message("-- WARNING: CASACORE not found: Unable to use Measurement Sets.")
    message("===============================================================================")
endif()
