#
# src/utility/binary/CMakeLists.txt
#

#
# TODO in order to make this into a full subproject which can have its
# own build directory assoicated some work will need to be done:
#   - install paths
#   - compiler options ?!
#   - packaging
#
# Some of this can be handled by moving cmake varialbes like
# compliler options & install locations & oskar version to a common module
# that everyone includes (e.g. oskar_install_dirs.cmake)
# oskar compiler flags would just need some modification...
#

cmake_minimum_required(VERSION 2.8.3 FATAL_ERROR)
# Set various policies to get old behviour with cmake 3.x
if (POLICY CMP0042)
    cmake_policy(SET CMP0042 OLD)
endif()
if (POLICY CMP0045)
    cmake_policy(SET CMP0045 OLD)
endif()
project(oskar_binary C)

if (NOT BUILD_STANDALONE)
    set(BUILD_STANDALONE OFF)
endif()

if (BUILD_STANDALONE)

    list(INSERT CMAKE_MODULE_PATH 0 ${PROJECT_SOURCE_DIR}/../../cmake)
    include(oskar_version)

#    include(oskar_compiler_options) ???
    set(CMAKE_C_FLAGS "-fPIC")
    set(CMAKE_C_FLAGS_RELEASE "-O2 -DNDEBUG")
    set(CMAKE_C_FLAGS_DEBUG "-O0 -g -Wall")
    set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -g -Wall")
    set(CMAKE_C_FLAGS_MINSIZEREL "-O1 -DNDEBUG")
    if ("${CMAKE_C_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=hidden")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fdiagnostics-show-option")
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wextra")
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -pedantic")
    endif()

    if (NOT OSKAR_LIB_INSTALL_DIR)
        set(OSKAR_LIB_INSTALL_DIR "lib")
    endif()
    if (NOT OSKAR_INCLUDE_INSTALL_DIR)
        set(OSKAR_INCLUDE_INSTALL_DIR "include/oskar_binary")
    endif()

    configure_file(${PROJECT_SOURCE_DIR}/../../cmake/oskar_version.h.in
        ${PROJECT_BINARY_DIR}/oskar_version.h @ONLY)

    include_directories(${PROJECT_SOURCE_DIR})
    include_directories(${PROJECT_BINARY_DIR})

    message(STATUS "=================================================")
    message(STATUS "Building in standalone mode....")
    message(STATUS "OSKAR VERSION = ${OSKAR_VERSION_STR}")
    message(STATUS "=================================================")
endif()

set(name oskar_binary)
file(GLOB ${name}_SRC src/*.c)

add_library(${name} STATIC ${${name}_SRC})
set_target_properties(${name} PROPERTIES
    INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${OSKAR_LIB_INSTALL_DIR}
    INSTALL_RPATH_USE_LINK_PATH TRUE
    SOVERSION ${OSKAR_VERSION}
    VERSION ${OSKAR_VERSION})
install(TARGETS ${name} DESTINATION ${OSKAR_LIB_INSTALL_DIR})

#add_subdirectory(test EXCLUDE_FROM_ALL)
add_subdirectory(test)
#add_subdirectory(examples EXCLUDE_FROM_ALL)

if (BUILD_STANDALONE)
    file(GLOB ${name}_H oskar*.h) # Only public headers!
    install(FILES ${${name}_H} DESTINATION ${OSKAR_INCLUDE_INSTALL_DIR})
    #set(CPACK_SOURCE_GENERATOR "ZIP")
    #set(CPACK_INSTALL_CMAKE_PROJECTS ${CMAKE_CURRENT_SOURCE_DIR};oskar_binary;oskar_binary;/)
    #include(CPack)
endif()
