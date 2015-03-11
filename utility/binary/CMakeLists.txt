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

#project(oskar_binary C)
#cmake_minimum_required(VERSION 2.8.3 FATAL_ERROR)
#
#if (POLICY CMP0042)
#    cmake_policy(SET CMP0042 OLD)
#endif()
#if (POLICY CMP0045)
#    cmake_policy(SET CMP0045 OLD)
#endif()
#
#list(INSERT CMAKE_MODULE_PATH 0 ${PROJECT_SOURCE_DIR}/../../cmake)
#include(oskar_compiler_options)
#
#if (NOT OSKAR_LIB_INSTALL_DIR)
#    set(OSKAR_LIB_INSTALL_DIR "lib")
#endif()
#if (NOT OSKAR_INC_INSTALL_DIR)
#    set(OSKAR_INC_INSTALL_DIR "include/oskar_binary")
#endif()

#include_directories(${PROJECT_SOURCE_DIR})

set(name oskar_binary)
file(GLOB ${name}_SRC src/*.c)
file(GLOB ${name}_H oskar*.h)
add_library(${name} STATIC ${${name}_SRC})
set_target_properties(${name} PROPERTIES
    INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${OSKAR_LIB_INSTALL_DIR}
    INSTALL_RPATH_USE_LINK_PATH TRUE
    SOVERSION ${OSKAR_VERSION}
    VERSION ${OSKAR_VERSION})
#install(TARGETS ${name} DESTINATION ${OSKAR_LIB_INSTALL_DIR})
#install(FILES ${${name}_H} DESTINATION ${OSKAR_INC_INSTALL_DIR})

add_subdirectory(test)

#
# Packing
#

#set(CPACK_SOURCE_GENERATOR "ZIP")
#set(CPACK_INSTALL_CMAKE_PROJECTS ${CMAKE_CURRENT_SOURCE_DIR};oskar_binary;oskar_binary;/)
#include(CPack)
