#! CMake initialisation.
cmake_minimum_required(VERSION 2.8.3)
if (POLICY CMP0045)
    cmake_policy(SET CMP0045 OLD)
endif()

#! Declare the CMake project.
project(oskar_binary C)

#! Set the OSKAR binary version used to version dynamic libraries.
set(OSKAR_BINARY_VERSION "2")

#! Set the name of the library.
set(libname ${PROJECT_NAME})

#! If OSKAR_VERSION_STR is not defined, we are building as stand-alone library.
if (NOT OSKAR_VERSION_STR)
    set(STANDALONE_BUILD ON)
endif()

#! Macros for building oskar_binary as a standalone library.
if (STANDALONE_BUILD)

    #! Build type detection.
    if(NOT CMAKE_BUILD_TYPE)
        get_filename_component(dirname ${CMAKE_BINARY_DIR} NAME)
        if (${dirname} MATCHES "dbg" OR ${dirname} MATCHES "debug")
            set(CMAKE_BUILD_TYPE debug)
        else()
            set(CMAKE_BUILD_TYPE release)
        endif()
    endif()

    #! Enable the ctest framework.
    enable_testing()

    #! Compiler options for standalone mode.
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
    include_directories(${PROJECT_SOURCE_DIR})

    #! Set install locations for standalone mode.
    if (NOT OSKAR_LIB_INSTALL_DIR)
        set(OSKAR_LIB_INSTALL_DIR "lib")
    endif()
    if (NOT OSKAR_INCLUDE_INSTALL_DIR)
        set(OSKAR_INCLUDE_INSTALL_DIR "include/oskar_binary")
    endif()

endif(STANDALONE_BUILD)

#! Declare build and install targets for the library.
file(GLOB ${libname}_SRC src/*.c)
add_library(${libname} STATIC ${${libname}_SRC})
set_target_properties(${libname} PROPERTIES
    INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${OSKAR_LIB_INSTALL_DIR}
    INSTALL_RPATH_USE_LINK_PATH TRUE
    SOVERSION ${OSKAR_BINARY_VERSION}
    VERSION ${OSKAR_BINARY_VERSION})
install(TARGETS ${libname} DESTINATION ${OSKAR_LIB_INSTALL_DIR})

#! Add unit tests / examples.
add_subdirectory(test)

#! Install headers and packaging macros.
if (STANDALONE_BUILD)
    file(GLOB ${libname}_H oskar*.h) # Public headers only (start with oskar_)
    install(FILES ${${libname}_H} DESTINATION ${OSKAR_INCLUDE_INSTALL_DIR})

    set(CPACK_PACKAGE_NAME "OSKAR_binary")
    set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/readme.txt")
    set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "OSKAR binary library (version ${OSKAR_BINARY_VERSION})")
    set(CPACK_PACKAGE_VENDOR "University of Oxford, Oxford e-Research Centre.")
    set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/copying.txt")
    set(CPACK_PACKAGE_VERSION_MAJOR "${OSKAR_BINARY_VERSION}")
    set(CPACK_PACKAGE_VERSION "${OSKAR_BINARY_VERSION}")
    set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY ON)
    set(CPACK_GENERATOR "TGZ")
    set(CPACK_SOURCE_GENERATOR "ZIP")
    set(CPACK_SOURCE_IGNORE_FILES "/build/;/debug/;/dbg/;/release/")
    include(CPack)
    add_custom_target(dist
        COMMAND ${CMAKE_MAKE_PROGRAM} package_source
        COMMENT "Packaging Source files"
        VERBATIM)
endif(STANDALONE_BUILD)

