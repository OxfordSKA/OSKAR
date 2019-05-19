#
# cmake/oskar_build_macros.cmake
#

#
# CMake macros used by the OSKAR build system.
#

include(CMakeParseArguments)

# Macro to build and install OSKAR apps.
#
# Usage:
#   oskar_app(NAME name
#         SOURCES source1 source2 ...
#         [EXTRA_LIBS lib1 lib2 ...]
#         [NO_INSTALL]
#   )
#
# NAME name      = Name of the app (binary name)
# SOURCES ...    = List of sources from which to build the app
# EXTRA_LIBS ... = List of additional libraries to link the app against.
# NO_INSTALL     = Do not install this app (i.e. don't add to the make install
#                  target).
#
# Note: Named options can appear in any order
#
macro(OSKAR_APP)
    cmake_parse_arguments(APP     # prefix
        "NO_INSTALL"              # boolean options
        "NAME"                    # single-value args
        "SOURCES;EXTRA_LIBS"      # multi-value args
        ${ARGN}
    )

    # Create a target name from the app name.
    # Note:
    # - Appending app allows for binaries to exist with the same name
    # as targets used elsewhere.
    # - app targets can then be build individually using the command:
    #      $ make <name>_app
    #   where <name> is the (binary) name of the app to be built.
    set(target "${APP_NAME}_app")

    add_executable(${target} ${APP_SOURCES})
    target_link_libraries(${target}
        oskar_apps          # default libs
        oskar_settings
        ${APP_EXTRA_LIBS}   # extra libs
    )
    set_target_properties(${target} PROPERTIES OUTPUT_NAME "${APP_NAME}")

    # Create the install target if the NO_INSTALL option isn't specified.
    if (NOT APP_NO_INSTALL)
        install(TARGETS ${target}
            DESTINATION ${OSKAR_BIN_INSTALL_DIR} COMPONENT applications)
    endif()
endmacro(OSKAR_APP)

# Wraps an OpenCL kernel source file to stringify it.
macro(OSKAR_WRAP_CL SRC_LIST)
    set(OPENCL_STRINGIFY ${PROJECT_SOURCE_DIR}/cmake/oskar_cl_stringify.cmake)
    foreach (CL_FILE ${ARGN})
        get_filename_component(name_ ${CL_FILE} NAME_WE)
        set(CL_FILE ${CMAKE_CURRENT_SOURCE_DIR}/${CL_FILE})
        set(CXX_FILE ${CMAKE_CURRENT_BINARY_DIR}/${name_}_cl_fragment.cpp)
        add_custom_command(OUTPUT ${CXX_FILE}
            COMMAND ${CMAKE_COMMAND}
                -D CL_FILE:FILEPATH=${CL_FILE}
                -D CXX_FILE:FILEPATH=${CXX_FILE}
                -P ${OPENCL_STRINGIFY}
            MAIN_DEPENDENCY ${CL_FILE}
            DEPENDS ${CL_FILE} ${OPENCL_STRINGIFY}
            VERBATIM
        )
        list(APPEND ${SRC_LIST} ${CXX_FILE})
    endforeach()
endmacro(OSKAR_WRAP_CL)
