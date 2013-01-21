#
# cmake/oskar_build_macros.cmake
#
#===============================================================================
# A collection of cmake macros used by the oskar build system.
#


#
# http://www.cmake.org/Wiki/CMakeMacroParseArguments
#
MACRO(PARSE_ARGUMENTS prefix arg_names option_names)
  SET(DEFAULT_ARGS)
  FOREACH(arg_name ${arg_names})
    SET(${prefix}_${arg_name})
  ENDFOREACH(arg_name)
  FOREACH(option ${option_names})
    SET(${prefix}_${option} FALSE)
  ENDFOREACH(option)

  SET(current_arg_name DEFAULT_ARGS)
  SET(current_arg_list)
  FOREACH(arg ${ARGN})
    SET(larg_names ${arg_names})
    LIST(FIND larg_names "${arg}" is_arg_name)
    IF (is_arg_name GREATER -1)
      SET(${prefix}_${current_arg_name} ${current_arg_list})
      SET(current_arg_name ${arg})
      SET(current_arg_list)
    ELSE (is_arg_name GREATER -1)
      SET(loption_names ${option_names})
      LIST(FIND loption_names "${arg}" is_option)
      IF (is_option GREATER -1)
         SET(${prefix}_${arg} TRUE)
      ELSE (is_option GREATER -1)
         SET(current_arg_list ${current_arg_list} ${arg})
      ENDIF (is_option GREATER -1)
    ENDIF (is_arg_name GREATER -1)
  ENDFOREACH(arg)
  SET(${prefix}_${current_arg_name} ${current_arg_list})
ENDMACRO(PARSE_ARGUMENTS)


#
# http://www.cmake.org/Wiki/CMakeMacroListOperations#CAR_and_CDR
#
MACRO(CAR var)
  SET(${var} ${ARGV1})
ENDMACRO(CAR)

MACRO(CDR var junk)
  SET(${var} ${ARGN})
ENDMACRO(CDR)


# Macro to build and install apps that depend on Qt.
#
# Usage:
#   qt_app(name
#         [NO_INSTALL]
#         source1 source2 ...
#         [QT_MOC_SRC source1 source2 ...]
#         [EXTRA_LIBS lib1 lib2 ...]
#   )
#
# name       = Name of the app (binary name)
# source1..N = List of sources from which to build the app
# QT_MOC_SRC = List of QT moc headers which require a precompile step.
# EXTRA_LIBS = List of additional libraries to link the app against.
# NO_INSTALL = Do not install this app (i.e. don't add to the make install
#              target).
#
# Note: Named options can appear in any order
#
macro(QT_APP)
    parse_arguments(APP   # prefix
        "QT_MOC_SRC;EXTRA_LIBS" # arg names
        "NO_INSTALL"      # option names
        ${ARGN}
    )
    CAR(APP_NAME ${APP_DEFAULT_ARGS})
    CDR(APP_SOURCES ${APP_DEFAULT_ARGS})

    if (NOT QT4_FOUND)
        message(CRITICAL "Unable to build oskar app ${APP_NAME}, Qt4 not found!")
    endif ()

    # Create a target name from the app name.
    # Note:
    # - Appending app allows for binaries to exist with the same name
    # as targets used elsewhere.
    # - app targets can then be build individually using the command:
    #      $ make <name>_app
    #   where <name> is the (binary) name of the app to be built.
    set(target "${APP_NAME}_app")

    qt4_wrap_cpp(APP_SOURCES ${APP_QT_MOC_SRC})

    add_executable(${target} ${APP_SOURCES})
    target_link_libraries(${target}
        ${QT_QTCORE_LIBRARY} # default libs
        ${APP_EXTRA_LIBS}                     # extra libs
    )
    set_target_properties(${target} PROPERTIES
        OUTPUT_NAME   ${APP_NAME}
        INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${OSKAR_LIB_INSTALL_DIR}
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )

    # Create the install target if the NO_INSTALL option isn't specified.
    if (NOT APP_NO_INSTALL)
        install(TARGETS ${target} DESTINATION ${OSKAR_BIN_INSTALL_DIR})
    endif()

endmacro(QT_APP)


# Macro to build and install oskar apps.
#
# Usage:
#   oskar_app(name
#         [NO_INSTALL]
#         source1 source2 ...
#         [EXTRA_LIBS lib1 lib2 ...]
#   )
#
# name       = Name of the app (binary name)
# source1..N = List of sources from which to build the app
# EXTRA_LIBS = List of additional libraries to link the app against.
# NO_INSTALL = Do not install this app (i.e. don't add to the make install
#              target).
#
# Note: Named options can appear in any order
#
macro(OSKAR_APP)
    parse_arguments(APP   # prefix
        "EXTRA_LIBS"      # arg names
        "NO_INSTALL"      # option names
        ${ARGN}
    )
    CAR(APP_NAME ${APP_DEFAULT_ARGS})
    CDR(APP_SOURCES ${APP_DEFAULT_ARGS})

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
        oskar               # default libs
        ${APP_EXTRA_LIBS}   # extra libs
    )
    # We do need the OpenMP flags here, otherwise programs will crash.
    if (MSVC)
        set_target_properties(${target} PROPERTIES
            COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
            OUTPUT_NAME   "${APP_NAME}"
            INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${OSKAR_LIB_INSTALL_DIR}"
            INSTALL_RPATH_USE_LINK_PATH TRUE
        )
    else ()
        set_target_properties(${target} PROPERTIES
            COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
            LINK_FLAGS    "${OpenMP_CXX_FLAGS}"
            OUTPUT_NAME   "${APP_NAME}"
            INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${OSKAR_LIB_INSTALL_DIR}"
            INSTALL_RPATH_USE_LINK_PATH TRUE
        )
    endif()

    # Create the install target if the NO_INSTALL option isn't specified.
    if (NOT APP_NO_INSTALL)
        install(TARGETS ${target} DESTINATION ${OSKAR_BIN_INSTALL_DIR})
    endif()

endmacro(OSKAR_APP)


# Macro to build and install oskar apps that depend on Qt.
#
# Usage:
#   oskar_app(name
#         [NO_INSTALL]
#         source1 source2 ...
#         [QT_MOC_SRC source1 source2 ...]
#         [EXTRA_LIBS lib1 lib2 ...]
#   )
#
# name       = Name of the app (binary name)
# source1..N = List of sources from which to build the app
# QT_MOC_SRC = List of QT moc headers which require a precompile step.
# EXTRA_LIBS = List of additional libraries to link the app against.
# NO_INSTALL = Do not install this app (i.e. don't add to the make install
#              target).
#
# Note: Named options can appear in any order
#
macro(OSKAR_QT_APP)
    parse_arguments(APP   # prefix
        "QT_MOC_SRC;EXTRA_LIBS" # arg names
        "NO_INSTALL"      # option names
        ${ARGN}
    )
    CAR(APP_NAME ${APP_DEFAULT_ARGS})
    CDR(APP_SOURCES ${APP_DEFAULT_ARGS})

    if (NOT QT4_FOUND)
        message(CRITICAL "Unable to build oskar app ${APP_NAME}, Qt4 not found!")
    endif ()

    #message("APP: NAME       = ${APP_NAME}")
    #message("     SRC        = ${APP_SOURCES}")
    #message("     MOC        = ${APP_QT_MOC_SRC}")
    #message("     EXTRA_LIBS = ${APP_EXTRA_LIBS}")
    #if (NOT APP_NO_INSTALL)
    #    message("     INSTALL    = yes")
    #else()
    #    message("     INSTALL    = no")
    #endif()

    # Create a target name from the app name.
    # Note:
    # - Appending app allows for binaries to exist with the same name
    # as targets used elsewhere.
    # - app targets can then be build individually using the command:
    #      $ make <name>_app
    #   where <name> is the (binary) name of the app to be built.
    set(target "${APP_NAME}_app")

    qt4_wrap_cpp(APP_SOURCES ${APP_QT_MOC_SRC})

    add_executable(${target} ${APP_SOURCES})
    target_link_libraries(${target}
        oskar oskar_apps ${QT_QTCORE_LIBRARY} # default libs
        ${APP_EXTRA_LIBS}                     # extra libs
    )
    # We do need the OpenMP flags here, otherwise programs will crash.
    if (MSVC)
        set_target_properties(${target} PROPERTIES
            COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
            OUTPUT_NAME   "${APP_NAME}"
            INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${OSKAR_LIB_INSTALL_DIR}"
            INSTALL_RPATH_USE_LINK_PATH TRUE
        )
    else ()
        set_target_properties(${target} PROPERTIES
            COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
            LINK_FLAGS    "${OpenMP_CXX_FLAGS}"
            OUTPUT_NAME   "${APP_NAME}"
            INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${OSKAR_LIB_INSTALL_DIR}"
            INSTALL_RPATH_USE_LINK_PATH TRUE
        )
    endif ()

    # Create the install target if the NO_INSTALL option isn't specified.
    if (NOT APP_NO_INSTALL)
        install(TARGETS ${target} DESTINATION ${OSKAR_BIN_INSTALL_DIR})
    endif()

endmacro(OSKAR_QT_APP)


# Path to the mex function defintion file.
set(mex_function_def ${OSKAR_SOURCE_DIR}/matlab/mex_function.def)

# Macro to build and install oskar mex functions.
#
# Usage:
#       oskar_qt_mex(name
#           source
#           [EXTRA_LIBS lib1 lib2 ...])
#
#       name       = name of mex function
#       source     = source file containing mex function
#       EXTRA_LIBS = list of additional libraries to link against.
#
macro(OSKAR_MEX)

    parse_arguments(MEX   # prefix
        "EXTRA_LIBS"      # arg names
        ""                # option names
        ${ARGN}
    )
    CAR(MEX_NAME ${MEX_DEFAULT_ARGS})
    CDR(MEX_SOURCES ${MEX_DEFAULT_ARGS})

    if (NOT MATLAB_FOUND)
        message(CRITICAL "Unable to build mex functions without a MATLAB install!")
    endif ()

    if (NOT DEFINED OSKAR_MEX_INSTALL_DIR)
        set(OSKAR_MEX_INSTALL_DIR ${OSKAR_MATLAB_INSTALL_DIR})
    endif()

    # Get a unique build target name
    get_filename_component(target ${MEX_SOURCES} NAME_WE)

    add_library(${target} SHARED ${MEX_SOURCES} ${mex_function_def})
    target_link_libraries(${target}
        oskar ${MATLAB_LIBRARIES}  # Default libraries
        ${MEX_EXTRA_LIBS})         # Extra libraries

    set_target_properties(${target} PROPERTIES
        PREFIX        ""
        OUTPUT_NAME   ${MEX_NAME}
        SUFFIX        ".${MATLAB_MEXFILE_EXT}"
        COMPILE_FLAGS ${MATLAB_CXX_FLAGS}
        LINK_FLAGS    ${MATLAB_CXX_FLAGS}
        INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${OSKAR_LIB_INSTALL_DIR}"
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )

    # Install target for mex function.
    install(TARGETS ${target} DESTINATION ${OSKAR_MEX_INSTALL_DIR})

endmacro(OSKAR_MEX)



macro(get_svn_revision dir variable)
    find_program(SVN_EXECUTABLE svn DOC "subversion command line client")
    if (SVN_EXECUTABLE AND EXISTS ${OSKAR_SOURCE_DIR}/.svn)
        execute_process(COMMAND
            ${SVN_EXECUTABLE} info ${dir}/oskar_global.h
            OUTPUT_VARIABLE ${variable}
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        string(REGEX REPLACE "^(.*\n)?Revision: ([^\n]+).*"
            "\\2" ${variable} "${${variable}}")
    endif()
endmacro(get_svn_revision)

macro(OSKAR_SET_MSVC_RUNTIME)
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
endmacro()
