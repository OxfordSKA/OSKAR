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
        "NO_INSTALL"   # option names
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
        ${APP_EXTRA_LIBS}    # extra libs
    )
    set_target_properties(${target} PROPERTIES
        OUTPUT_NAME   ${APP_NAME}
    )

    # FIXME hack for now to only build the oskar.app (make this an option?)
    set(OSX_APP FALSE)
    #if (${APP_NAME} STREQUAL "oskar")
    #    set(OSX_APP TRUE)
    #endif ()
    if (APPLE AND OSX_APP)
        set(target "${APP_NAME}_osx_app")
        add_executable(${target} MACOSX_BUNDLE ${APP_SOURCES})
        set(MACOSX_BUNDLE_BUNDLE_NAME ${APP_NAME})
        set(MACOSX_BUNDLE_ICON_FILE ${PROJECT_SOURCE_DIR}/widgets/icons/icon.icns)
        #add_custom_command(TARGET ${target} POST_BUILD
        #    COMMAND mkdir ARGS ${CMAKE_CURRENT_BINARY_DIR}/${APP_NAME}.app/Contents/Resources
        #    COMMAND cp ARGS ${MACOSX_BUNDLE_ICON_FILE} ${CMAKE_CURRENT_BINARY_DIR}/${APP_NAME}.app/Contents/Resources
        #    #COMMAND cp ARGS *.qm ${CMAKE_CURRENT_BINARY_DIR}/${target}.app/Content/Resources
        #)
        target_link_libraries(${target}
            ${QT_QTCORE_LIBRARY} # default libs
            ${APP_EXTRA_LIBS}    # extra libs
        )
        set_target_properties(${target} PROPERTIES
            OUTPUT_NAME   ${APP_NAME}
        )
    endif (APPLE AND OSX_APP)

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
        )
    else ()
        set_target_properties(${target} PROPERTIES
            COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
            LINK_FLAGS    "${OpenMP_CXX_FLAGS}"
            OUTPUT_NAME   "${APP_NAME}"
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
        # Defaults libs
        oskar
        oskar_apps
        ${QT_QTCORE_LIBRARY}
        # Extra libs
        ${APP_EXTRA_LIBS}
    )
    # We do need the OpenMP flags here, otherwise programs will crash.
    if (MSVC)
        set_target_properties(${target} PROPERTIES
            COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
            OUTPUT_NAME   "${APP_NAME}"
        )
    else ()
        set_target_properties(${target} PROPERTIES
            COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
            LINK_FLAGS    "${OpenMP_CXX_FLAGS}"
            OUTPUT_NAME   "${APP_NAME}"
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
        "LIBS"      # arg names
        ""          # option names
        ${ARGN})

    CAR(MEX_NAME ${MEX_DEFAULT_ARGS})
    CDR(MEX_SOURCES ${MEX_DEFAULT_ARGS})

    if (NOT MATLAB_FOUND)
        message(CRITICAL "Unable to build mex functions without a MATLAB install!")
    endif ()

    if (NOT DEFINED OSKAR_MEX_INSTALL_DIR)
        set(OSKAR_MEX_INSTALL_DIR ${OSKAR_MATLAB_INSTALL_DIR})
    endif()

    # Over-ride compiler flags for mex functions
    set(CMAKE_CXX_FLAGS "")
    set(CMAKE_CXX_FLAGS_RELEASE "-O2")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")

    # OS specific options.
    if (APPLE)
        set(MATLAB_MEXFILE_EXT mexmaci64)
        set(MATLAB_COMPILE_FLAGS "-DMATLAB_MEX_FILE -DMX_COMPAT_32 -fno-common -fexceptions")
        set(MATLAB_LINK_FLAGS "")
    else ()
        set(MATLAB_MEXFILE_EXT mexa64)
        set(CMAKE_CXX_FLAGS "-fPIC")
        set(MATLAB_COMPILE_FLAGS "-DMATLAB_MEX_FILE -DMX_COMPAT_32")
        set(MATLAB_LINK_FLAGS "")
    endif(APPLE)

    # Get a unique build target name
    get_filename_component(target ${MEX_SOURCES} NAME_WE)

    add_library(${target} MODULE ${MEX_SOURCES} ${mex_function_def})
    target_link_libraries(${target} ${MATLAB_LIBRARIES} ${MEX_LIBS})
    set_target_properties(${target} PROPERTIES
        PREFIX        ""
        OUTPUT_NAME   "${MEX_NAME}"
        SUFFIX        ".${MATLAB_MEXFILE_EXT}"
        COMPILE_FLAGS "${MATLAB_COMPILE_FLAGS}"
        LINK_FLAGS    "${MATLAB_LINK_FLAGS}"
    )

    # Install target for mex function.
    install(TARGETS ${target} DESTINATION ${OSKAR_MEX_INSTALL_DIR})

endmacro(OSKAR_MEX)


