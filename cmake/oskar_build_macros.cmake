#
# cmake/oskar_build_macros.cmake
#
#===============================================================================
# A collection of cmake macros used by the oskar build system.
#


# Path to the mex function defintion file.
set(mex_function_def ${OSKAR_SOURCE_DIR}/matlab/mex_function.def)

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


# Macro to build and install oskar mex functions.
#
# Usage:
#       oskar_qt_mex(name source [extra_libs])
#
#       name       = name of mex function
#       source     = source file containing mex function
#       extra_libs = list of extra libraries to link against.
#
# Source files with the *.cu extension are build using the CUDA compiler.
#
macro(oskar_mex name source)

    #message("= INFO: Building mex function: ${name} (${source})")

    # TODO: Add script to find MATLAB and MATLAB_FOUND guard?
    # TODO: Add critical fail on matlab not found.
    # TODO: Add critical fail on not having set other required variables.

    if (NOT DEFINED OSKAR_MEX_INSTALL_DIR)
        #message("-- NOTE: OSKAR_MEX_INSTALL_DIR not defined in "
        #        "source directory ${CMAKE_CURRENT_SOURCE_DIR}. "
        #        "Using default: ${OSKAR_MATLAB_INSTALL_DIR}")
        set(OSKAR_MEX_INSTALL_DIR ${OSKAR_MATLAB_INSTALL_DIR})
    else()
        #message("-- NOTE: OSKAR_MEX_INSTALL_DIR defined as ${OSKAR_MEX_INSTALL_DIR}")
    endif ()

    # Construct the build target name
    get_filename_component(target ${source} NAME_WE)
    #message("MEX: target = ${target}")
    #message("     name   = ${name}")
    #message("     dir    = ${OSKAR_MEX_INSTALL_DIR}\n")

    # Find out if this is a CUDA or C/C++ source file based on the extension
    # and use the appropriate add_library target.
    get_filename_component(ext ${source} EXT)
    if (${ext} STREQUAL ".cpp" OR ${ext} STREQUAL "*.c")
        add_library(${target} SHARED ${source} ${mex_function_def})
    elseif (${ext} STREQUAL ".cu")
        cuda_add_library(${target} SHARED ${source} ${mex_function_def})
    else ()
        message(CRITICAL "OSKAR_MEX: UNRECOGNISED SOURCE FILE EXTENSION!")
    endif ()

    target_link_libraries(${target} 
        oskar 
        ${MATLAB_LIBRARIES}
        ${ARGN})
    
    set_target_properties(${target} PROPERTIES
        PREFIX ""
        OUTPUT_NAME ${name} 
        SUFFIX ".${MATLAB_MEXFILE_EXT}"
        INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${OSKAR_LIB_INSTALL_DIR}"
        INSTALL_RPATH_USE_LINK_PATH TRUE
        COMPILE_FLAGS ${MATLAB_CXX_FLAGS}
        LINK_FLAGS ${MATLAB_CXX_FLAGS})
        
    # Install target for mex function.
    install(TARGETS ${target} DESTINATION ${OSKAR_MEX_INSTALL_DIR})
endmacro(oskar_mex)





# Macro to build and install oskar mex functions using Qt.
#
# Usage:
#       oskar_qt_mex(name source [extra_libs])
#
#       name       = name of mex function
#       source     = source file containing mex function
#       extra_libs = list of extra libraries to link against.
#
# Source files with the *.cu extension are build using the CUDA compiler.
#
macro(oskar_qt_mex name source)

    if (QT4_FOUND)

        #message("= INFO: Building mex function: ${name} (${source})")

        # TODO: Add script to find MATLAB and MATLAB_FOUND guard?
        # TODO: Add critical fail on matlab not found.
        # TODO: Add critical fail on not having set other required variables.

        if (NOT DEFINED OSKAR_MEX_INSTALL_DIR)
            #message("-- NOTE: OSKAR_MEX_INSTALL_DIR not defined in "
            #        "source directory ${CMAKE_CURRENT_SOURCE_DIR}. "
            #        "Using default: ${OSKAR_MATLAB_INSTALL_DIR}")
            set(OSKAR_MEX_INSTALL_DIR ${OSKAR_MATLAB_INSTALL_DIR})
        else()
            #message("-- NOTE: OSKAR_MEX_INSTALL_DIR defined as ${OSKAR_MEX_INSTALL_DIR}")
        endif ()

        # Construct the build target name
        get_filename_component(target ${source} NAME_WE)
        #message("MEX: target = ${target}")
        #message("     name   = ${name}")
        #message("     dir    = ${OSKAR_MEX_INSTALL_DIR}\n")

        # Find out if this is a CUDA or C/C++ source file based on the extension
        # and use the appropriate add_library target.
        get_filename_component(ext ${source} EXT)
        if (${ext} STREQUAL ".cpp" OR ${ext} STREQUAL "*.c")
            add_library(${target} SHARED ${source} ${mex_function_def})
        elseif (${ext} STREQUAL ".cpp" OR ${ext} STREQUAL "*.c")
            cuda_add_library(${target} SHARED ${source} ${mex_function_def})
        else (${ext} STREQUAL ".cpp" OR ${ext} STREQUAL "*.c")
            message(CRITICAL "OSKAR_MEX: UNRECOGNISED SOURCE FILE EXTENSION!")
        endif (${ext} STREQUAL ".cpp" OR ${ext} STREQUAL "*.c")

        target_link_libraries(${target} 
            oskar 
            ${MATLAB_QT_QTCORE_LIBRARY}
            ${MATLAB_LIBRARIES}
            ${ARGN})

        set_target_properties(${target} PROPERTIES
            PREFIX "" 
            OUTPUT_NAME ${name}
            SUFFIX ".${MATLAB_MEXFILE_EXT}"
            INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${OSKAR_LIB_INSTALL_DIR}
            INSTALL_RPATH_USE_LINK_PATH TRUE
            COMPILE_FLAGS ${MATLAB_CXX_FLAGS}
            LINK_FLAGS ${MATLAB_CXX_FLAGS})
            
        # Install target for mex function.
        install(TARGETS ${target} DESTINATION ${OSKAR_MEX_INSTALL_DIR})

    endif (QT4_FOUND)

endmacro(oskar_qt_mex)



# Macro to build oskar apps.
#
# Usage:
#   oskar_app(name
#         [NO_INSTALL] 
#         source1 source2 ...
#         [QT_MOC_SRC source1 source2 ...]
#         [LIBS lib1 lib2 ...]
#   )
#
# name       = Name of the app (binary name)
# source1..N = List of sources from which to build the app
# QT_MOC_SRC = List of QT moc headers which require a precompile step.
# LIBS       = List of additional libraries to link the app against.
# NO_INSTALL = Do not install this app (i.e. dont add to the make install
#              target).
#
# Note: Named options can appear in any order
#
macro(OSKAR_APP)
    parse_arguments(APP   # prefix
        "QT_MOC_SRC;LIBS" # arg names
        "NO_INSTALL"      # option names
        ${ARGN}
    )
    CAR(APP_NAME ${APP_DEFAULT_ARGS})
    CDR(APP_SOURCES ${APP_DEFAULT_ARGS})
    
    #message("APP: NAME    = ${APP_NAME}")
    #message("     SRC     = ${APP_SOURCES}")
    #message("     MOC     = ${APP_QT_MOC_SRC}")
    #message("     LIBS    = ${APP_LIBS}")
    #if (NOT APP_NO_INSTALL)
    #    message("     INSTALL = yes")
    #else()
    #    message("     INSTALL = no")
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
        ${APP_LIBS}                           # extra libs
    ) 
    set_target_properties(${target} PROPERTIES
        COMPILE_FLAGS ${OpenMP_CXX_FLAGS}
        LINK_FLAGS    ${OpenMP_CXX_FLAGS}
        OUTPUT_NAME   ${APP_NAME}
        INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${OSKAR_LIB_INSTALL_DIR}
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
    
    # Create the install target if the NO_INSTALL option isn't specified.
    if (NOT APP_NO_INSTALL)
        install(TARGETS ${target} DESTINATION ${OSKAR_BIN_INSTALL_DIR})
    endif()
    
endmacro(OSKAR_APP)








