# - Run Doxygen
#
# Adds a doxygen target that runs doxygen to generate the html
# and optionally the LaTeX API documentation.
# The doxygen target is added to the doc target as a dependency.
# i.e.: the API documentation is built with:
#  make doc
#
# USAGE: GLOBAL INSTALL
#
# Install it with:
#  cmake ./ && sudo make install
# Add the following to the CMakeLists.txt of your project:
#  include(UseDoxygen OPTIONAL)
# Optionally copy Doxyfile.in in the directory of CMakeLists.txt and edit it.
#
# USAGE: INCLUDE IN PROJECT
#
#  set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
#  include(UseDoxygen)
# Add the Doxyfile.in and UseDoxygen.cmake files to the projects source directory.
#
#
# CONFIGURATION
#
# To configure Doxygen you can edit Doxyfile.in and set some variables in cmake.
# Variables you may define are:
#  DOXYFILE_SOURCE_DIR - Path where the Doxygen input files are.
#  	Defaults to the current source directory.
#  DOXYFILE_EXTRA_SOURCES - Additional source diretories/files for Doxygen to scan.
#  	The Paths should be in double quotes and separated by space. e.g.:
#  	 "${CMAKE_CURRENT_BINARY_DIR}/foo.c" "${CMAKE_CURRENT_BINARY_DIR}/bar/"
#
#  DOXYFILE_OUTPUT_DIR - Path where the Doxygen output is stored.
#  	Defaults to "${CMAKE_CURRENT_BINARY_DIR}/doc".
#
#  DOXYFILE_LATEX - ON/OFF; Set to "ON" if you want the LaTeX documentation
#  	to be built.
#  DOXYFILE_LATEX_DIR - Directory relative to DOXYFILE_OUTPUT_DIR where
#  	the Doxygen LaTeX output is stored. Defaults to "latex".
#
#  DOXYFILE_HTML_DIR - Directory relative to DOXYFILE_OUTPUT_DIR where
#  	the Doxygen html output is stored. Defaults to "html".
#

#
#  Copyright (c) 2009, 2010, 2011 Tobias Rautenkranz <tobias@rautenkranz.ch>
#
#  Redistribution and use is allowed according to the terms of the New
#  BSD license.
#  For details see the accompanying COPYING-CMAKE-SCRIPTS file.
#

#
# Heavily modified for use with OSKAR and multi-document targets.
#

find_package(Doxygen)

#
#
#
macro(usedoxygen_set_default name value type docstring)
    if(NOT DEFINED "${name}")
        set("${name}" "${value}" CACHE "${type}" "${docstring}")
    endif()
endmacro()


#
# This macro adds a documentation target to OSKAR
#
# It serves a number of purposes:
#
# 1. Defines cmake variables that are replaced in the target specified
#    Doxyfile.in (TEMPLATE). If not specified defaults are used.
#     a. DOXYFILE_OUTPUT_DIR    - from DOC_NAME
#     b. DOXYFILE_INPUTS        - from DOC_INPUTS
#     c. DOXYFILE_LATEX_HEADER  - from LATEX_HEADER
# 2. Creates a target for the documentaton based on the value of the
#    DOC_NAME and appends this target to the specified TARGET_NAME
# 3. It provides and interface for spcifying per target doxygen build
#    options such as building latex.
#
# Example:
#
# add_doc(
#     DOC_NAME     user_doc
#     TARGET_NAME  doc
#     DOC_INPUTS   ${DOXYFILE_INPUTS}
#     TEMPLATE     ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
#     LATEX_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/oskar_latex_header.tex
#     NO_LATEX
#     VERBOSE)
#
# Defines the CMAKE variables:
#   - DOXYFILE_OUTPUT_DIR
#   - DOXYFILE_INPUTS
#   - DOXYFILE_LATEX_HEADER
#
# These are used for substitution in the Doxyfile.in template.
#
#
# Other variables that effect this macro (set them prior to calling this macro)
# to alter default behaviour.
#   - DOXYFILE_PROJECT_NAME
#   - DOXYFILE_PROJECT_NUMBER
#   - DOXYFILE_LATEX_EXTRA_FILES
#
function(add_doc)
    if (DOXYGEN_FOUND)
        # http://www.cmake.org/cmake/help/v2.8.3/cmake.html#module:CMakeParseArguments
        set(options NO_LATEX VERBOSE)
        set(oneValueArgs DOC_NAME PDF_NAME TARGET_NAME TEMPLATE LATEX_HEADER)
        set(multiValueArgs DOC_DIRS DOC_FILES)
        cmake_parse_arguments(DOX
            "${options}"
            "${oneValueArgs}"
            "${multiValueArgs}"
            ${ARGN})

        if (NOT DOX_DOC_NAME)
            message(FATAL_ERROR "ERROR: add_doc() macro requires a valid DOC_NAME argument.")
        endif()

        if (NOT DOX_PDF_NAME)
            set(DOX_PDF_NAME ${DOX_DOC_NAME})
        endif()

        if (DOX_UNPARSED_ARGUMENTS)
            message(FATAL_ERROR "ERROR: Unexpected arguments passed to add_doc() macro. '${DOX_UNPARSED_ARGUMENTS}'")
        endif()

        # Set CMAKE variables to be replaced in the tempalte doxyfile.
        #======================================================================
        usedoxygen_set_default(DOXYFILE_PROJECT_NUMBER "${OSKAR_VERSION_STR}" STRING "The Doxyfile PROJECT_NUMBER tag")
        usedoxygen_set_default(DOXYFILE_HTML_DIR  "html"  STRING "Doxygen HTML output directory")
        usedoxygen_set_default(DOXYFILE_LATEX_DIR "latex" STRING "LaTeX output directory")
        usedoxygen_set_default(DOXYFILE_LATEX "NO" BOOL "Generate LaTeX API documentation" OFF)

        set(DOXYFILE_OUTPUT_DIR "${CMAKE_BINARY_DIR}/doc/${DOX_DOC_NAME}")
        list(LENGTH DOX_DOC_DIRS NUM_DOX_DIRS)
        if (${NUM_DOX_DIRS} GREATER 1)
            foreach (path_ ${DOX_DOC_DIRS})
                set(DOXYFILE_SOURCE_DIRS
                    "${DOXYFILE_SOURCE_DIRS} \\
                          \"${path_}\"")
            endforeach()
        else()
            set(DOXYFILE_SOURCE_DIRS ${DOX_DOC_DIRS})
        endif()
        set(DOXYFILE_EXAMPLE_PATH ${DOXYFILE_SOURCE_DIRS})
        set(DOXYFILE_IMAGE_PATH ${DOXYFILE_SOURCE_DIRS})
        if (DOX_DOC_FILES)
            list(LENGTH DOX_DOC_FILES NUM_DOX_FILES)
            if (${NUM_DOX_DIRS} GREATER 1 OR ${NUM_DOX_FILES} GREATER 1)
                foreach (path_ ${DOX_DOC_FILES})
                    set(DOXYFILE_SOURCE_DIRS
                        "${DOXYFILE_SOURCE_DIRS} \\
                          \"${path_}\"")
                endforeach()
            else()
                set(DOXYFILE_SOURCE_DIRS ${DOX_DOC_FILES})
            endif()
        endif()
        if (NOT ${DOX_NO_LATEX})
            set(DOXYFILE_LATEX "ON")
        endif()
        set(target_ "${DOX_TARGET_NAME}_${DOX_DOC_NAME}")
        # Check we can find the specified doxyfile template.
        if (EXISTS ${DOX_TEMPLATE})
            set(DOXYFILE_IN "${DOX_TEMPLATE}")
            find_package_handle_standard_args(DOXYFILE_IN DEFAULT_MSG "DOXYFILE_IN")
            set(DOXYFILE_IN_FOUND YES)
        else ()
            message(FATAL_ERROR "ERROR: Unable to find specified Doxyfile template.")
            set(DOXFILE_IN_FOUND DOXYFILE_IN-NOTFOUND)
        endif()
        set(DOXYFILE_OUT "Doxyfile_${DOX_DOC_NAME}")
        set(DOXYFILE "${CMAKE_CURRENT_BINARY_DIR}/${DOXYFILE_OUT}")
        set(DOXYFILE_LATEX_HEADER "${DOX_LATEX_HEADER}")
        #======================================================================
        
    endif(DOXYGEN_FOUND)

    if (DOXYGEN_FOUND AND DOXYFILE_IN_FOUND)

        if (${DOX_VERBOSE})
            message("-----------------------------------------")
            message("-- Adding documentation target")
            message("-----------------------------------------")
            message("  >> DOC NAME     : ${DOX_DOC_NAME}")
            message("  >> INPUTS       : ${DOX_DOC_INPUTS}")
            message("  >> TEMPALTE     : ${DOX_TEMPLATE}")
            message("  >> LATEX_HEADER : ${DOX_LATEX_HEADER}")
            message("  >> TARGET       : ${DOX_TARGET_NAME}")
            message("  >> NO LATEX     : ${DOX_NO_LATEX}")
            message("-----------------------------------------")
            message("  -- DOXYFILE_OUTPUT_DIR     : ${DOXYFILE_OUTPUT_DIR}")
            message("  -- DOXYFILE_SOURCE_DIRS    : ${DOXYFILE_SOURCE_DIRS}")
            message("  -- DOXYFILE_LATEX          : ${DOXYFILE_LATEX}")
            message("  -- target                  : ${target_}")
            message("  -- DOXYFILE_IN             : ${DOXYFILE_IN}")
            message("  -- DOXYFILE_OUT            : ${DOXYFILE_OUT}")
            message("  -- DOXYFILE_PROJECT_NAME   : ${DOXYFILE_PROJECT_NAME}")
            message("  -- DOXYFILE_PROJECT_NUMBER : ${DOXYFILE_PROJECT_NUMBER}")
            message("  -- DOXYFILE_LATEX_HEADER   : ${DOXYFILE_LATEX_HEADER}")
            message("  -- DOXYFILE_IMAGE_PATH     : ${DOXYFILE_IMAGE_PATH}")
            message("  -- DOXYFILE_EXAMPLE_PATH   : ${DOXYFILE_EXAMPLE_PATH}")
            message("-----------------------------------------")
        endif()

        configure_file("${DOXYFILE_IN}" "${DOXYFILE}" @ONLY)

        set_property(DIRECTORY
            APPEND PROPERTY
            ADDITIONAL_MAKE_CLEAN_FILES
            "${DOXYFILE_OUTPUT_DIR}/${DOXYFILE_HTML_DIR}")
        set_property(DIRECTORY APPEND PROPERTY
            ADDITIONAL_MAKE_CLEAN_FILES
            "${DOXYFILE_OUTPUT_DIR}/${DOXYFILE_LATEX_DIR}")

        add_custom_target(${target_}
            COMMAND "${DOXYGEN_EXECUTABLE}" "${DOXYFILE}"
            COMMENT "Writing documentation to ${DOXYFILE_OUTPUT_DIR}..."
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

        if (${DOXYFILE_LATEX})
            find_package(LATEX)
            find_program(DOXYFILE_MAKE make)
            if (LATEX_COMPILER AND MAKEINDEX_COMPILER AND DOXYFILE_MAKE)
                add_custom_command(TARGET ${target_}
                    POST_BUILD
                    COMMAND "${DOXYFILE_MAKE}"
                    COMMENT "Running LaTeX for Doxygen documentation in ${DOXYFILE_OUTPUT_DIR}/${DOXYFILE_LATEX_DIR}..."
                    WORKING_DIRECTORY "${DOXYFILE_OUTPUT_DIR}/${DOXYFILE_LATEX_DIR}")
                set(refman_src ${DOXYFILE_OUTPUT_DIR}/${DOXYFILE_LATEX_DIR}/refman.pdf)
                set(refman_dst ${CMAKE_BINARY_DIR}/doc/${DOX_PDF_NAME})
                add_custom_command(TARGET ${target_} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy ${refman_src} ${refman_dst}
                    COMMENT "Copying ${refman_src} to ${refman_dst}" VERBATIM)
            endif()
        endif()

        get_target_property(DOC_TARGET ${DOX_TARGET_NAME} TYPE)
        if(NOT DOC_TARGET)
            add_custom_target(${DOX_TARGET_NAME})
        endif()
        get_target_property(DOC_ALL_TARGET doc_all TYPE)
        if(NOT DOC_ALL_TARGET)
            add_custom_target(doc_all)
        endif()
        add_dependencies(${DOX_TARGET_NAME} ${target_})
        add_dependencies(doc_all ${target_})

    endif(DOXYGEN_FOUND AND DOXYFILE_IN_FOUND)
endfunction(add_doc)


































set(DOXYGEN_OLD OFF)

if (${DOXYGEN_OLD})
    if(DOXYGEN_FOUND)
        find_file(DOXYFILE_IN "Doxyfile.in"
            PATHS "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_ROOT}/Modules/"
            NO_DEFAULT_PATH
            DOC "Path to the doxygen configuration template file")
        set(DOXYFILE "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile")
        include(FindPackageHandleStandardArgs)
        find_package_handle_standard_args(DOXYFILE_IN DEFAULT_MSG "DOXYFILE_IN")
    endif()

    if(DOXYGEN_FOUND AND DOXYFILE_IN_FOUND)
        usedoxygen_set_default(DOXYFILE_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/doc"
            PATH "Doxygen output directory")
        usedoxygen_set_default(DOXYFILE_HTML_DIR "html"
            STRING "Doxygen HTML output directory")
        usedoxygen_set_default(DOXYFILE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
            PATH "Input files source directory")
        usedoxygen_set_default(DOXYFILE_EXTRA_SOURCE_DIRS ""
            STRING "Additional source files/directories separated by space")
        foreach (path_ ${DOXYFILE_INPUTS})
            set(DOXYFILE_INPUT
                "${DOXYFILE_INPUT} \\
                             \"${path_}\"")
        endforeach()
        #set(DOXYFILE_SOURCE_DIRS "\"${DOXYFILE_SOURCE_DIR}\" ${DOXYFILE_EXTRA_SOURCES}")
        set(DOXYFILE_SOURCE_DIRS "${DOXYFILE_INPUT}")

        usedoxygen_set_default(DOXYFILE_LATEX YES BOOL "Generate LaTeX API documentation" OFF)
        usedoxygen_set_default(DOXYFILE_LATEX_DIR "latex" STRING "LaTex output directory")

        mark_as_advanced(DOXYFILE_OUTPUT_DIR DOXYFILE_HTML_DIR DOXYFILE_LATEX_DIR
            DOXYFILE_SOURCE_DIR DOXYFILE_EXTRA_SOURCE_DIRS DOXYFILE_IN)

        set_property(DIRECTORY
            APPEND PROPERTY
            ADDITIONAL_MAKE_CLEAN_FILES
            "${DOXYFILE_OUTPUT_DIR}/${DOXYFILE_HTML_DIR}")

        add_custom_target(doxygen
            COMMAND "${DOXYGEN_EXECUTABLE}"
                "${DOXYFILE}"
            COMMENT "Writing documentation to ${DOXYFILE_OUTPUT_DIR}..."
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

        set(DOXYFILE_DOT "NO")
        if(DOXYGEN_DOT_EXECUTABLE)
            set(DOXYFILE_DOT "YES")
        endif()

        ## LaTeX
        set(DOXYFILE_PDFLATEX "NO")

        set_property(DIRECTORY APPEND PROPERTY
            ADDITIONAL_MAKE_CLEAN_FILES
            "${DOXYFILE_OUTPUT_DIR}/${DOXYFILE_LATEX_DIR}")

        if(DOXYFILE_LATEX STREQUAL "ON")
            set(DOXYFILE_GENERATE_LATEX "YES")
            find_package(LATEX)
            find_program(DOXYFILE_MAKE make)
            mark_as_advanced(DOXYFILE_MAKE)
            if(LATEX_COMPILER AND MAKEINDEX_COMPILER AND DOXYFILE_MAKE)
                if(PDFLATEX_COMPILER)
                    set(DOXYFILE_PDFLATEX "YES")
                endif()
            add_custom_command(TARGET doxygen
                POST_BUILD
                COMMAND "${DOXYFILE_MAKE}"
                COMMENT	"Running LaTeX for Doxygen documentation in ${DOXYFILE_OUTPUT_DIR}/${DOXYFILE_LATEX_DIR}..."
                WORKING_DIRECTORY "${DOXYFILE_OUTPUT_DIR}/${DOXYFILE_LATEX_DIR}")
            else()
            set(DOXYGEN_LATEX "NO")
            endif()
        else()
            set(DOXYFILE_GENERATE_LATEX "NO")
        endif()

        configure_file("${DOXYFILE_IN}" "${DOXYFILE}" @ONLY)

        get_target_property(DOC_TARGET doc TYPE)
        if(NOT DOC_TARGET)
            add_custom_target(doc)
        endif()

        add_dependencies(doc doxygen)
    endif()
endif()
