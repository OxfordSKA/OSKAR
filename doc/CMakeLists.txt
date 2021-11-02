#
# doc/CMakeLists.txt
#

find_package(Sphinx)
find_package(PythonInterp)

# Define a function to add build targets for the Sphinx documentation.
# It will create build targets based on the target root name,
# plus "_html" and "_pdf" to build the HTML and PDF versions.
#
# add_sphinx_doc(
#     TARGET_ROOT <target root name>
#     DOC_SOURCE_DIR <source directory>
# )
#
function(add_sphinx_doc)
    if (Sphinx_FOUND AND PythonInterp_FOUND)
        set(options)
        set(oneValueArgs TARGET_ROOT DOC_SOURCE_DIR)
        set(multiValueArgs)
        cmake_parse_arguments(SPHINX
            "${options}"
            "${oneValueArgs}"
            "${multiValueArgs}"
            ${ARGN})

        if (NOT SPHINX_TARGET_ROOT)
            message(FATAL_ERROR "ERROR: add_sphinx_doc() requires a valid TARGET_ROOT.")
        endif()
        if (NOT SPHINX_DOC_SOURCE_DIR)
            message(FATAL_ERROR "ERROR: add_sphinx_doc() requires a valid DOC_SOURCE_DIR.")
        endif()
        if (SPHINX_UNPARSED_ARGUMENTS)
            message(FATAL_ERROR "ERROR: Unexpected arguments passed to add_sphinx_doc(): '${SPHINX_UNPARSED_ARGUMENTS}'")
        endif()

        set(SPHINX_BINARY_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/_build")
        set(SPHINX_CACHE_DIR "${CMAKE_CURRENT_BINARY_DIR}/_doctrees")

        configure_file("${SPHINX_DOC_SOURCE_DIR}/conf.py"
            "${SPHINX_BINARY_BUILD_DIR}/conf.py" @ONLY)

        # message("-- Using Python: ${PYTHON_EXECUTABLE}")
        add_custom_target(${SPHINX_TARGET_ROOT}_html
            COMMAND "${PYTHON_EXECUTABLE}"
                "${PROJECT_BINARY_DIR}/doc/settings/xml_to_rst.py"
            COMMAND "${SPHINX_EXECUTABLE}"
                -M html
                "${SPHINX_DOC_SOURCE_DIR}"
                "${SPHINX_BINARY_BUILD_DIR}"
                -c "${SPHINX_BINARY_BUILD_DIR}"
                -d "${SPHINX_CACHE_DIR}"
            COMMENT "Building HTML documentation with Sphinx"
        )
        add_custom_target(${SPHINX_TARGET_ROOT}_pdf
            COMMAND "${PYTHON_EXECUTABLE}"
                "${PROJECT_BINARY_DIR}/doc/settings/xml_to_rst.py"
            COMMAND "${SPHINX_EXECUTABLE}"
                -M latexpdf
                "${SPHINX_DOC_SOURCE_DIR}"
                "${SPHINX_BINARY_BUILD_DIR}"
                -c "${SPHINX_BINARY_BUILD_DIR}"
                -d "${SPHINX_CACHE_DIR}"
            COMMENT "Building PDF documentation with Sphinx"
        )
    else()
        message("-- WARNING: Sphinx or Python not found - cannot build documentation")
    endif()
endfunction()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/settings/xml_to_rst.py
    ${CMAKE_CURRENT_BINARY_DIR}/settings/xml_to_rst.py @ONLY)

add_sphinx_doc(
    TARGET_ROOT doc
    DOC_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
)


# -------------------------------------------------
# FIXME Remove the rest of this file (old PDF docs)
# -------------------------------------------------
set(OSKAR_PDF_VERSION "${OSKAR_VERSION_MAJOR}.${OSKAR_VERSION_MINOR}")
set(DOXYFILE_PROJECT_NAME "OSKAR User Documentation")
set(DOXYFILE_LATEX_EXTRA_FILES "${CMAKE_SOURCE_DIR}/doc/oskar_latex.sty")
include(UseDoxygen/UseDoxygen)

# User documentation
set(USER_DOC_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/install
    ${CMAKE_CURRENT_SOURCE_DIR}/example
    ${CMAKE_CURRENT_SOURCE_DIR}/theory
    ${CMAKE_CURRENT_SOURCE_DIR}/apps
    ${CMAKE_CURRENT_SOURCE_DIR}/sky_model
    ${CMAKE_CURRENT_SOURCE_DIR}/telescope_model
    ${CMAKE_CURRENT_SOURCE_DIR}/pointing_file
    ${CMAKE_CURRENT_SOURCE_DIR}/settings
    ${CMAKE_CURRENT_SOURCE_DIR}/binary_file
)

add_doc(
    DOC_NAME     user
    PDF_NAME     OSKAR-${OSKAR_PDF_VERSION}-User-Documentation.pdf
    TARGET_NAME  doc
    DOC_DIRS     ${USER_DOC_DIRS}
    DOC_FILES    ${CMAKE_CURRENT_SOURCE_DIR}/intro.dox
    TEMPLATE     ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
    LATEX_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/oskar_latex_header.tex
    LATEX_FOOTER ${CMAKE_CURRENT_SOURCE_DIR}/oskar_latex_footer.tex)

# Intro
add_doc(
    DOC_NAME     intro
    PDF_NAME     OSKAR-${OSKAR_PDF_VERSION}-Introduction.pdf
    TARGET_NAME  doc_single
    DOC_FILES    ${CMAKE_CURRENT_SOURCE_DIR}/intro.dox
    TEMPLATE     ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
    LATEX_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/oskar_latex_header_single.tex
    LATEX_FOOTER ${CMAKE_CURRENT_SOURCE_DIR}/oskar_latex_footer.tex)

# Release Notes
add_doc(
    DOC_NAME     release_notes
    PDF_NAME     OSKAR-${OSKAR_PDF_VERSION}-Release-Notes.pdf
    TARGET_NAME  doc_single
    DOC_DIRS     ${CMAKE_CURRENT_SOURCE_DIR}/release_notes
    TEMPLATE     ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
    LATEX_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/oskar_latex_header_single.tex
    LATEX_FOOTER ${CMAKE_CURRENT_SOURCE_DIR}/oskar_latex_footer.tex)

# Apps
add_doc(
    DOC_NAME     apps
    PDF_NAME     OSKAR-${OSKAR_PDF_VERSION}-Apps.pdf
    TARGET_NAME  doc_single
    DOC_DIRS     ${CMAKE_CURRENT_SOURCE_DIR}/apps
    TEMPLATE     ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
    LATEX_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/oskar_latex_header_single.tex
    LATEX_FOOTER ${CMAKE_CURRENT_SOURCE_DIR}/oskar_latex_footer.tex)

# Binary file
add_doc(
    DOC_NAME     binary_file
    PDF_NAME     OSKAR-${OSKAR_PDF_VERSION}-Binary-File-Format.pdf
    TARGET_NAME  doc_single
    DOC_DIRS     ${CMAKE_CURRENT_SOURCE_DIR}/binary_file
    TEMPLATE     ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
    LATEX_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/oskar_latex_header_single.tex
    LATEX_FOOTER ${CMAKE_CURRENT_SOURCE_DIR}/oskar_latex_footer.tex)

# Example
add_doc(
    DOC_NAME     example
    PDF_NAME     OSKAR-${OSKAR_PDF_VERSION}-Example.pdf
    TARGET_NAME  doc_single
    DOC_DIRS     ${CMAKE_CURRENT_SOURCE_DIR}/example
    TEMPLATE     ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
    LATEX_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/oskar_latex_header_single.tex
    LATEX_FOOTER ${CMAKE_CURRENT_SOURCE_DIR}/oskar_latex_footer.tex)

# Install
add_doc(
    DOC_NAME     install
    PDF_NAME     OSKAR-${OSKAR_PDF_VERSION}-Install.pdf
    TARGET_NAME  doc_single
    DOC_DIRS     ${CMAKE_CURRENT_SOURCE_DIR}/install
    TEMPLATE     ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
    LATEX_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/oskar_latex_header_single.tex
    LATEX_FOOTER ${CMAKE_CURRENT_SOURCE_DIR}/oskar_latex_footer.tex)

# Pointing file
add_doc(
    DOC_NAME     pointing_file
    PDF_NAME     OSKAR-${OSKAR_PDF_VERSION}-Pointing-File.pdf
    TARGET_NAME  doc_single
    DOC_DIRS     ${CMAKE_CURRENT_SOURCE_DIR}/pointing_file
    TEMPLATE     ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
    LATEX_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/oskar_latex_header_single.tex
    LATEX_FOOTER ${CMAKE_CURRENT_SOURCE_DIR}/oskar_latex_footer.tex)

# Settings
add_doc(
    DOC_NAME     settings
    PDF_NAME     OSKAR-${OSKAR_PDF_VERSION}-Settings.pdf
    TARGET_NAME  doc_single
    DOC_DIRS     ${CMAKE_CURRENT_SOURCE_DIR}/settings
    TEMPLATE     ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
    LATEX_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/oskar_latex_header_single.tex
    LATEX_FOOTER ${CMAKE_CURRENT_SOURCE_DIR}/oskar_latex_footer.tex)

# Sky model
add_doc(
    DOC_NAME     sky_model
    PDF_NAME     OSKAR-${OSKAR_PDF_VERSION}-Sky-Model.pdf
    TARGET_NAME  doc_single
    DOC_DIRS     ${CMAKE_CURRENT_SOURCE_DIR}/sky_model
    TEMPLATE     ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
    LATEX_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/oskar_latex_header_single.tex
    LATEX_FOOTER ${CMAKE_CURRENT_SOURCE_DIR}/oskar_latex_footer.tex)

# Telescope model
add_doc(
    DOC_NAME     telescope_model
    PDF_NAME     OSKAR-${OSKAR_PDF_VERSION}-Telescope-Model.pdf
    TARGET_NAME  doc_single
    DOC_DIRS     ${CMAKE_CURRENT_SOURCE_DIR}/telescope_model
    TEMPLATE     ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
    LATEX_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/oskar_latex_header_single.tex
    LATEX_FOOTER ${CMAKE_CURRENT_SOURCE_DIR}/oskar_latex_footer.tex)

# Theory
add_doc(
    DOC_NAME     theory
    PDF_NAME     OSKAR-${OSKAR_PDF_VERSION}-Theory.pdf
    TARGET_NAME  doc_single
    DOC_DIRS     ${CMAKE_CURRENT_SOURCE_DIR}/theory
    TEMPLATE     ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
    LATEX_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/oskar_latex_header_single.tex
    LATEX_FOOTER ${CMAKE_CURRENT_SOURCE_DIR}/oskar_latex_footer.tex)



configure_file(${CMAKE_CURRENT_SOURCE_DIR}/settings/xml_to_settings_doc.py
    ${CMAKE_CURRENT_BINARY_DIR}/settings/xml_to_settings_doc.py @ONLY)
