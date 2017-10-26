#
# doc/CMakeLists.txt
#

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
    #${CMAKE_CURRENT_SOURCE_DIR}/matlab
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

# MATLAB
#add_doc(
#    DOC_NAME     matlab
#    PDF_NAME     OSKAR-${OSKAR_PDF_VERSION}-MATLAB-Interface.pdf
#    TARGET_NAME  doc_single
#    DOC_DIRS     ${CMAKE_CURRENT_SOURCE_DIR}/matlab
#    TEMPLATE     ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
#    LATEX_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/oskar_latex_header_single.tex
#    LATEX_FOOTER ${CMAKE_CURRENT_SOURCE_DIR}/oskar_latex_footer.tex)

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
