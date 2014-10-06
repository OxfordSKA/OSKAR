#
# doc/CMakeLists.txt
#

include(UseDoxygen/UseDoxygen)

set(DOXYFILE_PROJECT_NAME "OSKAR User Documentation")
set(DOXYFILE_LATEX_FOOTER "${CMAKE_SOURCE_DIR}/doc/oskar_latex_footer.tex")
set(DOXYFILE_LATEX_EXTRA_FILES "${CMAKE_SOURCE_DIR}/doc/oskar_latex.sty")
#
# Note/FIXME (src/doc/Doxyfile.in):
# - EXAMPLE_PATH is current hardcoded on line 875 of the doxyfile!
# - IMAGE_PATH is current hardcoded on line 895 of the doxyfile!
#
#if(DOXYGEN_FOUND) # Needed for older versions of doxygen?
#    include(oskar_copy_files)
#    copy_file(${CMAKE_CURRENT_SOURCE_DIR}/oskar_latex.sty
#        ${CMAKE_CURRENT_BINARY_DIR}/oskar_latex.sty)
#endif()


# User documentation
set(USER_DOC_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/intro.dox
    ${CMAKE_CURRENT_SOURCE_DIR}/install
    ${CMAKE_CURRENT_SOURCE_DIR}/example
    ${CMAKE_CURRENT_SOURCE_DIR}/theory
    ${CMAKE_CURRENT_SOURCE_DIR}/apps
    ${CMAKE_CURRENT_SOURCE_DIR}/sky_model
    ${CMAKE_CURRENT_SOURCE_DIR}/telescope_model
    ${CMAKE_CURRENT_SOURCE_DIR}/pointing_file
    ${CMAKE_CURRENT_SOURCE_DIR}/matlab
    ${CMAKE_CURRENT_SOURCE_DIR}/settings
    ${CMAKE_CURRENT_SOURCE_DIR}/binary_file
)
set(DOXYFILE_EXAMPLE_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/settings)
set(DOXYFILE_IMAGE_PATHS
    ${CMAKE_CURRENT_SOURCE_DIR}/example
    ${CMAKE_CURRENT_SOURCE_DIR}/theory
    ${CMAKE_CURRENT_SOURCE_DIR}/sky_model
    ${CMAKE_CURRENT_SOURCE_DIR}/telescope_model
    ${CMAKE_CURRENT_SOURCE_DIR}/matlab
    ${CMAKE_CURRENT_SOURCE_DIR}/binary_file
)
add_doc(
    DOC_NAME     user
    TARGET_NAME  doc
    DOC_INPUTS   ${USER_DOC_SRC}
    TEMPLATE     ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
    LATEX_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/oskar_latex_header.tex)

set(DOXYFILE_LATEX_FOOTER "${CMAKE_SOURCE_DIR}/doc/oskar_latex_footer_single.tex")
set(DOXYFILE_IMAGE_PATHS "")
set(DOXYFILE_EXAMPLE_PATHS "")
# Install documentation
add_doc(
    DOC_NAME     install
    TARGET_NAME  doc_single
    DOC_INPUTS   ${CMAKE_CURRENT_SOURCE_DIR}/install
    TEMPLATE     ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
    LATEX_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/oskar_latex_header_single.tex)

# Settings documentation
set(DOXYFILE_EXAMPLE_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/settings)
set(DOXYFILE_IMAGE_PATHS "")
add_doc(
    DOC_NAME     settings
    TARGET_NAME  doc_single
    DOC_INPUTS   ${CMAKE_CURRENT_SOURCE_DIR}/settings
    TEMPLATE     ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
    LATEX_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/oskar_latex_header_single.tex)

# binary file documentation
set(DOXYFILE_IMAGE_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/binary_file)
set(DOXYFILE_EXAMPLE_PATHS "")
add_doc(
    DOC_NAME     binary_file
    TARGET_NAME  doc_single
    DOC_INPUTS   ${CMAKE_CURRENT_SOURCE_DIR}/binary_file
    TEMPLATE     ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
    LATEX_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/oskar_latex_header_single.tex)











