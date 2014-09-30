#
# doc/CMakeLists.txt
#

set(DOXYFILE_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(DOXYFILE_LATEX ON)
include(oskar_copy_files)
include(UseDoxygen)
copy_file(${CMAKE_CURRENT_SOURCE_DIR}/version.txt
    ${CMAKE_CURRENT_BINARY_DIR}/version.txt)
copy_file(${CMAKE_CURRENT_SOURCE_DIR}/oskar_latex.sty
    ${CMAKE_CURRENT_BINARY_DIR}/oskar_latex.sty)


#add_doxygen_target(doc Doxyfile)
