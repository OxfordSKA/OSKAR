#
# doc/CMakeLists.txt
#

set(DOXYFILE_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(DOXYFILE_LATEX ON)
include(oskar_copy_files)
include(UseDoxygen)
#if(DOXYGEN_FOUND) # Needed for older versions of doxygen?
#    copy_file(${CMAKE_CURRENT_SOURCE_DIR}/version.txt
#        ${CMAKE_CURRENT_BINARY_DIR}/version.txt)
#    copy_file(${CMAKE_CURRENT_SOURCE_DIR}/oskar_latex.sty
#        ${CMAKE_CURRENT_BINARY_DIR}/oskar_latex.sty)
#endif()

#add_doxygen_target(doc Doxyfile)
