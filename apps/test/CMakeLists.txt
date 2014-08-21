#
# src/apps/test/CMakeLists.txt
#

macro(configure_test_file NAME)
    set(SRC ${CMAKE_CURRENT_SOURCE_DIR}/${NAME})
    set(DST ${CMAKE_CURRENT_BINARY_DIR}/${NAME})
    configure_file(${SRC} ${DST} @ONLY)
endmacro()

macro(copy_test_file NAME)
    include(oskar_copy_files)
    set(SRC ${CMAKE_CURRENT_SOURCE_DIR}/${NAME})
    set(DST ${CMAKE_CURRENT_BINARY_DIR}/${NAME})
    copy_file(${SRC} ${DST}) 
endmacro()

configure_test_file(run_example_add_noise.sh)
configure_test_file(run_example_beam_pattern_equatorial.sh)
configure_test_file(run_example_interferometry.sh)
configure_test_file(run_examples.sh)
configure_test_file(test_utility.sh)
