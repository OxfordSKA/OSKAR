#
# src/apps/test/CMakeLists.txt
#

macro(copy_test_file NAME)
    set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
    set(DST_DIR ${CMAKE_CURRENT_BINARY_DIR})
    configure_file(${SRC_DIR}/${NAME} ${DST_DIR}/${NAME} @ONLY)
endmacro()

copy_test_file(run_example_3_3.sh)

