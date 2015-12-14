#
# src/apps/test/CMakeLists.txt
#

macro(copy_test_file NAME)
    include(oskar_copy_files)
    set(SRC ${CMAKE_CURRENT_SOURCE_DIR}/${NAME})
    set(DST ${CMAKE_CURRENT_BINARY_DIR}/${NAME})
    copy_file(${SRC} ${DST})
endmacro()

macro(configure_test_file NAME)
    set(SRC ${CMAKE_CURRENT_SOURCE_DIR}/${NAME})
    set(DST ${CMAKE_CURRENT_BINARY_DIR}/${NAME})
    configure_file(${SRC} ${DST} @ONLY)
endmacro()

file(GLOB test_scripts *.sh)
foreach (script ${test_scripts})
    get_filename_component(script_name ${script} NAME)
    configure_test_file(${script_name})
endforeach()

copy_test_file(fits_image_stats.py)
copy_test_file(plot_beam_text_file.py)
