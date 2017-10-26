#
# apps/test/CMakeLists.txt
#

macro(configure_test_file NAME)
    set(SRC ${CMAKE_CURRENT_SOURCE_DIR}/${NAME})
    set(DST ${CMAKE_CURRENT_BINARY_DIR}/${NAME})
    configure_file(${SRC} ${DST} @ONLY)
endmacro()

# Copy test scripts to the build tree.
file(GLOB test_scripts *.sh)
foreach (script ${test_scripts})
    get_filename_component(script_name ${script} NAME)
    configure_test_file(${script_name})
endforeach()
file(GLOB test_scripts *.py)
foreach (script ${test_scripts})
    file(COPY ${script} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endforeach()

# Copy test data to the build tree.
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
