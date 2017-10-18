#
# src/utility/binary/test/CMakeLists.txt
#

project(oskar_binary_test C)

set(name binary_test)
set(${name}_SRC Test_binary.c)
add_executable(${name} ${${name}_SRC})
target_link_libraries(${name} oskar_binary)

# Add the oskar_binary_test target to the 'tests' target.
if (NOT TARGET tests)
    add_custom_target(tests)
endif()
add_dependencies(tests ${name})

add_test(binary_test ${name})

set(name test_binary_vis_read_write)
add_executable(${name} Test_binary_vis_read_write.c)
target_link_libraries(${name} oskar_binary)
