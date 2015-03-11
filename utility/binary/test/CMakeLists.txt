#
# src/utility/binary/test/CMakeLists.txt
#

set(name binary_test)
add_executable(${name} Test_binary.cpp)
target_link_libraries(${name} oskar_binary)
add_test(${name} ${name})

set(name test_vis_read_write)
add_executable(${name} Test_binary_vis_read_write.c)
target_link_libraries(${name} oskar_binary)
