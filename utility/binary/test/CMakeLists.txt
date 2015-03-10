#
# src/utility/binary/test/CMakeLists.txt
#

set(name binary_test)
set(${name}_SRC
    Test_binary.cpp
)

add_executable(${name} ${${name}_SRC})
target_link_libraries(${name} oskar_binary gtest_main)
add_test(binary_test ${name})

