#
# oskar/utility/test/CMakeLists.txt
#

set(name utility_test)
set(${name}_SRC
    main.cpp
    Test_crc.cpp
    Test_dir.cpp
    Test_getline.cpp
    Test_string_to_array.cpp
    Test_Thread.cpp
    Test_Timer.cpp
)
add_executable(${name} ${${name}_SRC})
target_link_libraries(${name} oskar gtest)
add_test(utility_test ${name})
