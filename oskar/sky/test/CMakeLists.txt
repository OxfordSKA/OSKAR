#
# oskar/sky/test/CMakeLists.txt
#

set(name sky_test)
set(${name}_SRC
    main.cpp
    Test_Sky.cpp
)
add_executable(${name} ${${name}_SRC})
target_link_libraries(${name} oskar gtest)
add_test(sky_test ${name})
