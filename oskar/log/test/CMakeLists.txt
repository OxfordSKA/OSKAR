#
# oskar/log/test/CMakeLists.txt
#

set(name log_test)
set(${name}_SRC
    main.cpp
    Test_Log.cpp
)
add_executable(${name} ${${name}_SRC})
target_link_libraries(${name} gtest oskar)
add_test(log_test ${name})
