#
# oskar/ms/test/CMakeLists.txt
#

set(name ms_test)
set(${name}_SRC
    Test_MeasurementSet.cpp
)
add_executable(${name} ${${name}_SRC})
target_link_libraries(${name} oskar_ms gtest_main)
add_test(NAME ${name} COMMAND ${name})
