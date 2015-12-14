#
# src/ms/test/CMakeLists.txt
#

set(name ms_test)

set(${name}_SRC
    Test_MeasurementSet.cpp
)
add_executable(${name} ${${name}_SRC})
target_link_libraries(${name} oskar oskar_ms gtest_main)
set_target_properties(${name} PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}")
#add_test(ms_test ${name})
