#
# src/fits/test/CMakeLists.txt
#

set(name fits_test)
set(${name}_SRC
    Test_fits_write.cpp
)
add_executable(${name} ${${name}_SRC})
target_link_libraries(${name} oskar oskar_fits gtest_main)
set_target_properties(${name} PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}")
add_test(fits_test ${name})
