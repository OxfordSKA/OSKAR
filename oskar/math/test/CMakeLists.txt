#
# oskar/math/test/CMakeLists.txt
#

set(name math_test)
set(${name}_SRC
    main.cpp
    Test_find_closest_match.cpp
    Test_linspace.cpp
    Test_matrix_multiply.cpp
    Test_random.cpp
    Test_cond2_2x2.cpp
    Test_fit_ellipse.cpp
    Test_prefix_sum.cpp
)
add_executable(${name} ${${name}_SRC})
target_link_libraries(${name} oskar gtest)
set_target_properties(${name} PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}")
add_test(math_test ${name})
