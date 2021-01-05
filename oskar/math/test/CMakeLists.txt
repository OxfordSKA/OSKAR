#
# oskar/math/test/CMakeLists.txt
#

set(name math_test)
set(${name}_SRC
    main.cpp
    Test_dft.cpp
    Test_find_closest_match.cpp
    Test_legendre.cpp
    Test_linspace.cpp
    Test_matrix_multiply.cpp
    Test_random.cpp
    Test_cond2_2x2.cpp
    Test_fit_ellipse.cpp
    Test_prefix_sum.cpp
    #Test_spherical_harmonics.cpp
)
add_executable(${name} ${${name}_SRC})
target_link_libraries(${name} oskar gtest)
add_test(math_test ${name})
