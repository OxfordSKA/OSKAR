#
# src/math/test/CMakeLists.txt
#

set(name math_test)
set(${name}_SRC
    main.cpp
    Test_find_closest_match.cpp
    Test_linspace.cpp
    Test_matrix_multiply.cpp
    Test_random.cpp
    Test_cond2_2x2.cpp
)
if (OSKAR_USE_LAPACK)
    list(APPEND ${name}_SRC Test_fit_ellipse.cpp)
endif ()
add_executable(${name} ${${name}_SRC})
target_link_libraries(${name} oskar gtest)
if (OSKAR_USE_LAPACK)
    target_link_libraries(${name} ${OSKAR_LAPACK})
endif()
set_target_properties(${name} PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}")
add_test(math_test ${name})
