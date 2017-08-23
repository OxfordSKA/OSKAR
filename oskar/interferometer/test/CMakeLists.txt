#
# oskar/simulator/test/CMakeLists.txt
#

set(name jones_test)
set(${name}_SRC
    main.cpp
    Test_Jones.cpp
    Test_evaluate_jones_K.cpp
)
add_executable(${name} ${${name}_SRC})
target_link_libraries(${name} oskar gtest)
set_target_properties(${name} PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}")
add_test(jones_test ${name})

