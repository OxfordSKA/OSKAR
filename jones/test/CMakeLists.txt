#
# src/jones/test/CMakeLists.txt
#

set(name jones_test)
include_directories(${CUDA_INCLUDE_DIRS})
set(${name}_SRC
    main.cpp
    Test_Jones.cpp
    Test_evaluate_jones_K.cpp
)
cuda_add_executable(${name} ${${name}_SRC})
target_link_libraries(${name} oskar gtest ${CUDA_LIBRARIES})
set_target_properties(${name} PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}")
add_test(jones_test ${name})

