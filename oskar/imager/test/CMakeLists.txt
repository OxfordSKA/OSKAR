#
# oskar/imager/test/CMakeLists.txt
#

set(name imager_test)
include_directories(${CUDA_INCLUDE_DIRS})
set(${name}_SRC
    main.cpp
    Test_fits_write.cpp
    Test_grid_sum.cpp
)
add_executable(${name} ${${name}_SRC})
target_link_libraries(${name} oskar gtest)
set_target_properties(${name} PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}")
add_test(imager_test ${name})
