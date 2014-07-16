#
# src/convert/test/CMakeLists.txt
#


include_directories(${CUDA_INCLUDE_DIRS})

set(name convert_test)
set(${name}_SRC
    main.cpp
    Test_coordinate_conversions.cpp
)
cuda_add_executable(${name} ${${name}_SRC})
target_link_libraries(${name} oskar gtest ${CUDA_LIBRARIES})
add_test(${name} ${name})

