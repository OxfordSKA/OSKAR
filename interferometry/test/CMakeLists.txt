#
# src/interferometry/test/CMakeLists.txt
#

set(name interferometry_test)
set(${name}_SRC
    main.cpp
    Test_evaluate_baselines.cpp
    Test_station_coord_transforms.cpp
    Test_Visibilities.cpp
)
add_executable(${name} ${${name}_SRC})
include_directories(${CUDA_INCLUDE_DIRS})
target_link_libraries(${name} oskar gtest)
set_target_properties(${name} PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}")
add_test(interferometry_test ${name})
