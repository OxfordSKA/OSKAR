#
# oskar/telescope/test/CMakeLists.txt
#

set(name telescope_test)
set(${name}_SRC
    main.cpp
    Test_evaluate_baselines.cpp
    Test_station_coord_transforms.cpp
    Test_telescope_model_load_save.cpp
)
add_executable(${name} ${${name}_SRC})
target_link_libraries(${name} oskar gtest)
add_test(${name} ${name})
