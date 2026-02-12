#
# oskar/sky/test/CMakeLists.txt
#

set(name sky_test)
set(${name}_SRC
    main.cpp
    Test_Sky_append.cpp
    Test_Sky_append_to_set.cpp
    Test_Sky_clear_source_flux.cpp
    Test_Sky_column_type.cpp
    Test_Sky_copy_contents.cpp
    Test_Sky_copy_source_data.cpp
    Test_Sky_copy.cpp
    Test_Sky_create_columns.cpp
    Test_Sky_evaluate_gaussian_source_parameters.cpp
    Test_Sky_evaluate_relative_directions.cpp
    Test_Sky_filter_by_flux.cpp
    Test_Sky_filter_by_radius.cpp
    Test_Sky_from_fits_file.cpp
    Test_Sky_horizon_clip.cpp
    Test_Sky_load.cpp
    Test_Sky_load_named_columns.cpp
    Test_Sky_resize.cpp
    Test_Sky_scale_flux_with_frequency.cpp
)
add_executable(${name} ${${name}_SRC})
target_link_libraries(${name} oskar cfitsio gtest)
add_test(NAME ${name} COMMAND ${name})
if (OpenMP_CXX_FOUND)
    target_link_libraries(${name} OpenMP::OpenMP_CXX)
endif()
