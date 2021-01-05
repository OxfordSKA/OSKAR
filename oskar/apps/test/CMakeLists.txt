#
# oskar/apps/test/CMakeLists.txt
#

set(name apps_test)
set(${name}_SRC
    create_sky_model.cpp
    create_telescope_model.cpp
    main.cpp
    Test_BeamPattern.cpp
    Test_Imager.cpp
    Test_Interferometer.cpp
    Test_Sky.cpp
    Test_Telescope.cpp
)
add_executable(${name} ${${name}_SRC})
target_link_libraries(${name} oskar oskar_apps gtest)
add_test(apps_test ${name})
