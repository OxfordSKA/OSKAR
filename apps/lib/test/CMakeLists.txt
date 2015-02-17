#
# src/apps/test/CMakeLists.txt
#

set(name apps_lib_test)
set(${name}_SRC
    Test_telescope_model_load_save.cpp
    Test_sim_interferometer.cpp
)
if (CASACORE_FOUND)
    list(APPEND ${name}_SRC Test_write_ms.cpp)
endif ()
add_executable(${name} ${${name}_SRC})
target_link_libraries(${name} oskar_apps gtest_main ${QT_QTCORE_LIBRARY})
set_target_properties(${name} PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}")
add_test(apps_lib_test ${name})

# Binary to test the option parser.
add_executable(test_OptionParser Test_OptionParser.cpp)
target_link_libraries(test_OptionParser oskar_apps)

set(name test_beam_pattern_coordinates)
add_executable(${name}
    Test_beam_pattern_coordinates.cpp)
target_link_libraries(${name} oskar gtest_main ${QT_QTCORE_LIBRARY})
