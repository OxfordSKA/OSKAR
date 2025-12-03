#
# oskar/interferometer/test/CMakeLists.txt
#

set(name jones_test)
set(${name}_SRC
    main.cpp
    Test_apply_cable_length_errors.cpp
    Test_Jones.cpp
    Test_evaluate_jones_K.cpp
)

if (HDF5_FOUND)
    list(APPEND ${name}_SRC
        Test_apply_station_gains.cpp
    )
endif()

add_executable(${name} ${${name}_SRC})
target_link_libraries(${name} oskar gtest)
add_test(NAME ${name} COMMAND ${name})
