#
# oskar/utility/test/CMakeLists.txt
#
set(name utility_test)
set(${name}_SRC
    main.cpp
    Test_crc.cpp
    Test_dir.cpp
    Test_getline.cpp
    Test_string_to_array.cpp
    Test_Timer.cpp
)

if (OpenCL_FOUND)
    list(APPEND ${name}_SRC
        Test_cl.cpp
    )
endif()

add_executable(${name} ${${name}_SRC})

target_link_libraries(${name} oskar gtest)
if (OpenCL_FOUND)
    target_link_libraries(${name} ${OpenCL_LIBRARIES})
endif()

set_target_properties(${name} PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}")
add_test(utility_test ${name})

set(name memory_test)
add_executable(${name} Test_get_memory_usage.cpp)
target_link_libraries(${name} oskar gtest_main)
set_target_properties(${name} PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}")
