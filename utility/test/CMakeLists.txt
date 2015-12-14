#
# src/utility/test/CMakeLists.txt
#
set(name utility_test)
include_directories(${CUDA_INCLUDE_DIRS})
set(${name}_SRC
    main.cpp
    Test_mem_binary.cpp
    Test_crc.cpp
    Test_getline.cpp
    Test_Mem_add.cpp
    Test_Mem_append.cpp
    Test_Mem_ascii.cpp
    Test_Mem_copy.cpp
    Test_Mem_different.cpp
    Test_Mem_realloc.cpp
    Test_Mem_scale_real.cpp
    Test_Mem_set_value_real.cpp
    Test_Mem_stats.cpp
    Test_Mem_to_type.cpp
    Test_Mem_type_check.cpp
    Test_Mem_random.cpp
    Test_string_to_array.cpp
    Test_Timer.cpp
)

add_executable(${name} ${${name}_SRC})
target_link_libraries(${name} oskar gtest)
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
