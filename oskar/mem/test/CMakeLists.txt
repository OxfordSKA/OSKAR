#
# oskar/mem/test/CMakeLists.txt
#
set(name mem_test)
set(${name}_SRC
    main.cpp
    Test_Mem_binary.cpp
    Test_Mem_add.cpp
    Test_Mem_append.cpp
    Test_Mem_ascii.cpp
    Test_Mem_copy.cpp
    Test_Mem_different.cpp
    Test_Mem_normalise.cpp
    Test_Mem_realloc.cpp
    Test_Mem_scale_real.cpp
    Test_Mem_set_value_real.cpp
    Test_Mem_stats.cpp
    Test_Mem_to_type.cpp
    Test_Mem_type_check.cpp
    Test_Mem_random.cpp
)
add_executable(${name} ${${name}_SRC})
target_link_libraries(${name} oskar gtest)

add_test(${name} ${name})
