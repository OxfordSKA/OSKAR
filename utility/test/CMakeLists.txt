#
# src/utility/test/CMakeLists.txt
#

if (CPPUNIT_FOUND)

    set(utility_test_SRC
        src/CppUnitMain.cpp
        src/Test_binary_file.cpp
        src/Test_CudaInfo.cpp
        src/Test_getline.cpp
        src/Test_Mem.cpp
        src/Test_string_to_array.cpp
    )

    include_directories(${CPPUNIT_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})
    add_executable(utility_test ${utility_test_SRC})
    target_link_libraries(utility_test oskar ${CPPUNIT_LIBRARIES} ${CUDA_LIBRARIES})

    add_test(utility_test utility_test)

endif ()
