#
# src/utility/test/CMakeLists.txt
#


if (CPPUNIT_FOUND)

    set(utility_test_SRC
        src/CppUnitMain.cu
        src/GetlineTest.cpp
        src/oskar_Mem_test.cpp
        src/oskar_StringToArrayTest.cpp
    )

    include_directories(${CPPUNIT_INCLUDE_DIR})
    cuda_add_executable(utilityTest ${utility_test_SRC})
    target_link_libraries(utilityTest
        oskar
        ${CPPUNIT_LIBRARIES})

    add_test(utilityTest utilityTest)

endif ()
