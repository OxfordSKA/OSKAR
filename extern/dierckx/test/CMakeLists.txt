#
# src/extern/dierckx/test/CMakeLists.txt
#

if (CPPUNIT_FOUND)

    include_directories(${CPPUNIT_INCLUDE_DIR})

    set(dierckx_test_SRC
        src/CppUnitMain.cpp
        src/Test_dierckx.cu
    )

    cuda_add_executable(dierckx_test ${dierckx_test_SRC})
    target_link_libraries(dierckx_test dierckx oskar ${CPPUNIT_LIBRARIES} ${CUDA_LIBRARIES})

    add_test(dierckx_test dierckx_test)

endif()

