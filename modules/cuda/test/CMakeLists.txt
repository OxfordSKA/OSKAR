#
# src/modules/cuda/test
#

# CppUnit
# ==============================================================================
if (CPPUNIT_FOUND)
    set(modules_cuda_test_SRC
        src/CppUnitMain.cpp
        src/ModuleCudaCorrelatorTest.cpp
    )

    include_directories(${CPPUNIT_INCLUDE_DIR})
    add_executable(modulesCudaTest ${modules_cuda_test_SRC})
    target_link_libraries(modulesCudaTest
        oskar
        ${CPPUNIT_LIBRARIES})
endif()
