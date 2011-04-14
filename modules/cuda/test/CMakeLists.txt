#
# src/modules/cuda/test
#

# CppUnit
# ==============================================================================
set(modules_cuda_test_SRC
    src/CppUnitMain.cpp
    src/ModuleCudaCorrelatorTest.cpp
)

if (CPPUNIT_FOUND)
    #message("=============== ${CPPUNIT_INCLUDE_DIR}")
    include_directories(${CPPUNIT_INCLUDE_DIR})
    add_executable(modulesCudaTest ${modules_cuda_test_SRC})
    target_link_libraries(modulesCudaTest oskar_modules ${CPPUNIT_LIBRARIES})
endif()
