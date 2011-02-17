#
# src/cuda/test
#


# CppUnit
# ==============================================================================
set(data_test_SRC
    src/CppUnitMain.cpp
    src/CudaVectorAddTest.cpp
    src/CudaBeamPatternTest.cpp
    src/CudaAntennaSignalTest.cpp
    src/CudaBeamformerMatrixVectorTest.cpp
    src/CudaImagerDftTest.cpp
)

include_directories()

add_executable(cudaTest ${data_test_SRC})

target_link_libraries(cudaTest
    oskar_cuda
    ${CPPUNIT_LIBRARIES}
)
