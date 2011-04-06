#
# src/cuda/test
#

# CppUnit
# ==============================================================================
set(data_test_SRC
    src/CppUnitMain.cpp
    src/CudaVectorAddTest.cpp
    #src/CudaBeamPatternTest.cpp
    src/CudaHierarchicalBeamPatternTest.cpp
    src/CudaBeamPatternGaussianTest.cpp
    src/CudaBeamPatternSineTest.cpp
    src/CudaAntennaSignalTest.cpp
    src/CudaBeamformerMatrixVectorTest.cpp
    src/CudaImagerDftTest.cpp
    src/CudaEquatorialToHorizontalTest.cpp
)

if (CPPUNIT_FOUND)
	#message("=============== ${CPPUNIT_INCLUDE_DIR}")
	include_directories(${CPPUNIT_INCLUDE_DIR})
    add_executable(cudaTest ${data_test_SRC})
    target_link_libraries(cudaTest oskar_cuda ${CPPUNIT_LIBRARIES})
endif()
