#
# src/cuda/test
#

# CppUnit
set(cuda_test_SRC
    src/CppUnitMain.cpp
    src/CudaVectorAddTest.cpp
    src/CudaMatrixManipTest.cu
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
    # === Macros to build the oskar-lib cuda test module.
    set(CUDA_GENERATED_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/oskar_cuda_test.dir/src)

    #message("=============== ${CPPUNIT_INCLUDE_DIR}")
    include_directories(${CUDA_INCLUDE_DIRS} ${CPPUNIT_INCLUDE_DIR})
    cuda_add_executable(cudaTest ${cuda_test_SRC})
    target_link_libraries(cudaTest oskar_cuda ${CPPUNIT_LIBRARIES})
endif()
