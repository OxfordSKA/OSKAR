#
# src/cuda/test/CMakeLists.txt
#

if (CPPUNIT_FOUND)

    set(cuda_test_SRC
        src/CppUnitMain.cpp
        src/CudaVectorAddTest.cpp
        src/CudaThrustTest.cu
        src/CudaBeamPatternTest.cpp
        #src/CudaHierarchicalBeamPatternTest.cpp
        #src/CudaBeamPatternGaussianTest.cpp
        #src/CudaBeamPatternSineTest.cpp
        #src/CudaAntennaSignalTest.cpp
        #src/CudaBeamformerMatrixVectorTest.cu
        #src/CudaImagerDftTest.cpp
        #src/CudaEquatorialToHorizontalTest.cpp
    )

    # === Macros to build the oskar-lib cuda test module.
    set(CUDA_GENERATED_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/oskar_cuda_test.dir/src)

    include_directories(${CPPUNIT_INCLUDE_DIR})
    cuda_add_executable(cudaTest ${cuda_test_SRC})
    target_link_libraries(cudaTest oskar ${CPPUNIT_LIBRARIES})

endif()
