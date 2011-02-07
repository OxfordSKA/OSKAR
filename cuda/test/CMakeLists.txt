#
# src/cuda/test
#


# CppUnit
# ==============================================================================
set(data_test_SRC
    src/CppUnitMain.cpp
    src/VectorAddTest.cpp
    src/BeamPatternTest.cpp
    src/BeamPatternRandomTest.cpp
    src/AntennaSignalTest.cpp
    src/BeamformerMatrixVectorTest.cpp
)

include_directories()

add_executable(cudaTest ${data_test_SRC})

target_link_libraries(cudaTest
    cuda
    ${CPPUNIT_LIBRARIES}
)
