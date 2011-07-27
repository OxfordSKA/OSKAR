#
# src/sky/test/CMakeLists.txt
#

# CppUnit
set(sky_test_SRC
    src/CppUnitMain.cpp
    src/SkyTest.cpp
    src/EquatorialToHorizontalTest.cu
)

cuda_add_executable(skyTest ${sky_test_SRC})
target_link_libraries(skyTest
    oskar_sky
    ${CPPUNIT_LIBRARIES}
)
