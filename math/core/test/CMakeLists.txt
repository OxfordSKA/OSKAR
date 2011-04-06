#
# /src/math/core/test
#

# CppUnit
# ==============================================================================
set(math_core_test_SRC
    src/CppUnitMain.cpp
    src/SphericalPositionsTest.cpp
    src/Matrix3Test.cpp
    src/Matrix4Test.cpp
)

if (CPPUNIT_FOUND)
	include_directories(${CPPUNIT_INCLUDE_DIR})
    add_executable(mathCoreTest ${math_core_test_SRC})
    target_link_libraries(mathCoreTest ${CPPUNIT_LIBRARIES})
endif()

