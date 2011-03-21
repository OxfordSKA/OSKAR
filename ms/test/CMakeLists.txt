#
# src/ms/test/CMakeLists.txt
#

# CppUnit
# ==============================================================================
set(ms_test_SRC
    src/CppUnitMain.cpp
    src/MsCreateTest.cpp
    src/MsAppendTest.cpp
)

if (CPPUNIT_FOUND)
    add_executable(msTest ${ms_test_SRC})
    target_link_libraries(msTest oskar_ms ${CPPUNIT_LIBRARIES})
endif()
