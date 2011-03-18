#
# src/ms/test
#


# CppUnit
# ==============================================================================
set(ms_test_SRC
    src/CppUnitMain.cpp
    src/MsCreateTest.cpp
)

include_directories()

add_executable(msTest ${ms_test_SRC})

target_link_libraries(msTest
    oskar_ms
    ${CPPUNIT_LIBRARIES}
)
