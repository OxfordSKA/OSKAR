#
# src/ms/test/CMakeLists.txt
#

# CppUnit
if (CPPUNIT_FOUND)

    set(ms_test_SRC
        src/CppUnitMain.cpp
        src/MeasurementSetTest.cpp
    )

    add_executable(msTest ${ms_test_SRC})
    target_link_libraries(msTest oskar_ms ${CPPUNIT_LIBRARIES})

endif ()
