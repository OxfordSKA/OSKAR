#
# src/ms/test/CMakeLists.txt
#

# CppUnit
if (CPPUNIT_FOUND)

    set(ms_test_SRC
        src/CppUnitMain.cpp
        src/Test_MeasurementSet.cpp
    )

	include_directories(${CPPUNIT_INCLUDE_DIR})
    add_executable(ms_test ${ms_test_SRC})
    target_link_libraries(ms_test oskar_ms ${CPPUNIT_LIBRARIES})

    #add_test(ms_test ms_test)

endif ()
