#
# src/utility/test/CMakeLists.txt
#


if (CPPUNIT_FOUND)

    set(utility_test_SRC
        src/CppUnitMain.cpp
        src/LoadCoordinatesTest.cpp
    )

    include_directories(${CPPUNIT_INCLUDE_DIR})
    include_directories(${QT_INCLUDE_DIR})
    add_executable(utilityTest ${utility_test_SRC})
    target_link_libraries(utilityTest
        oskar
        ${CPPUNIT_LIBRARIES}
        ${QT_QTCORE_LIBRARIES})

    add_test(utilityTest utilityTest)

endif ()
