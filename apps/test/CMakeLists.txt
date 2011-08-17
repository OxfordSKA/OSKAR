#
# src/apps/test/CMakeLists.txt
#

if (CPPUNIT_FOUND)

    include_directories(${CPPUNIT_INCLUDE_DIR})
    include_directories(${QT_INCLUDE_DIR})

    set(apps_test_SRC
        src/CppUnitMain.cpp
        src/LoadStationsTest.cpp
    )

    add_executable(appsTest ${apps_test_SRC})

    set_target_properties(appsTest PROPERTIES
       COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
       LINK_FLAGS "${OpenMP_CXX_FLAGS}")

    target_link_libraries(appsTest
        oskar_apps
        ${CPPUNIT_LIBRARIES}
        ${QT_QTCORE_LIBRARIES}
    )

    add_test(appsTest appsTest)

endif()

