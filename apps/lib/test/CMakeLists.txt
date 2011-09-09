#
# src/apps/test/CMakeLists.txt
#

if (CPPUNIT_FOUND)

    include_directories(${CPPUNIT_INCLUDE_DIR})
    include_directories(${QT_INCLUDE_DIR})

    set(apps_test_SRC
        src/CppUnitMain.cpp
        src/LoadStationsTest.cpp
        src/LoadTelescopeTest.cpp
        src/VisDataTest.cpp
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


# TEST BINARY: write_ms_test
add_executable(write_ms_test src/WriteMS_Test.cpp)
target_link_libraries(write_ms_test
    oskar
    oskar_apps
    ${QT_QTCORE_LIBRARIES}
)


if (BUILD_OSKAR_WIDGETS)
    include_directories(${Qwt5_INCLUDE_DIR})
    include_directories(${QT_QTGUI_INCLUDE_DIR})
    include_directories(${QT_INCLUDE_DIR})
    include_directories(${QT_QTCORE_INCLUDE_DIR})
    add_executable(imager_dft_test src/imager_dft_test.cpp)
    target_link_libraries(imager_dft_test
        oskar_apps
        oskar_widgets
        oskar
        ${QT_QTCORE_LIBRARY}
        ${QT_QTGUI_LIBRARY}
    )
endif ()
