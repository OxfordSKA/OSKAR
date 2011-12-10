#
# src/apps/test/CMakeLists.txt
#

if (CPPUNIT_FOUND)

    include_directories(${CPPUNIT_INCLUDE_DIR})
    include_directories(${QT_INCLUDE_DIR})

    set(apps_test_SRC
        src/CppUnitMain.cpp
        src/Test_load_stations.cpp
    )

    if(BUILD_OSKAR_MS)
        list(APPEND apps_test_SRC src/Test_write_ms.cpp)
    endif ()

    add_executable(apps_lib_test ${apps_test_SRC})

    set_target_properties(apps_lib_test PROPERTIES
       COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
       LINK_FLAGS "${OpenMP_CXX_FLAGS}")

    target_link_libraries(apps_lib_test
        oskar_apps
        ${CPPUNIT_LIBRARIES}
        ${QT_QTCORE_LIBRARIES}
    )

    add_test(apps_lib_test apps_lib_test)

endif()



if (BUILD_OSKAR_WIDGETS)
    include_directories(${Qwt5_INCLUDE_DIR})
    include_directories(${QT_QTGUI_INCLUDE_DIR})
    include_directories(${QT_INCLUDE_DIR})
    include_directories(${QT_QTCORE_INCLUDE_DIR})
    #add_executable(imager_dft_test src/imager_dft_test.cpp)
    #target_link_libraries(imager_dft_test
    #    oskar_apps
    #    oskar_widgets
    #    oskar
    #    ${QT_QTCORE_LIBRARY}
    #    ${QT_QTGUI_LIBRARY}
    #)
endif ()
