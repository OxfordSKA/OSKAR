#
# src/apps/test/CMakeLists.txt
#

if (CPPUNIT_FOUND)

    include_directories(${CPPUNIT_INCLUDE_DIR})

    set(apps_test_SRC
        src/CppUnitMain.cpp
        src/Test_telescope_model_load_save.cpp
        src/Test_Settings.cpp
    )

    if (CASACORE_FOUND)
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
