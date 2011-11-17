#
# src/imaging/test/CMakeLists.txt
#

if (BUILD_OSKAR_WIDGETS)

    include_directories(${Qwt5_INCLUDE_DIR})
    include_directories(${QT_INCLUDE_DIR})
    include_directories(${QT_QTCORE_INCLUDE_DIR})
    include_directories(${QT_QTGUI_INCLUDE_DIR})

    # ==== Test genereating gridding convolution kernels.
    add_executable(conv_func_test src/conv_func_test_main.cpp)
    target_link_libraries(conv_func_test
        oskar_imaging
        oskar_widgets
        oskar
        ${QT_QTCORE_LIBRARY}
        ${QT_QTGUI_LIBRARY}
    )

    # ==== Gridding test. Needs fixing for new visibility structure.
    # add_executable(gridding_test src/gridding_test_main.cpp)
    # target_link_libraries(gridding_test
    #     oskar_imaging
    #     oskar_widgets
    #     oskar
    #     ${QT_QTCORE_LIBRARY}
    #     ${QT_QTGUI_LIBRARY}
    # )

    # ==== Imaging test.
    add_executable(imaging_test src/imaging_test_main.cpp)
    target_link_libraries(imaging_test
        oskar_imaging
        oskar_widgets
        oskar
        ${QT_QTCORE_LIBRARY}
        ${QT_QTGUI_LIBRARY}
    )

endif (BUILD_OSKAR_WIDGETS)



if (QT_QTTEST_FOUND)
endif (QT_QTTEST_FOUND)




#if (CPPUNIT_FOUND)
#    if (BUILD_OSKAR_WIDGETS)
#
#        include_directories(${Qwt5_INCLUDE_DIR})
#        include_directories(${QT_INCLUDE_DIR})
#        include_directories(${QT_QTCORE_INCLUDE_DIR})
#        include_directories(${QT_QTGUI_INCLUDE_DIR})
#
#        set(imaging_test_SRC
#            src/CppUnitMain.cpp
#            src/ImagingTest.cpp
#        )
#
#        add_executable(imagingTest ${imaging_test_SRC})
#        target_link_libraries(imagingTest
#            oskar_imaging
#            oskar_widgets
#            ${QT_QTCORE_LIBRARY}
#            ${CPPUNIT_LIBRARIES}
#        )
#        add_test(imagingTest imagingTest)
#
#    endif()
#endif()
#
#
#
#if (QT_QTTEST_FOUND)
#    if (BUILD_OSKAR_WIDGETS)
#
        # WProjConvFunc test.
#        qt4_wrap_cpp(WProjConvFunc_test_MOC QTest_WProjConvFunc.h)
#        add_executable(qtest_WProjConvFunc ${WProjConvFunc_test_MOC})
#        target_link_libraries(qtest_WProjConvFunc
#            oskar_imaging
#            oskar_widgets
#            ${QT_QTTEST_LIBRARY}
#            ${QT_QTGUI_LIBRARY}
#        )
#
        # ConvFunc test.
#        qt4_wrap_cpp(convFunc_test_MOC QTest_ConvFunc.h)
#        add_executable(qtest_ConvFunc ${convFunc_test_MOC})
#        target_link_libraries(qtest_ConvFunc
#            oskar_imaging
#            oskar_widgets
#            ${QT_QTTEST_LIBRARY}
#            ${QT_QTGUI_LIBRARY}
#        )
#
        # Gridding test.
#        qt4_wrap_cpp(gridding_test_MOC QTest_Gridding.h)
#        add_executable(qtest_Gridder ${gridding_test_MOC})
#        target_link_libraries(qtest_Gridder
#            oskar_imaging
#            oskar_widgets
#            ${QT_QTTEST_LIBRARY}
#            ${QT_QTGUI_LIBRARY}
#        )
#
        # Grid Image (standard -2d- kernel)
#        qt4_wrap_cpp(standard_grid_image_test_MOC QTest_StandardGridImage.h)
#        add_executable(qtest_standard_grid_image ${standard_grid_image_test_MOC})
#        target_link_libraries(qtest_standard_grid_image
#            oskar_imaging
#            oskar_widgets
#            ${QT_QTTEST_LIBRARY}
#            ${QT_QTGUI_LIBRARY}
#        )
#
#    endif (BUILD_OSKAR_WIDGETS)
#endif (QT_QTTEST_FOUND)
#
#
#if (CFITSIO_FOUND)
#    include_directories(${CFITSIO_INCLUDE_DIR})
#    include_directories(${CMAKE_CURRENT_SOURCE_DIR})
#    add_executable(write_uvfits src/write_uvfits.cpp)
#    target_link_libraries(write_uvfits
#        ${CFITSIO_LIBRARIES}
#        ${QT_QTCORE_LIBRARY}
#    )
#endif ()


