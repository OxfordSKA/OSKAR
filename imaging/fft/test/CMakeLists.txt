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

