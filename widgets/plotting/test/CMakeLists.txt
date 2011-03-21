#
# src/widgets/plotting/test
#

set(plotting_test_SRC
    src/CppUnitMain.cpp
)

include_directories(${QT_QTCORE_INCLUDE_DIR})

add_executable(plottingTest ${plotting_test_SRC})

target_link_libraries(plottingTest
    oskar_widgets_plotting
    ${CPPUNIT_LIBRARIES}
    ${QT_QTCORE_LIBRARY}
)


# ==============================================================================

qt4_wrap_cpp(plotting_test_MOC QTestPlotWidget.h)
add_executable(qtest_plotWidget ${plotting_test_MOC})
target_link_libraries(qtest_plotWidget
    oskar_widgets_plotting
    ${QT_QTTEST_LIBRARY}
    ${QT_QTGUI_LIBRARY}
)

