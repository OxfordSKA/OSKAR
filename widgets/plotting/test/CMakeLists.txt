#
# src/widgets/plotting/test/CMakeLists.txt
#

include_directories(${Qwt5_INCLUDE_DIR})
include_directories(${QT_INCLUDE_DIR})
include_directories(${QT_QTCORE_INCLUDE_DIR})
include_directories(${QT_QTGUI_INCLUDE_DIR})

# ==== CppUnit unit testing.
if (CPPUNIT_FOUND)
    set(plotting_test_SRC
        src/CppUnitMain.cpp
    )
    include_directories(${CPPUNIT_INCLUDE_DIR})
    add_executable(plottingTest ${plotting_test_SRC})
    target_link_libraries(plottingTest
        oskar_widgets
        ${CPPUNIT_LIBRARIES}
        ${QT_QTCORE_LIBRARY})
endif ()

# ==== QtTest unit testing.
if (QT_QTTEST_FOUND)
    qt4_wrap_cpp(plotting_test_MOC QTestPlotWidget.h)
    add_executable(qtest_plotWidget ${plotting_test_MOC})
    target_link_libraries(qtest_plotWidget
        oskar_widgets
        ${QT_QTTEST_LIBRARY}
        ${QT_QTGUI_LIBRARY})
endif ()

