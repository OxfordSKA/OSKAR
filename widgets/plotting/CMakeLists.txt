#
# widgets/plotting
#

include_directories(${Qwt5_INCLUDE_DIR} ${QT_INCLUDE_DIR}
    ${QT_QTCORE_INCLUDE_DIR} ${QT_QTGUI_INCLUDE_DIR})

set(widgets_plotting_MOC
    ColourMap.h
    PlotWidget.h
)

set(widgets_plotting_SRC
    src/ColourMap.cpp
    src/ImagePlot.cpp
    src/ImagePlotData.cpp
    src/PlotMagnifier.cpp
    src/PlotPanner.cpp
    src/PlotPicker.cpp
    src/PlotWidget.cpp
    src/PlotZoomer.cpp
)

qt4_wrap_cpp(widgets_plotting_SRC ${widgets_plotting_MOC})
add_library(oskar_widgets_plotting ${widgets_plotting_SRC})
target_link_libraries(oskar_widgets_plotting ${QT_QTCORE_LIBRARY}
    ${QT_QTGUI_LIBRARY} ${Qwt5_Qt4_LIBRARY})

install(TARGETS oskar_widgets_plotting LIBRARY DESTINATION ${OSKAR_LIB_INSTALL_DIR})

# Recurse into test subdirectory.
add_subdirectory(test)
