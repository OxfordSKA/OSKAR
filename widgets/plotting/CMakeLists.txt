#
# widgets/plotting/CMakeLists.txt
#

set(widgets_plotting_MOC
    ColourMap.h
    PlotWidget.h
)
set(widgets_plotting_MOC "${widgets_plotting_MOC}" PARENT_SCOPE)

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
set(widgets_plotting_SRC "${widgets_plotting_SRC}" PARENT_SCOPE)

# ==== Recurse into test subdirectory.
add_subdirectory(test)
