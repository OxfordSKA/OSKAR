#
# widgets/plotting/CMakeLists.txt
#

if (Qwt5_FOUND)
    include_directories(${Qwt5_INCLUDE_DIR})

    set(widgets_plotting_MOC
        oskar_ColourMap.h
        oskar_PlotWidget.h
    )
    set(widgets_plotting_MOC "${widgets_plotting_MOC}" PARENT_SCOPE)
    
    set(widgets_plotting_SRC
        src/oskar_ColourMap.cpp
        src/oskar_ImagePlot.cpp
        src/oskar_ImagePlotData.cpp
        src/oskar_PlotMagnifier.cpp
        src/oskar_PlotPanner.cpp
        src/oskar_PlotPicker.cpp
        src/oskar_PlotWidget.cpp
        src/oskar_PlotZoomer.cpp
    )
    set(widgets_plotting_SRC "${widgets_plotting_SRC}" PARENT_SCOPE)
    
    # Recurse into test subdirectory.
    add_subdirectory(test)

endif ()