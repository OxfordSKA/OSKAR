#
# widgets/plotting/CMakeLists.txt
#
#===============================================================================

include_directories(
    ${Qwt5_INCLUDE_DIR}
    ${QT_INCLUDE_DIR}
    ${QT_QTCORE_INCLUDE_DIR}
    ${QT_QTGUI_INCLUDE_DIR}
)

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

# === Macros to build the oskar-lib plotting widget sub-module.
qt4_wrap_cpp(widgets_plotting_SRC ${widgets_plotting_MOC})
add_library(oskar_widgets_plotting SHARED ${widgets_plotting_SRC})
target_link_libraries(oskar_widgets_plotting
    ${QT_QTCORE_LIBRARY}
    ${QT_QTGUI_LIBRARY}
    ${Qwt5_Qt4_LIBRARY}
)
set_target_properties(oskar_widgets_plotting PROPERTIES
    INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib
    INSTALL_RPATH_USE_LINK_PATH TRUE)


# === Install target for plotting widgets sub-module.
set_target_properties(oskar_widgets_plotting PROPERTIES
    SOVERSION ${oskar-lib_SOVERSION})
install(TARGETS oskar_widgets_plotting DESTINATION ${OSKAR_LIB_INSTALL_DIR})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DESTINATION ${OSKAR_INCLUDE_INSTALL_DIR}
    FILES_MATCHING PATTERN "*.h"
    PATTERN "src" EXCLUDE
    PATTERN "test" EXCLUDE
    PATTERN ".svn" EXCLUDE)



# === Recurse into test subdirectory.
add_subdirectory(test)
