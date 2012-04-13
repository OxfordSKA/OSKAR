#
# src/apps/CMakeLists.txt
#

# Recurse into lib subdirectory.
add_subdirectory(lib)

include(oskar_build_macros)

# BINARY: oskar_sim_interferometer
set(name oskar_sim_interferometer)
oskar_app(${name} ${name}_main.cpp)

# BINARY: oskar_sim_beam_pattern
set(name oskar_sim_beam_pattern)
oskar_app(${name} ${name}_main.cpp)

# BINARY: oskar_imager
set(name oskar_imager)
oskar_app(${name} ${name}_main.cpp)

# BINARY: oskar
if (BUILD_OSKAR_WIDGETS)
    oskar_app(oskar oskar_gui_main.cpp oskar_MainWindow.cpp 
        QT_MOC_SRC oskar_MainWindow.h
        EXTRA_LIBS oskar_widgets ${QT_QTGUI_LIBRARY})
endif ()
