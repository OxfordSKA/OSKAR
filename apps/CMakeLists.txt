#
# src/apps/CMakeLists.txt
#

if (QT4_FOUND AND CUDA_FOUND)

    include_directories(${QT_INCLUDE_DIR})

    # Build oskar_app library
    add_subdirectory(lib)

    # BINARY: oskar_binary_file_query
    set(name oskar_binary_file_query)
    oskar_app(${name} ${name}_main.cpp)

    # BINARY: oskar_sim_interferometer
    set(name oskar_sim_interferometer)
    oskar_app(${name} ${name}_main.cpp)

    # BINARY: oskar_sim_beam_pattern
    set(name oskar_sim_beam_pattern)
    oskar_app(${name} ${name}_main.cpp)

    # BINARY: oskar_imager
    set(name oskar_imager)
    oskar_app(${name} ${name}_main.cpp)

    # BINARY: oskar_settings_set
    set(name oskar_settings_set)
    oskar_app(${name} ${name}_main.cpp)

    # BINARY: oskar_settings_get
    set(name oskar_settings_get)
    oskar_app(${name} ${name}_main.cpp)

    # BINARY: oskar_visibilities_to_ms
    set(name oskar_visibilities_to_ms)
    oskar_app(${name} ${name}_main.cpp)

    # BINARY: oskar
    oskar_app(oskar oskar_main.cpp oskar_MainWindow.cpp
        QT_MOC_SRC oskar_MainWindow.h
        EXTRA_LIBS oskar_widgets ${QT_QTGUI_LIBRARY})

endif ()
