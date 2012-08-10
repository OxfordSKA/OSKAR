#
# src/apps/CMakeLists.txt
#

if (QT4_FOUND)
    include_directories(${QT_INCLUDE_DIR})

    # BINARY: oskar_settings_set
    set(name oskar_settings_set)
    qt_app(${name} ${name}_main.cpp)

    # BINARY: oskar_settings_get
    set(name oskar_settings_get)
    qt_app(${name} ${name}_main.cpp)

    # BINARY: oskar
    qt_app(oskar oskar_main.cpp oskar_MainWindow.cpp
        QT_MOC_SRC oskar_MainWindow.h
        EXTRA_LIBS oskar_widgets ${QT_QTGUI_LIBRARY})
endif ()

if (CUDA_FOUND)
    # BINARY: oskar_cuda_system_info
    set(name oskar_cuda_system_info)
    oskar_app(${name} ${name}_main.cpp)

    # BINARY: oskar_binary_file_query
    set(name oskar_binary_file_query)
    oskar_app(${name} ${name}_main.cpp)

    # BINARY: oskar_image_stats
    set(name oskar_image_stats)
    oskar_app(${name} ${name}_main.cpp)

    # BINARY: oskar_image_summary
    #set(name oskar_image_summary)
    #oskar_app(${name} ${name}_main.cpp)

    #if (CFITSIO_FOUND)
    #    # BINARY: oskar_fits_to_sky_model
    #    set(name oskar_fits_to_sky_model)
    #    oskar_app(${name} ${name}_main.cpp EXTRA_LIBS oskar_fits)
    #endif ()
endif ()

if (QT4_FOUND AND CUDA_FOUND)
    # Build oskar_app library
    add_subdirectory(lib)

    # BINARY: oskar_sim_interferometer
    set(name oskar_sim_interferometer)
    oskar_qt_app(${name} ${name}_main.cpp)

    # BINARY: oskar_sim_beam_pattern
    set(name oskar_sim_beam_pattern)
    oskar_qt_app(${name} ${name}_main.cpp)

    # BINARY: oskar_imager
    set(name oskar_imager)
    oskar_qt_app(${name} ${name}_main.cpp)

    # BINARY: oskar_visibilities_to_ms
    set(name oskar_visibilities_to_ms)
    oskar_qt_app(${name} ${name}_main.cpp)
endif ()

