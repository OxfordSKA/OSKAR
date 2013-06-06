#
# src/apps/CMakeLists.txt
#

# === Include build macros used for apps and mex functions.
include(${OSKAR_SOURCE_DIR}/cmake/oskar_build_macros.cmake)

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
        EXTRA_LIBS oskar_widgets ${QT_QTGUI_LIBRARY} ${QT_QTNETWORK_LIBRARY})
endif ()

if (CUDA_FOUND)
    macro(declare_oskar_apps)
        foreach (name ${ARGN})
            oskar_app(${name} ${name}_main.cpp)
        endforeach()
    endmacro(declare_oskar_apps)

    declare_oskar_apps(
        oskar_cuda_system_info
        oskar_binary_file_query
        oskar_image_stats
        oskar_image_summary
        oskar_visibilities_add
        oskar_visibilities_summary
        #oskar_rebin_sky
    )

    if (CFITSIO_FOUND)
        set(name oskar_fits_image_to_sky_model)
        oskar_app(${name} ${name}_main.cpp EXTRA_LIBS oskar_fits)
    endif ()
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

    # BINARY: oskar_visibilities_add_noise
    set(name oskar_visibilities_add_noise)
    oskar_qt_app(${name} ${name}_main.cpp)

    set(name oskar_visibility_stats)
    oskar_qt_app(${name} ${name}_main.cpp)

    # BINARY: osakr_sim_tec_screen
    #set(name oskar_sim_tec_screen)
    #oskar_app(${name} ${name}_main.cpp
    #    EXTRA_LIBS oskar_apps
    #    NO_INSTALL)

    # BINARY: osakr_evaluate_pierce_points
    #set(name oskar_evaluate_pierce_points)
    #oskar_app(${name} ${name}_main.cpp
    #    EXTRA_LIBS oskar_apps
    #    NO_INSTALL)
endif ()


#if (CUDA_FOUND AND PNG_FOUND)
#    include_directories(${PNG_INCLUDE_DIRS})
#
#    set(name oskar_image_to_png)
#    oskar_app(${name} ${name}_main.cpp
#        EXTRA_LIBS ${PNG_LIBRARY}
#        NO_INSTALL)
#endif ()

#if (CUDA_FOUND AND PNG_FOUND AND QT4_FOUND)
#    include_directories(${PNG_INCLUDE_DIRS})
#    set(name oskar_ionosphere_to_kml)
#    oskar_app(${name} ${name}_main.cpp
#        EXTRA_LIBS ${PNG_LIBRARY} oskar_apps
#        NO_INSTALL
#    )
#endif ()
