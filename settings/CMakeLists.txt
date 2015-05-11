#
# src/settings/CMakeLists.txt
#

#
# Known issues:
# 1. Log dependency of settings file I/O library
# 2. log_settings function in utility (this is really an app function!)
#

add_subdirectory(xml)

#add_subdirectory(utility)
#add_subdirectory(types)
#add_subdirectory(list)

#
# Note this module is currently being built as a STATIC library. This may need
# review...
#
set(lib_type STATIC)

# Settings widgets library (Qt4 based)
# -----------------------------------------------------------------------------
# Dependencies:
#   Qt4 : widgets / model
# -----------------------------------------------------------------------------
if (QT4_FOUND)
    include_directories(${QT_INCLUDE_DIR})

    set(name oskar_settings_widgets)
    set(dir_ widgets)
    set(${name}_MOC_SRC
        ${dir_}/oskar_DoubleSpinBox.h
        ${dir_}/oskar_SettingsDelegate.h
        ${dir_}/oskar_SettingsModel.h
        #${dir_}/oskar_SettingsModelApps.h
        ${dir_}/oskar_SettingsModelXML.h
        ${dir_}/oskar_SettingsView.h
    )
    set(dir_ widgets/src)
    set(${name}_SRC
        ${dir_}/oskar_DoubleSpinBox.cpp
        ${dir_}/oskar_SettingsDelegate.cpp
        ${dir_}/oskar_SettingsItem.cpp
        ${dir_}/oskar_SettingsModel.cpp
        #${dir_}/oskar_SettingsModelApps.cpp
        ${dir_}/oskar_SettingsModelXML.cpp
        ${dir_}/oskar_SettingsView.cpp
    )
    qt4_wrap_cpp(${name}_SRC ${${name}_MOC_SRC})
    add_library(${name} ${lib_type} ${${name}_SRC} ${${name}_MOC_SRC})
    target_link_libraries(${name}
        ${QT_QTCORE_LIBRARY}
        ${QT_QTGUI_LIBRARY}
    )
    #======================== XML settings unit test
    include_directories(${CMAKE_CURRENT_BINARY_DIR}/xml)
    add_executable(test_oskar_settings_widgets
        widgets/test/Test_oskar_SettingsModelXML.cpp
    )
    target_link_libraries(test_oskar_settings_widgets
        gtest_main oskar_settings_widgets)
    #=========================
endif (QT4_FOUND)


# Settings structure handling library
# ----------------------------------------------------------------------------
set(name oskar_settings_struct)
set(dir_ struct/src)
set(${name}_SRC
    ${dir_}/oskar_settings_init.c
    ${dir_}/oskar_settings_free.c
    ${dir_}/oskar_Settings.cpp
)
add_library(${name} ${lib_type} ${${name}_SRC})

# Settings logging library
# ----------------------------------------------------------------------------
set(name oskar_settings_log)
set(${name}_SRC
    oskar_settings_log.c
)
add_library(${name} ${lib_type} ${${name}_SRC})

# Settings file library (Qt4 ini file loaded based)
# -----------------------------------------------------------------------------
# Dependencies:
#   Qt4                 : ini file parsing
#   cuda                : oskar_settings_load_simulator
#
# Changes from oskar library version
#   1.  Removal of use of oskar_log This adds a library dependency that should
#       be avoided. Use error codes instead! (observation settings)
# -----------------------------------------------------------------------------
if (QT4_FOUND)
    include_directories(${QT_INCLUDE_DIR})

    set(name oskar_settings_load)
    set(dir_ load/src)
    set(${name}_C_SRC
        ${dir_}/oskar_settings_convert_date_time_to_mjd.c
        ${dir_}/oskar_settings_file_exists.c
        ${dir_}/oskar_settings_getline.c
        ${dir_}/oskar_settings_load_tid_parameter_file.c
        ${dir_}/oskar_settings_load.c
        ${dir_}/oskar_settings_string_to_array.c
    )

    set(${name}_CXX_SRC
        ${dir_}/oskar_settings_load_beam_pattern.cpp
        ${dir_}/oskar_settings_load_element_fit.cpp
        ${dir_}/oskar_settings_load_image.cpp
        ${dir_}/oskar_settings_load_interferometer.cpp
        ${dir_}/oskar_settings_load_ionosphere.cpp
        ${dir_}/oskar_settings_load_observation.cpp
        ${dir_}/oskar_settings_load_simulator.cpp
        ${dir_}/oskar_settings_load_sky.cpp
        ${dir_}/oskar_settings_load_telescope.cpp
    )

    add_library(${name} ${lib_type} ${${name}_C_SRC} ${${name}_CXX_SRC})
    target_link_libraries(${name}
        oskar_settings_struct
        ${QT_QTCORE_LIBRARY}
        ${CUDA_LIBRARIES})
endif (QT4_FOUND)
