#
# apps/gui/widgets/CMakeLists.txt
#

if (QT4_FOUND)
    include_directories(${QT_INCLUDE_DIR})

    set(name oskar_settings_widgets)
    set(${name}_MOC_SRC
        oskar_DoubleSpinBox.h
        oskar_SettingsDelegate_new.hpp
        oskar_SettingsModel_new.hpp
        oskar_SettingsView.h
    )
    set(${name}_SRC
        src/oskar_DoubleSpinBox.cpp
        src/oskar_SettingsDelegate_new.cpp
        src/oskar_SettingsModel_new.cpp
        src/oskar_SettingsView.cpp
    )
    qt4_wrap_cpp(${name}_SRC ${${name}_MOC_SRC})
    add_library(${name} STATIC ${${name}_SRC} ${${name}_MOC_SRC})
    target_link_libraries(${name}
        ${QT_QTCORE_LIBRARY}
        ${QT_QTGUI_LIBRARY}
    )
    #======================== XML settings unit test
    include_directories(${CMAKE_CURRENT_BINARY_DIR}/xml)
    add_executable(test_oskar_settings_widgets
        test/Test_oskar_SettingsModel_new.cpp
    )
    target_link_libraries(test_oskar_settings_widgets
        gtest_main
        oskar_settings_widgets
        oskar_settings_containers)
    #=========================
endif (QT4_FOUND)

