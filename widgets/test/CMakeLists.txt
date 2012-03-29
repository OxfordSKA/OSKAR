#
# src/widgets/test/CMakeLists.txt
#

include_directories(${QT_INCLUDE_DIR})
add_executable(SettingsModelTest src/SettingsModelTest.cpp)
target_link_libraries(SettingsModelTest
    oskar_widgets
    ${QT_QTCORE_LIBRARY}
    ${QT_QTGUI_LIBRARY}
)

add_executable(DataFileModelTest src/DataFileModelTest.cpp)
target_link_libraries(DataFileModelTest
    oskar_widgets
    ${QT_QTCORE_LIBRARY}
    ${QT_QTGUI_LIBRARY}
)
