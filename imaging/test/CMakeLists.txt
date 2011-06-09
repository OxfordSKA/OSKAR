#
# src/imaging/test/CMakeLists.txt
#

include_directories(
    ${Qwt5_INCLUDE_DIR}
    ${QT_INCLUDE_DIR}
    ${QT_QTCORE_INCLUDE_DIR}
    ${QT_QTGUI_INCLUDE_DIR}
)

# CppUnit
set(imaging_test_SRC
    src/CppUnitMain.cpp
    src/ImagingTest.cpp
)

add_executable(imagingTest ${imaging_test_SRC})
target_link_libraries(imagingTest
    oskar_imaging
    oskar_widgets_plotting
    ${QT_QTCORE_LIBRARY}
    ${CPPUNIT_LIBRARIES}
)



# === QtTest unit testing.
if (QT_QTTEST_FOUND)
    qt4_wrap_cpp(imaging_test_MOC QTestImaging.h)
    add_executable(qtest_Imaging ${imaging_test_MOC})
    target_link_libraries(qtest_Imaging
        oskar_imaging
        oskar_widgets_plotting
        ${QT_QTTEST_LIBRARY}
        ${QT_QTGUI_LIBRARY}
    )
endif ()
