#
# src/widgets/test/CMakeLists.txt
#


# CppUnit Tests
# ==============================================================    
set(widgets_test_SRC
    src/CppUnitMain.cpp
)

add_executable(widgetsTest ${widgets_test_SRC})
target_link_libraries(widgetsTest
    oskar_widgets
    ${CPPUNIT_LIBRARIES}
)
    
# QTest Tests
# ==============================================================
    
qt4_wrap_cpp(configTable_MOC QTestConfigOptionsTable.h)
add_executable(qtest_ConfigOptionsTable ${configTable_MOC})
target_link_libraries(qtest_ConfigOptionsTable
    oskar_widgets
    ${QT_QTTEST_LIBRARY}
)
    