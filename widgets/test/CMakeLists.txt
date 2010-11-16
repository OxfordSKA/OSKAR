set(widgets_test_MOC_SRC
)

set(widgets_test_SRC
)

qt4_wrap_cpp(widgets_test_MOC ${widgets_test_MOC_SRC})

include_directories()

add_executable(widgetsTest ${widgets_test_SRC})

target_link_libraries(widgetsTest
    widgets
    ${CPPUNIT_LIBRARIES}
    ${QT_QTCORE_LIBRARIES}
    ${QT_QTTEST_LIBRARY}
)


