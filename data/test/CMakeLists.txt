#
# src/data/test
#



# CppUnit
# ==============================================================================
set(data_test_SRC
    src/CppUnitMain.cpp
)

include_directories()

add_executable(dataTest ${data_test_SRC})

target_link_libraries(dataTest
    data
    ${CPPUNIT_LIBRARIES}
)

# QTest
# ==============================================================================
set(data_qtest_MOC_SRC
)

set(data_qtest_SRC
)

qt4_wrap_cpp(data_qtest_SRC ${data_qtest_MOC_SRC})

add_executable(dataQTest ${data_qtest_SRC})

target_link_libraries(dataQTest
    data
    ${QT_QTTEST_LIBRARY}
)
