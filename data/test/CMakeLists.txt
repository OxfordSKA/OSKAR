# CppUnit
# ==============================================================================
set(data_test_SRC
    src/CppUnitDataTest.cpp
)

include_directories(${QT_QTCORE_INCLUDE_DIR})

add_executable(dataTest ${data_test_SRC})

target_link_libraries(dataTest
    data
    ${QT_QTCORE_LIBRARY}
    ${CPPUNIT_LIBRARIES}
)

# QTest
# ==============================================================================
set(data_MOC_SRC
)

set(data_qtest_SRC
)

qt4_wrap_cpp(data_qtest_SRC ${data_MOC_SRC})

add_executable(dataQTest ${data_qtest_SRC})

target_link_libraries(dataQTest
    data
    ${QT_QTCORE_LIBRARY}
    ${QT_QTTEST_LIBRARY}
)
