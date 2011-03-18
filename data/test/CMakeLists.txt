#
# src/data/test
#



# CppUnit
# ==============================================================================
set(data_test_SRC
    src/CppUnitMain.cpp
)

if (CPPUNIT_FOUND)
    add_executable(dataTest ${data_test_SRC})
    target_link_libraries(dataTest data ${CPPUNIT_LIBRARIES})
endif()




# QTest framework unit tests.
# ==============================================================================
set(data_qtest_MOC_SRC
)

set(data_qtest_SRC
)

if(QtTest_FOUND)
    qt4_wrap_cpp(data_qtest_SRC ${data_qtest_MOC_SRC})
    add_executable(dataQTest ${data_qtest_SRC})
    target_link_libraries(dataQTest data ${QT_QTTEST_LIBRARY})
endif()
