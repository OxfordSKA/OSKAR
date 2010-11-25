#
# /src/math/core/test
#


# CppUnit
# ==============================================================================
set(math_core_test_SRC
    src/CppUnitMain.cpp
    src/SphericalPositionsTest.cpp
)

include_directories()

add_executable(mathCoreTest ${math_core_test_SRC})

target_link_libraries(mathCoreTest
    math-core
    ${CPPUNIT_LIBRARIES}
)



# QTest
# ==============================================================================
#set(math_core_qtest_MOC_SRC
#)

#set(math_core_qtest_SRC
#)

#qt4_wrap_cpp(math_core_qtest_SRC ${math_core_qtest_MOC_SRC})

#add_executable(mathCoreQTest ${math_core_qtest_SRC})

#target_link_libraries(mathCoreQTest
#    math-core
#    ${QT_QTTEST_LIBRARY}
#)
