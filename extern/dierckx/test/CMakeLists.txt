#
# src/extern/dierckx/test/CMakeLists.txt
#

if (CPPUNIT_FOUND AND OSKAR_FOUND)

    set(dierckx_test_SRC
        src/CppUnitMain.cpp
        src/Test_dierckx.cpp
    )

    include_directories(${OSKAR_INCLUDES})
    add_executable(dierckx_test ${dierckx_test_SRC})
    target_link_libraries(dierckx_test
        dierckx
        oskar
        ${CPPUNIT_LIBRARIES})

    add_test(dierckx_test dierckx_test)

endif ()

