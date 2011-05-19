#
# src/sky/test/CMakeLists.txt
#

# CppUnit
set(sky_test_SRC
    src/CppUnitMain.cpp
    src/SkyTest.cpp
)

add_executable(skyTest ${sky_test_SRC})
target_link_libraries(skyTest
    oskar_sky
    ${QT_QTCORE_LIBRARY}
    ${CPPUNIT_LIBRARIES}
)

