#
# oskar/convert/test/CMakeLists.txt
#

set(name convert_test)
set(${name}_SRC
    main.cpp
    Test_coordinate_conversions.cpp
    #Test_sofa.cpp
)
add_executable(${name} ${${name}_SRC})
target_link_libraries(${name}
    #sofa_c
    oskar gtest
    #${CASACORE_LIBRARIES}
)
add_test(${name} ${name})
