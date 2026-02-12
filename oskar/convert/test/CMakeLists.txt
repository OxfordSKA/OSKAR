#
# oskar/convert/test/CMakeLists.txt
#

set(name convert_test)
set(${name}_SRC
    main.cpp
    Test_convert_pqr_to_ecef_matrix.cpp
    Test_coordinate_conversions.cpp
    #Test_sofa.cpp
)
add_executable(${name} ${${name}_SRC})
target_link_libraries(${name}
    #sofa_c
    oskar gtest
    #${CASACORE_LIBRARIES}
)
if (OpenMP_CXX_FOUND)
    target_link_libraries(${name} OpenMP::OpenMP_CXX)
endif()
add_test(NAME ${name} COMMAND ${name})
