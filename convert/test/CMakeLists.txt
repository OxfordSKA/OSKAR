#
# src/convert/test/CMakeLists.txt
#

include_directories(${CUDA_INCLUDE_DIRS})

set(name convert_test)
set(${name}_SRC
    main.cpp
    Test_coordinate_conversions.cpp
    #Test_sofa.cpp
)
add_executable(${name} ${${name}_SRC})
target_link_libraries(${name}
    #sofa_c
    oskar gtest ${CUDA_LIBRARIES}
    #${CASACORE_LIBRARIES}
)
add_test(${name} ${name})
