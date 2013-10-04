#
# src/imaging/test/CMakeLists.txt
#

if (CUDA_FOUND)
    set(name imaging_test)
    include_directories(${PROJECT_SOURCE_DIR}/extern/gtest-1.6.0/include)
    set(${name}_SRC
        Test_make_image.cpp
    )
    add_executable(${name} ${${name}_SRC})
    target_link_libraries(${name} oskar gtest_main)
    if(CFITSIO_FOUND)
        target_link_libraries(${name} oskar_fits)
    endif()
    set_target_properties(${name} PROPERTIES
        COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
        LINK_FLAGS "${OpenMP_CXX_FLAGS}")
    add_test(imaging_test ${name})
endif()
