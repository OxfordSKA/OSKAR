#
# src/imaging/test/CMakeLists.txt
#

if (CUDA_FOUND)
    set(name imaging_test)
    include_directories(${CUDA_INCLUDE_DIRS})
    set(${name}_SRC
        main.cpp
        Test_make_image.cpp
    )
    add_executable(${name} ${${name}_SRC})
    target_link_libraries(${name} oskar gtest oskar_fits)
    set_target_properties(${name} PROPERTIES
        COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
        LINK_FLAGS "${OpenMP_CXX_FLAGS}")
    add_test(imaging_test ${name})
endif()
