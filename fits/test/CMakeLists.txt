#
# src/fits/test/CMakeLists.txt
#

#add_executable(test_uvfits_writer src/test_uvfits_writer.cpp)
#target_link_libraries(test_uvfits_writer oskar_fits)

set(name fits_test)
set(${name}_SRC
    Test_fits_healpix_write.cpp
    Test_fits_image_write.cpp
    Test_fits_write.cpp
    Test_fits_to_sky_model.cpp
)
add_executable(${name} ${${name}_SRC})
target_link_libraries(${name} oskar oskar_fits gtest_main)
set_target_properties(${name} PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}")
add_test(fits_test ${name})
