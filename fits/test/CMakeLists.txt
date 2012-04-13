#
# src/fits/test/CMakeLists.txt
#

#add_executable(test_uvfits_writer src/test_uvfits_writer.cpp)
#target_link_libraries(test_uvfits_writer oskar_fits)


if (CPPUNIT_FOUND AND CUDA_FOUND)

    set(fits_test_SRC
        src/CppUnitMain.cpp
        src/Test_fits_image_write.cpp
        src/Test_fits_write.cpp
    )

    include_directories(${CPPUNIT_INCLUDE_DIR})
    add_executable(fits_test ${fits_test_SRC})
    target_link_libraries(fits_test oskar oskar_fits ${CPPUNIT_LIBRARIES})
    add_test(fits_test fits_test)

endif (CPPUNIT_FOUND AND CUDA_FOUND)
