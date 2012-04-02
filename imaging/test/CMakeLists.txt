#
# src/imaging/test/CMakeLists.txt
#

if (CPPUNIT_FOUND)

    set(imaging_test_SRC
        src/CppUnitMain.cpp
        src/Test_make_image.cpp
    )

    include_directories(${CPPUNIT_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})
    add_executable(imaging_test ${imaging_test_SRC})
    target_link_libraries(imaging_test oskar ${CPPUNIT_LIBRARIES} ${CUDA_LIBRARIES})
    if(BUILD_OSKAR_FITS)
        target_link_libraries(imaging_test oskar_fits)
    endif()
    add_test(imaging_test imaging_test)

endif()
