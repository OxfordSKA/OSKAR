#
# oskar/imager/test/CMakeLists.txt
#

set(name imager_test)
set(${name}_SRC
    main.cpp
    Test_fits_write.cpp
    Test_grid_sum.cpp
    Test_Imager.cpp
)
add_executable(${name} ${${name}_SRC})
target_link_libraries(${name} oskar gtest)
add_test(imager_test ${name})
