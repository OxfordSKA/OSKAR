#
# src/fits/test/CMakeLists.txt
#

add_executable(test_write_uvfits src/test_write_uvfits.cpp)
target_link_libraries(test_write_uvfits
    oskar_fits
)