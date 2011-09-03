#
# src/fits/test/CMakeLists.txt
#

add_executable(test_uvfits_writer src/test_uvfits_writer.cpp)
target_link_libraries(test_uvfits_writer
    oskar_fits
)