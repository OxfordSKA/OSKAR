#
# src/matlab/CMakeLists.txt
#

if (MATLAB_FOUND)

    include(oskar_build_macros)
    
    add_subdirectory(binary_file)
    add_subdirectory(cuda)
    add_subdirectory(image)
    add_subdirectory(imager)
    add_subdirectory(settings)
    add_subdirectory(sky)
    add_subdirectory(visibilities)

endif (MATLAB_FOUND)

