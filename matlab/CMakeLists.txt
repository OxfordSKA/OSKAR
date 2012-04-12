#
# src/matlab/CMakeLists.txt
#

if (MATLAB_FOUND)

    include(oskar_build_macros)
    
    add_subdirectory(settings)
    add_subdirectory(binary_file)

endif (MATLAB_FOUND)

