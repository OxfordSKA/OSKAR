#
# src/matlab/CMakeLists.txt
#

if (MATLAB_FOUND AND CUDA_FOUND)

    include_directories(${MATLAB_INCLUDE_DIR})

    add_subdirectory(binary_file)
    add_subdirectory(convert)
    add_subdirectory(cuda)
    add_subdirectory(image)
    #add_subdirectory(math)
    add_subdirectory(settings)
    add_subdirectory(sky)
    #add_subdirectory(time_freq_data)
    add_subdirectory(vis)

endif (MATLAB_FOUND AND CUDA_FOUND)

