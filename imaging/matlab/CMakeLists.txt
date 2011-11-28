#
# src/interferometry/matlab/CMakeLists.txt
#

# ==== Matlab mex funtions
if (MATLAB_FOUND)

    include(oskar_build_macros)
    
    include_directories(${CUDA_INCLUDE_DIRS})
    
    oskar_mex(oskar_make_image oskar_make_image.cpp)
    
    # Copy *.m files from current directory to destination.
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
        DESTINATION "${OSKAR_MATLAB_INSTALL_DIR}"
        FILES_MATCHING PATTERN "*.m"
        PATTERN ".svn" EXCLUDE)
    
endif (MATLAB_FOUND)

