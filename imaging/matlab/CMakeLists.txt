#
# src/imaging/matlab/CMakeLists.txt
#

# ==== Matlab mex funtions
if (MATLAB_FOUND)

    include(oskar_build_macros)
    
    include_directories(${CUDA_INCLUDE_DIRS})

    add_library(oskar_mex_imaging STATIC
        src/oskar_mex_image_settings_from_matlab.cpp
        src/oskar_mex_image_to_matlab_struct.cpp)
    
    oskar_mex(oskar_make_image oskar_mex_make_image_old.cpp)
    oskar_mex(oskar_make_image_new oskar_mex_make_image.cpp)
    target_link_libraries(oskar_make_image_new 
        oskar_mex_imaging
        oskar_mex_interferometry)
    oskar_mex(oskar_image_read oskar_mex_image_read.cpp)
    target_link_libraries(oskar_image_read oskar_mex_imaging)
    
    # Copy *.m files from current directory to destination.
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
        DESTINATION "${OSKAR_MATLAB_INSTALL_DIR}"
        FILES_MATCHING PATTERN "*.m"
        PATTERN ".svn" EXCLUDE)
    
endif (MATLAB_FOUND)

