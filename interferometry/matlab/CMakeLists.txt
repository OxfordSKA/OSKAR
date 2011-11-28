#
# src/interferometry/matlab/CMakeLists.txt
#

# ==== Matlab mex funtions
if (MATLAB_FOUND)
    include(oskar_build_macros)

    add_subdirectory(@Visibilities)
    
    # Copy *.m files from current directory to destination.
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
        DESTINATION "${OSKAR_MATLAB_INSTALL_DIR}"
        FILES_MATCHING PATTERN "*.m"
        PATTERN ".svn" EXCLUDE)

endif (MATLAB_FOUND)

