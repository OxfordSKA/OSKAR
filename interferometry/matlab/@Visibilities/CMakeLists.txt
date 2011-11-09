#
# src/math/matlab/@Jones/CMakeLists.txt
#

# ==== Matlab mex funtions
if (MATLAB_FOUND)
    include(oskar_build_macros)
    #set(OSKAR_MEX_INSTALL_DIR "${OSKAR_MATLAB_INSTALL_DIR}/+oskar/@Visibilties/private")
    #set(OSKAR_MEX_INSTALL_DIR "${OSKAR_MATLAB_INSTALL_DIR}/+oskar")
    oskar_mex(oskar_visibilities_read     oskar_mex_visibilities_read.cpp)

    # Copy *.m files from current directory to destination.
    #install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
    #    DESTINATION "${OSKAR_MATLAB_INSTALL_DIR}/+oskar/@Jones"
    #    FILES_MATCHING PATTERN "*.m"
    #    PATTERN ".svn" EXCLUDE)

    
    #add_subdirectory(test)

endif ()

