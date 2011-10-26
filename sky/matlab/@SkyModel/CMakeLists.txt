#
# src/sky/matlab/@SkyModel/CMakeLists.txt
#

if (MATLAB_FOUND)
    include(oskar_build_macros)
    set(OSKAR_MEX_INSTALL_DIR "${OSKAR_MATLAB_INSTALL_DIR}/+oskar/@SkyModel/private")
    #oskar_mex(Jones_constructor     oskar_Jones_constructor.cpp)

    # Copy *.m files from current directory to destination.
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
        DESTINATION "${OSKAR_MATLAB_INSTALL_DIR}/+oskar/@SkyModel"
        FILES_MATCHING PATTERN "*.m"
        PATTERN ".svn" EXCLUDE)
    
    add_subdirectory(test)

endif ()
