#
# src/utility/matlab/@oskar_location/CMakeLists.txt
#

if (MATLAB_FOUND)    
    #include(oskar_build_macros)
    #set(OSKAR_MEX_INSTALL_DIR 
    #    "${OSKAR_MATLAB_INSTALL_DIR}/+oskar/@location/private")                          
    #oskar_mex(oskar_mem_constructor oskar_mex_mem_constructor.cpp)
    
    # Copy *.m files from current directy to destination
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
        DESTINATION "${OSKAR_MATLAB_INSTALL_DIR}/+oskar/@Location"
        FILES_MATCHING PATTERN "*.m"
        PATTERN ".svn" EXCLUDE)
        
endif (MATLAB_FOUND)

