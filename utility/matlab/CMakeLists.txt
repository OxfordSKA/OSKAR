#
# src/utility/matlab/CMakeLists.txt
#

if (MATLAB_FOUND)    
    include(oskar_build_macros)
    set(OSKAR_MEX_INSTALL_DIR "${OSKAR_MATLAB_INSTALL_DIR}/+cuda")
    oskar_mex(device_info device_info.cu)
    
    # Copy *.m files from current directy to destination
#    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
#        DESTINATION "${OSKAR_MATLAB_INSTALL_DIR}/+cuda"
#        FILES_MATCHING PATTERN "*.m"
#        PATTERN ".svn" EXCLUDE)
        
endif (MATLAB_FOUND)

