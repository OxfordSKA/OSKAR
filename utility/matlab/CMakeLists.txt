#
# src/utility/matlab/CMakeLists.txt
#

if (MATLAB_FOUND)    
    include(oskar_build_macros)
    set(OSKAR_MEX_INSTALL_DIR "${OSKAR_MATLAB_INSTALL_DIR}/+cuda")
    # Note has to be *.cu to link correctly with this macro.
    oskar_mex(device_info oskar_mex_device_info.cu) 
    
    set(OSKAR_MEX_INSTALL_DIR "${OSKAR_MATLAB_INSTALL_DIR}")
    oskar_mex(oskar_binary_file_read oskar_mex_binary_file_read.cpp)
    oskar_mex(oskar_binary_file_query oskar_mex_binary_file_query.cpp)
    oskar_mex(oskar_binary_file_read_record oskar_mex_binary_file_read_record.cpp)
    
    add_subdirectory("@Mem")
    
    add_subdirectory("@Type")
    
    add_subdirectory("@Location")
    
    # Copy *.m files from current directy to destination
    #install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
    #    DESTINATION "${OSKAR_MATLAB_INSTALL_DIR}"
    #    FILES_MATCHING PATTERN "*.m"
    #    PATTERN ".svn" EXCLUDE)
        
endif (MATLAB_FOUND)

