#
# src/utility/matlab/CMakeLists.txt
#

if (MATLAB_FOUND)    
    include(oskar_build_macros)
    
    add_subdirectory("@Mem")
    
    add_subdirectory("@Type")
    
    add_subdirectory("@Location")
    
    # Copy *.m files from current directy to destination
    #install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
    #    DESTINATION "${OSKAR_MATLAB_INSTALL_DIR}"
    #    FILES_MATCHING PATTERN "*.m"
    #    PATTERN ".svn" EXCLUDE)
        
endif (MATLAB_FOUND)

