#
# src/math/matlab/CMakeLists.txt
#

# ==== Matlab mex funtions
if (MATLAB_FOUND)

    #message("======== ${CUDA_INCLUDE_DIRS}")
    include_directories(${CUDA_INCLUDE_DIRS})
  
    # ==== Broken power law.
    cuda_add_library(oskar_random_bpl SHARED oskar_random_bpl.cpp mexFunction.def)
    #message("========== ${MATLAB_LIBRARIES}")
    #message("========== ${MATLAB_CXX_FLAGS}")
    target_link_libraries(oskar_random_bpl oskar ${MATLAB_LIBRARIES})
    set_target_properties(oskar_random_bpl PROPERTIES
        PREFIX "" SUFFIX ".${MATLAB_MEXFILE_EXT}"
        INSTALL_RPATH_USE_LINK_PATH TRUE
        COMPILE_FLAGS ${MATLAB_CXX_FLAGS}
        LINK_FLAGS ${MATLAB_CXX_FLAGS})
    install(TARGETS oskar_random_bpl DESTINATION ${OSKAR_MATLAB_INSTALL_DIR})

    # ==== Spherical positions.
    cuda_add_library(oskar_spherical_positions SHARED oskar_spherical_positions.cpp
        mexFunction.def)
    target_link_libraries(oskar_spherical_positions oskar ${MATLAB_LIBRARIES})
    set_target_properties(oskar_spherical_positions PROPERTIES
        PREFIX "" SUFFIX ".${MATLAB_MEXFILE_EXT}"
        INSTALL_RPATH_USE_LINK_PATH TRUE
        COMPILE_FLAGS ${MATLAB_CXX_FLAGS}
        LINK_FLAGS ${MATLAB_CXX_FLAGS})
    install(TARGETS oskar_spherical_positions DESTINATION ${OSKAR_MATLAB_INSTALL_DIR})

    # ==== Install *.m help files.
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION "${OSKAR_MATLAB_INSTALL_DIR}/.."
        FILES_MATCHING PATTERN "*.m"
        PATTERN ".svn" EXCLUDE)

endif ()

