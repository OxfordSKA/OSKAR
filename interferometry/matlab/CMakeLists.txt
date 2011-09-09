#
# src/interferometry/matlab/CMakeLists.txt
#

# ==== Matlab mex funtions
if (MATLAB_FOUND)

    cuda_add_library(interferometer1_scalar_d SHARED
        oskar_interferometer1_scalar_d_mex.cu mexFunction.def)
    target_link_libraries(interferometer1_scalar_d oskar ${MATLAB_LIBRARIES})
    set_target_properties(interferometer1_scalar_d PROPERTIES
        PREFIX "" SUFFIX ".${MATLAB_MEXFILE_EXT}"
        INSTALL_RPATH_USE_LINK_PATH TRUE COMPILE_FLAGS ${MATLAB_CXX_FLAGS}
        LINK_FLAGS ${MATLAB_CXX_FLAGS})
    install(TARGETS interferometer1_scalar_d DESTINATION ${OSKAR_MATLAB_INSTALL_DIR})
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION "${OSKAR_MATLAB_INSTALL_DIR}/.."
        FILES_MATCHING PATTERN "*.m"
        PATTERN ".svn" EXCLUDE)

endif (MATLAB_FOUND)

