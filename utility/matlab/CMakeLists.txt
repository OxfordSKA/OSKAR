#
# src/utility/matlab/CMakeLists.txt
#

# ==== Matlab mex funtions
if (MATLAB_FOUND)

    include_directories(${CUDA_INCLUDE_DIRS})

    # ==== oskar_check_cuda_error.
    cuda_add_library(oskar_check_cuda_error SHARED oskar_check_cuda_error_mex.cu mexFunction.def)
    target_link_libraries(oskar_check_cuda_error oskar ${MATLAB_LIBRARIES})
    set_target_properties(oskar_check_cuda_error PROPERTIES
        PREFIX "" SUFFIX ".${MATLAB_MEXFILE_EXT}"
        INSTALL_RPATH_USE_LINK_PATH TRUE
        COMPILE_FLAGS ${MATLAB_CXX_FLAGS}
        LINK_FLAGS ${MATLAB_CXX_FLAGS})
    install(TARGETS oskar_check_cuda_error DESTINATION ${OSKAR_MATLAB_INSTALL_DIR})

    # ==== Install *.m help files.
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION "${OSKAR_MATLAB_INSTALL_DIR}/.."
        FILES_MATCHING PATTERN "*.m"
        PATTERN ".svn" EXCLUDE)

endif ()

