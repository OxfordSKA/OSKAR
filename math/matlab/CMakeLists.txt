#
# src/math/matlab/CMakeLists.txt
#
#==============================================================================

# ==== Matlab mex funtions
if (MATLAB_FOUND)

    # ==== Broken power law.
    set(MATLAB_MEXFILE_EXT mexa64)
    include_directories(${MATLAB_INCLUDE_DIR})
    add_library(oskar_random_bpl SHARED oskar_random_bpl.cpp mexFunction.def)
    target_link_libraries(oskar_random_bpl oskar ${MATLAB_LIBRARIES})
    set_target_properties(oskar_random_bpl PROPERTIES
        PREFIX "" SUFFIX ".${MATLAB_MEXFILE_EXT}")

    # ==== Spherical positions.
    set(MATLAB_MEXFILE_EXT mexa64)
    include_directories(${MATLAB_INCLUDE_DIR})
    add_library(oskar_spherical_positions SHARED oskar_spherical_positions.cpp
        mexFunction.def)
    target_link_libraries(oskar_spherical_positions oskar ${MATLAB_LIBRARIES})
    set_target_properties(oskar_spherical_positions PROPERTIES
        PREFIX "" SUFFIX ".${MATLAB_MEXFILE_EXT}")

    # ==== Install targets
    if (NOT WIN32)
        install(TARGETS oskar_random_bpl          DESTINATION ${OSKAR_MATLAB_INSTALL_DIR})
        install(TARGETS oskar_spherical_positions DESTINATION ${OSKAR_MATLAB_INSTALL_DIR})
        install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            DESTINATION "${OSKAR_MATLAB_INSTALL_DIR}/.."
            FILES_MATCHING PATTERN "*.m"
            PATTERN ".svn" EXCLUDE)
    endif ()

endif (MATLAB_FOUND)

