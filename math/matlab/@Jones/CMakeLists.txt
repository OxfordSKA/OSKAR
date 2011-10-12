#
# src/math/matlab/CMakeLists.txt
#

# ==== Matlab mex funtions
if (MATLAB_FOUND)
    include(oskar_build_macros)
    set(OSKAR_MEX_INSTALL_DIR "${OSKAR_MATLAB_INSTALL_DIR}/+oskar/@Jones")
    oskar_mex(mex_constructor     mex_constructor.cpp)
    oskar_mex(mex_destructor      mex_destructor.cpp)
    oskar_mex(mex_copy            mex_copy.cpp)
    oskar_mex(mex_get_values      mex_get_values.cpp)
    oskar_mex(mex_real_scalar     mex_set_real_scalar.cpp)
    oskar_mex(mex_get_parameter   mex_get_parameter.cpp)
    oskar_mex(mex_set_values      mex_set_values.cpp)
    oskar_mex(mex_join            mex_join.cpp)
    oskar_mex(mex_join_from_right mex_join_from_right.cpp)
    oskar_mex(mex_join_to_left    mex_join_to_left.cpp)

    # Copy *.m files
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION "${OSKAR_MATLAB_INSTALL_DIR}/+oskar"
        FILES_MATCHING PATTERN "*.m"
        PATTERN ".svn" EXCLUDE)

endif ()

