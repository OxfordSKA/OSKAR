#
# src/math/matlab/@Jones/CMakeLists.txt
#

# ==== Matlab mex funtions
if (MATLAB_FOUND)
    include(oskar_build_macros)
    set(OSKAR_MEX_INSTALL_DIR "${OSKAR_MATLAB_INSTALL_DIR}/+oskar/@Jones/private")
    oskar_mex(Jones_constructor     oskar_Jones_constructor.cpp)
    oskar_mex(Jones_copy            oskar_Jones_copy.cpp)
    oskar_mex(Jones_destructor      oskar_Jones_destructor.cpp)
    oskar_mex(Jones_get_parameter   oskar_Jones_get_parameter.cpp)
    oskar_mex(Jones_get_values      oskar_Jones_get_values.cpp)
    oskar_mex(Jones_join_from_right oskar_Jones_join_from_right.cpp)
    oskar_mex(Jones_join            oskar_Jones_join.cpp)
    oskar_mex(Jones_set_real_scalar oskar_Jones_set_real_scalar.cpp)
    oskar_mex(Jones_set_values      oskar_Jones_set_values.cpp)

    # Copy *.m files from current directory to destination.
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
        DESTINATION "${OSKAR_MATLAB_INSTALL_DIR}/+oskar/@Jones"
        FILES_MATCHING PATTERN "*.m"
        PATTERN ".svn" EXCLUDE)
  
    add_subdirectory(test)

endif ()

