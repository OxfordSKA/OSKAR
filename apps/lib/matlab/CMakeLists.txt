#
# src/apps/lib/matlab/CMakeLists.txt
#

if (MATLAB_FOUND)

    include(oskar_build_macros)

    set(OSKAR_MEX_INSTALL_DIR "${OSKAR_MATLAB_INSTALL_DIR}/+apps")
    oskar_qt_mex(set_setting oskar_mex_set_setting.cpp)
    #target_link_libraries(set_setting ${MATLAB_QT_QTCORE_LIBRARY})

endif (MATLAB_FOUND)

