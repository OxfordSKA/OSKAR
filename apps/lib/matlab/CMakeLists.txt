#
# src/apps/lib/matlab/CMakeLists.txt
#

if (MATLAB_FOUND)

    include(oskar_build_macros)

    set(OSKAR_MEX_INSTALL_DIR "${OSKAR_MATLAB_INSTALL_DIR}/+apps")
    oskar_mex(set_setting  oskar_mex_set_setting.cpp)
    #target_link_libraries(set_setting ${QT_QTCORE_LIBRARY})
    message("********** ${QT_QTCORE_LIBRARY}")
    message("********** ${MATLAB_SYS}")
    target_link_libraries(set_setting /usr/local/matlab/bin/glnxa64/libQtCore.so.4)


endif (MATLAB_FOUND)

