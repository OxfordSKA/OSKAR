#
# src/matlab/settings/CMakeLists.txt
#

# Note: Needs the Qt4 headers from the dev package.
if (QT4_FOUND)

    include_directories(${QT_INCLUDE_DIR})

    set(OSKAR_MEX_INSTALL_DIR "${OSKAR_MATLAB_INSTALL_DIR}/+oskar/+settings")

    oskar_mex(set oskar_mex_set_setting.cpp EXTRA_LIBS
        ${MATLAB_QT_QTCORE_LIBRARY})

endif (QT4_FOUND)