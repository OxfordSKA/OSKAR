#
# python/CMakeLists.txt
#

if (PYTHON_FOUND)

    include_directories(${PYTHON_INCLUDE_DIRS})
    include_directories(${NUMPY_INCLUDE_DIRS})

    set(OSKAR_PYTHON_INSTALL_DIR "${OSKAR_PYTHON_INSTALL_DIR}/oskar")

    if (CASACORE_FOUND)
        oskar_py_module(_measurement_set_lib src/oskar_measurement_set_lib.c LIBS oskar)
    endif()

    oskar_py_module(_bda_utils src/oskar_bda_utils.c)
    oskar_py_module(_imager_lib src/oskar_imager_lib.c LIBS oskar)
    oskar_py_module(_simulator_lib src/oskar_simulator_lib.c LIBS oskar oskar_apps)
    oskar_py_module(_sky_lib src/oskar_sky_lib.c LIBS oskar)
    oskar_py_module(_telescope_lib src/oskar_telescope_lib.c LIBS oskar)
    oskar_py_module(_vis_block_lib src/oskar_vis_block_lib.c LIBS oskar)
    oskar_py_module(_vis_header_lib src/oskar_vis_header_lib.c LIBS oskar)

    configure_file(${PROJECT_SOURCE_DIR}/python/version.py.in
        ${PROJECT_BINARY_DIR}/python/version.py @ONLY)

    # Install module scripts
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
        DESTINATION "${OSKAR_PYTHON_INSTALL_DIR}"
        FILES_MATCHING PATTERN "*.py"
        PATTERN ".svn" EXCLUDE
        PATTERN "src" EXCLUDE
        PATTERN "test" EXCLUDE
    )

    # Install configured module scripts
    install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/"
        DESTINATION "${OSKAR_PYTHON_INSTALL_DIR}"
        FILES_MATCHING PATTERN "*.py"
        PATTERN "CMakeFiles" EXCLUDE
        PATTERN "test" EXCLUDE
    )

endif()