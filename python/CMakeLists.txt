#
# src/python/CmakeLists.txt
#

if (PYTHON_FOUND AND CUDA_FOUND)

    include_directories(${PYTHON_INCLUDE_DIRS})
    include_directories(${NUMPY_INCLUDE_DIRS})

    set(OSKAR_PYTHON_INSTALL_DIR "${OSKAR_PYTHON_INSTALL_DIR}/oskar")
    oskar_py_module(_image_lib src/_oskar_image_lib.c LIBS oskar)
    oskar_py_module(_mem_lib src/_oskar_mem_lib.c LIBS oskar)
    oskar_py_module(_vis_lib src/_oskar_vis_lib.c LIBS oskar)

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

    # Install test files
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/test/"
        DESTINATION "${OSKAR_PYTHON_INSTALL_DIR}/.."
        FILES_MATCHING PATTERN "*.py"
        PATTERN ".svn" EXCLUDE
    )

endif()