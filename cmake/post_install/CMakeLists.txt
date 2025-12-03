if (APPLE)
    install(CODE "
        get_filename_component(BUNDLE_DIR \${CMAKE_INSTALL_PREFIX} DIRECTORY)
        get_filename_component(BUNDLE_DIR \${BUNDLE_DIR} DIRECTORY)
        get_filename_component(BUNDLE_DIR \${BUNDLE_DIR} DIRECTORY)
        set(BUNDLE_DIR \${BUNDLE_DIR}/OSKAR.app)
        if (EXISTS \${BUNDLE_DIR})
            list(INSERT CMAKE_MODULE_PATH 0 ${PROJECT_SOURCE_DIR}/cmake)
            message(\"Fixing up \${BUNDLE_DIR}\")
            execute_process(COMMAND
                python ${PROJECT_SOURCE_DIR}/cmake/fixup_bundle.py
                \${BUNDLE_DIR} ${QtMac_location} ${QtMacStyle_location})
        endif()
        " COMPONENT runtime)
elseif (WIN32)
    configure_file(
        ${PROJECT_SOURCE_DIR}/cmake/oskar_bundle_deps.cmake.in
        ${PROJECT_BINARY_DIR}/oskar_bundle_deps.cmake
        @ONLY
    )
    install(
        SCRIPT "${PROJECT_BINARY_DIR}/oskar_bundle_deps.cmake"
        COMPONENT applications
    )
endif()
