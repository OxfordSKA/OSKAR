if (APPLE)
    # Need a Python 3 interpreter to run fixup_bundle.py at install time.
    # (find_package here so Python3_EXECUTABLE is defined in this scope; the
    # result is cached, so this does not trigger a fresh search.)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    install(CODE "
        get_filename_component(BUNDLE_DIR \${CMAKE_INSTALL_PREFIX} DIRECTORY)
        get_filename_component(BUNDLE_DIR \${BUNDLE_DIR} DIRECTORY)
        get_filename_component(BUNDLE_DIR \${BUNDLE_DIR} DIRECTORY)
        set(BUNDLE_DIR \${BUNDLE_DIR}/OSKAR.app)
        if (EXISTS \${BUNDLE_DIR})
            list(INSERT CMAKE_MODULE_PATH 0 ${PROJECT_SOURCE_DIR}/cmake)
            message(\"Fixing up \${BUNDLE_DIR}\")
            # fixup_bundle.py relocates dependencies into the bundle AND
            # re-signs the relocated Mach-O files (install_name_tool rewrites
            # invalidate signatures, which arm64 refuses to load). The signing
            # identity is passed via OSKAR_CODESIGN_IDENTITY. CPack seals the
            # completed .app after it adds Info.plist.
            execute_process(COMMAND
                \${CMAKE_COMMAND} -E env
                \"OSKAR_CODESIGN_IDENTITY=${OSKAR_CODESIGN_IDENTITY}\"
                ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/cmake/fixup_bundle.py
                \${BUNDLE_DIR} ${QtMac_location} ${QtMacStyle_location}
                RESULT_VARIABLE fixup_result)
            # Fail loudly: a silent fixup failure produces a bundle that only
            # runs where the Homebrew dependencies happen to be installed.
            if (NOT fixup_result EQUAL 0)
                message(FATAL_ERROR
                    \"fixup_bundle.py failed (exit \${fixup_result})\")
            endif()
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
