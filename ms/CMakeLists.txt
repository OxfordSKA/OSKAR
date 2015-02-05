#
# src/ms/CMakeLists.txt
#

if (CASACORE_FOUND)

    include_directories(${CASACORE_INCLUDE_DIR})
    include_directories(${CASACORE_INCLUDE_DIR}/casacore)

    set(ms_SRC
        src/oskar_measurement_set.cpp
    )

    # Build and install the oskar_ms library.
    add_library(oskar_ms ${ms_SRC})
    set_target_properties(oskar_ms PROPERTIES 
        COMPILE_FLAGS "-Wno-cast-qual -fvisibility=default")
    target_link_libraries(oskar_ms ${CASACORE_LIBRARIES})
    set_target_properties(oskar_ms PROPERTIES
        INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${OSKAR_LIB_INSTALL_DIR}
        INSTALL_RPATH_USE_LINK_PATH TRUE
        SOVERSION ${OSKAR_VERSION}
        VERSION ${OSKAR_VERSION})
    install(TARGETS oskar_ms DESTINATION ${OSKAR_LIB_INSTALL_DIR})

    # ==== Recurse into test subdirectory.
    add_subdirectory(test)

endif(CASACORE_FOUND)
