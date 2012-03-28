#
# src/ms/CMakeLists.txt
#

set(ms_SRC
    src/oskar_MeasurementSet.cpp
    src/oskar_ms_append_vis.cpp
    src/oskar_ms_append_vis1.cpp
    src/oskar_ms_create_meta.cpp
    src/oskar_ms_create_meta1.cpp
)
set(ms_SRC "${ms_SRC}" PARENT_SCOPE)

# ==== Macros to build the oskar-lib ms module.
include_directories(${CASACORE_INCLUDE_DIR}/casacore)
add_library(oskar_ms ${ms_SRC})
set_target_properties(oskar_ms PROPERTIES COMPILE_FLAGS "-Wno-cast-qual")
target_link_libraries(oskar_ms ${CASACORE_LIBRARIES})
set_target_properties(oskar_ms
    PROPERTIES
    INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${OSKAR_LIB_INSTALL_DIR}
    INSTALL_RPATH_USE_LINK_PATH TRUE
    SOVERSION ${OSKAR_VERSION})

# ==== Install configuration for the OSKAR ms module.
install(TARGETS oskar_ms DESTINATION ${OSKAR_LIB_INSTALL_DIR})

# ==== Recurse into test subdirectory.
add_subdirectory(test)

