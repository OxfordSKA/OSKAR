#
# src/ms/CMakeLists.txt
#
#==============================================================================

set(ms_SRC
    src/MsAppend.cpp
    src/MsCreate.cpp
    src/MsManip.cpp
    src/oskar_ms_append_vis1.cpp
    src/oskar_ms_create_empty.cpp
    src/oskar_ms_create_meta1.cpp
    src/oskar_ms_create_vis1.cpp
)
set(ms_SRC "${ms_SRC}" PARENT_SCOPE)

# === Macros to build the oskar-lib ms module.
include_directories(${CASACORE_INCLUDE_DIR}/casacore)
add_library(oskar_ms ${ms_SRC})
set_target_properties(oskar_ms PROPERTIES COMPILE_FLAGS "-Wno-cast-qual")
target_link_libraries(oskar_ms ${CASACORE_LIBRARIES})

# === Install configuration for the oskar-lib ms module.
install(TARGETS oskar_ms LIBRARY DESTINATION ${OSKAR_LIB_INSTALL_DIR})
set_target_properties(oskar_ms PROPERTIES
    SOVERSION ${oskar-lib_SOVERSION})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DESTINATION ${OSKAR_INCLUDE_INSTALL_DIR}
    FILES_MATCHING PATTERN "*.h"
    PATTERN "src" EXCLUDE
    PATTERN "test" EXCLUDE
    PATTERN ".svn" EXCLUDE)

# === Recurse into test subdirectory.
add_subdirectory(test)
