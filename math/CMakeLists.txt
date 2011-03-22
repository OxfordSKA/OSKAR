#
# src/math/CMakeLists.txt
#
#==============================================================================

# === Build the oskar-lib core math sub-module.
add_subdirectory(core)

# === Build the oskar-lib beamforming math sub-module.
add_subdirectory(beamforming)

# === Build the oskar-lib ??? math sub-module.
#add_subdirectory(modules)

# === Install target for oskar-lib math header files.
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DESTINATION ${OSKAR_INCLUDE_INSTALL_DIR}
    FILES_MATCHING PATTERN "*.h"
    PATTERN "src" EXCLUDE
    PATTERN "test" EXCLUDE)
