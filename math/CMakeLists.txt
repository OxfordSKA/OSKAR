#
# src/math/CMakeLists.txt
#

add_subdirectory(core)
add_subdirectory(beamforming)
#add_subdirectory(modules)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DESTINATION ${OSKAR_INCLUDE_INSTALL_DIR}
    FILES_MATCHING PATTERN "*.h"
    PATTERN "src" EXCLUDE
    PATTERN "test" EXCLUDE)
