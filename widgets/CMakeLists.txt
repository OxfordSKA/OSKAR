#
# src/widgets/CMakeLists.txt
#
#==============================================================================


# === Plotting widgets sub-module.
if (Qwt5_FOUND)
    add_subdirectory(plotting)
endif()


#install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#    DESTINATION ${OSKAR_INCLUDE_INSTALL_DIR}
#    FILES_MATCHING PATTERN "*.h"
#    PATTERN "src" EXCLUDE
#    PATTERN ".svn" EXCLUDE
#    PATTERN "test" EXCLUDE)

