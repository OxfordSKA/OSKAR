#
# src/interferometry/CMakeLists.txt
#
#==============================================================================

include_directories(${oskar-lib_SOURCE_DIR}/interferometry)

# Add the source files in this directory.
set(interferometry_SRC
    src/oskar_interferometry_baselines.c
    src/oskar_interferometry_xyz2uvw.c
)
set(interferometry_SRC "${interferometry_SRC}" PARENT_SCOPE)

#cuda_add_library(oskar_interferometry ${interferometry_SRC})
#set_target_properties(oskar_interferometry PROPERTIES
#    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
#    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
#)
#set_target_properties(oskar_interferometry PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
#if (NOT WIN32)
#    install(TARGETS oskar_interferometry LIBRARY DESTINATION ${OSKAR_LIB_INSTALL_DIR})
#    set_target_properties(oskar_interferometry PROPERTIES
#        SOVERSION ${oskar-lib_SOVERSION})
#    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#        DESTINATION ${OSKAR_INCLUDE_INSTALL_DIR}
#        FILES_MATCHING PATTERN "*.h"
#        PATTERN "src" EXCLUDE
#        PATTERN "test" EXCLUDE
#        PATTERN ".svn" EXCLUDE)
#endif ()
