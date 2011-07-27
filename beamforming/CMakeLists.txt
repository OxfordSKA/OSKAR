#
# src/beamforming/CMakeLists.txt
#
#==============================================================================

set(beamforming_SRC
    src/beamPattern.cpp
    #src/oskar_bf_cuda_ant_sig_iso_2d.cu
)
set(beamforming_SRC "${beamforming_SRC}" PARENT_SCOPE)

cuda_add_library(oskar_beamforming ${beamforming_SRC})
set_target_properties(oskar_beamforming PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}")
target_link_libraries(oskar_beamforming
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_LIBRARIES})
set_target_properties(oskar_beamforming PROPERTIES
    INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib
    INSTALL_RPATH_USE_LINK_PATH TRUE)

# === Install target
if (NOT WIN32)
    install(TARGETS oskar_beamforming LIBRARY DESTINATION ${OSKAR_LIB_INSTALL_DIR})
    set_target_properties(oskar_beamforming PROPERTIES
        SOVERSION ${oskar-lib_SOVERSION})
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION ${OSKAR_INCLUDE_INSTALL_DIR}
        FILES_MATCHING PATTERN "*.h"
        PATTERN "src" EXCLUDE
        PATTERN "test" EXCLUDE
        PATTERN ".svn" EXCLUDE)
endif ()


# === Recurse into test subdirectory.
add_subdirectory(test)
