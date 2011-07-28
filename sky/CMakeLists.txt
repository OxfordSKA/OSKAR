#
# src/sky/CMakeLists.txt
#
#==============================================================================

include_directories(${oskar-lib_SOURCE_DIR}/math)

# Add the source files in this directory.
set(sky_SRC
    src/angles_from_lm.cpp
    src/filter_sources_by_flux.cpp
    src/filter_sources_by_radius.cpp
    src/generate_random_sources.cpp
    src/oskar_sky_cuda_horizon_clip.cu
    src/oskar_sky_cuda_ra_dec_to_az_el.cu
    src/oskar_sky_cuda_ra_dec_to_hor_lmn.cu
    src/oskar_sky_cuda_ra_dec_to_relative_lmn.cu
    src/oskar_sky_date_time_to_mjd.cpp
    src/oskar_sky_equation_of_equinoxes_fast.cpp
    src/oskar_sky_mjd_to_gmst.cpp
    src/oskar_sky_mjd_to_last_fast.cpp
    src/oskar_sky_mjd_to_lmst.cpp
    src/rotate_sources.cpp
)

# Add the CUDA kernels.
add_subdirectory(cudak)
foreach (file ${sky_cuda_kernels_SRC})
    list(APPEND sky_SRC cudak/${file})
endforeach ()

set(sky_SRC "${sky_SRC}" PARENT_SCOPE)

# Build the library with CUDA.
#cuda_add_library(oskar_sky SHARED ${sky_SRC})
#
# Link the OSKAR math library.
#target_link_libraries(oskar_sky oskar_math)
#
#
#if (NOT WIN32)
#    install(TARGETS oskar_sky LIBRARY DESTINATION ${OSKAR_LIB_INSTALL_DIR})
#    set_target_properties(oskar_sky PROPERTIES
#        SOVERSION ${oskar-lib_SOVERSION})
#    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#        DESTINATION ${OSKAR_INCLUDE_INSTALL_DIR}
#        FILES_MATCHING PATTERN "*.h"
#        PATTERN "src" EXCLUDE
#        PATTERN "test" EXCLUDE
#        PATTERN ".svn" EXCLUDE)
#endif ()

# Recurse into test subdirectory.
add_subdirectory(test)
