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
    src/rotate_sources.cpp
)

# If CUDA is present, then add the kernels directory.
if (CUDA_FOUND)
    add_subdirectory(cudak)
    foreach (file ${sky_cuda_kernels_SRC})
        list(APPEND sky_SRC cudak/${file})
    endforeach ()

    # Add the CUDA source files.
    set(sky_SRC ${sky_SRC}
        src/oskar_sky_cuda_horizon_clip.cu
        src/oskar_sky_cuda_ra_dec_to_hor_lmn.cu
    )

    # Build the library with CUDA.
    cuda_add_library(oskar_sky SHARED ${sky_SRC})
else ()
    # Build the library normally.
    add_library(oskar_sky SHARED ${sky_SRC})
endif ()

# Link the OSKAR math library.
target_link_libraries(oskar_sky oskar_math)

# Set the source list to be in the parent scope.
set(sky_SRC "${sky_SRC}" PARENT_SCOPE)

# Recurse into test subdirectory.
add_subdirectory(test)
