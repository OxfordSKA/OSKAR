#
# src/sky/CMakeLists.txt
#
#==============================================================================

include_directories(${QT_INCLUDE_DIR})

# === Create list of source files for the oskar-lib sky module.
set(sky_SRC
    src/angles_from_lm.cpp
    src/filter_sources_by_flux.cpp
    src/filter_sources_by_radius.cpp
    src/generate_random_sources.cpp
    src/rotate_sources.cpp
)
set(sky_SRC "${sky_SRC}" PARENT_SCOPE)

# === Build the oskar-lib sky module.
add_library(oskar_sky SHARED ${sky_SRC})
set_target_properties(oskar_sky PROPERTIES PREFIX "lib")
target_link_libraries(oskar_sky
    #
)

# === Recurse into test subdirectory.
add_subdirectory(test)

