#
# src/sky/CMakeLists.txt
#
#==============================================================================

# === Create list of source files for the oskar-lib sky module.
set(sky_SRC
    src/generate_random_sources.cpp
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

