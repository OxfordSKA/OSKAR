#
# oskar/beam_pattern/CMakeLists.txt
#

set(beam_pattern_SRC
    src/oskar_beam_pattern_accessors.c
    src/oskar_beam_pattern_check_init.c
    src/oskar_beam_pattern_create.c
    src/oskar_beam_pattern_free.c
    src/oskar_beam_pattern_reset_cache.c
    src/oskar_beam_pattern_run.c
    src/private_beam_pattern_free_device_data.c
    src/private_beam_pattern_generate_coordinates.c
)

set(beam_pattern_SRC "${beam_pattern_SRC}" PARENT_SCOPE)

# Tests
if (BUILD_TESTING OR NOT DEFINED BUILD_TESTING)
    add_subdirectory(test)
endif()
