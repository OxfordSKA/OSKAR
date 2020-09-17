#
# oskar/vis/CMakeLists.txt
#

set(vis_SRC
    src/oskar_vis_block_accessors.c
    src/oskar_vis_block_add_system_noise.c
    src/oskar_vis_block_clear.c
    src/oskar_vis_block_copy.c
    src/oskar_vis_block_create.c
    src/oskar_vis_block_create_from_header.c
    src/oskar_vis_block_free.c
    src/oskar_vis_block_read.c
    src/oskar_vis_block_resize.c
    src/oskar_vis_block_station_to_baseline_coords.c
    src/oskar_vis_block_write.c
    src/oskar_vis_header_accessors.c
    src/oskar_vis_header_create.c
    src/oskar_vis_header_create_copy.c
    src/oskar_vis_header_free.c
    src/oskar_vis_header_read.c
    src/oskar_vis_header_write.c

    # Deprecated:
    src/oskar_vis_accessors.c
    src/oskar_vis_create.c
    src/oskar_vis_free.c
    src/oskar_vis_read.c
    src/oskar_vis_write.c
)

if (CASACORE_FOUND)
    list(APPEND vis_SRC
        src/oskar_vis_block_write_ms.c
        src/oskar_vis_header_write_ms.c
    )
endif()

set(vis_SRC "${vis_SRC}" PARENT_SCOPE)

add_subdirectory(test)
