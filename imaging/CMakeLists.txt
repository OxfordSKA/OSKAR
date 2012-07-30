#
# src/imaging/CMakeLists.txt
#

#add_subdirectory(fft)

# Add the source files in this directory.
set(imaging_SRC
    src/oskar_evaluate_image_lm_grid.c
    src/oskar_get_image_vis_amps.cpp
    src/oskar_get_image_baseline_coords.cpp
    src/oskar_get_image_stokes.c
    src/oskar_image_evaluate_ranges.c
    src/oskar_image_free.c
    src/oskar_image_init.c
    src/oskar_image_read.c
    src/oskar_image_resize.c
    src/oskar_image_write.c
    src/oskar_Image.cpp
    src/oskar_make_image_dft.c
    src/oskar_make_image.cpp
    src/oskar_setup_image.cpp
)
set(imaging_SRC "${imaging_SRC}" PARENT_SCOPE)

add_subdirectory(test)
