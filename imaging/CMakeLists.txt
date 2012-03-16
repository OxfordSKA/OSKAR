#
# src/imaging/CMakeLists.txt
#


#add_subdirectory(fft)
add_subdirectory(matlab)

# FIXME: probably shouldn't have this include if we want to enforce
# header paths.
include_directories(${oskar-lib_SOURCE_DIR}/imaging)


# Add the source files in this directory.
set(imaging_SRC
    src/oskar_evaluate_image_lm_grid.c
    src/oskar_image_free.c
    src/oskar_image_init.c
    src/oskar_image_resize.c
    src/oskar_image_write.c
    src/oskar_Image.cpp
    src/oskar_make_image_dft.c
)
set(imaging_SRC "${imaging_SRC}" PARENT_SCOPE)

#add_subdirectory(test)

