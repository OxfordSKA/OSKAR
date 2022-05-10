#
# oskar/mem/CMakeLists.txt
#

set(mem_SRC
    define_mem_add.h
    define_mem_conjugate.h
    define_mem_multiply.h
    define_mem_normalise.h
    define_mem_scale_real.h
    define_mem_set_value_real.h
    src/oskar_binary_read_mem.c
    src/oskar_binary_write_mem.c
    src/oskar_mem_accessors.c
    src/oskar_mem_add.c
    src/oskar_mem_add_real.c
    src/oskar_mem_append_raw.c
    src/oskar_mem_clear_contents.c
    src/oskar_mem_conjugate.c
    src/oskar_mem_convert_precision.c
    src/oskar_mem_copy.c
    src/oskar_mem_copy_contents.c
    src/oskar_mem_create_alias_from_raw.c
    src/oskar_mem_create_alias.c
    src/oskar_mem_create_copy.c
    src/oskar_mem_create.c
    src/oskar_mem_data_type_string.c
    src/oskar_mem_different.c
    src/oskar_mem_element_size.c
    src/oskar_mem_ensure.c
    src/oskar_mem_evaluate_relative_error.c
    src/oskar_mem_free.c
    src/oskar_mem_get_element.c
    src/oskar_mem_load_ascii.c
    src/oskar_mem_multiply.c
    src/oskar_mem_normalise.c
    src/oskar_mem_random_gaussian.c
    src/oskar_mem_random_range.c
    src/oskar_mem_random_uniform.c
    src/oskar_mem_read_binary_raw.c
    src/oskar_mem_read_element.c
    src/oskar_mem_read_fits.c
    src/oskar_mem_read_fits_image_plane.c
    src/oskar_mem_read_healpix_fits.c
    src/oskar_mem_realloc.c
    src/oskar_mem_save_ascii.c
    src/oskar_mem_scale_real.c
    src/oskar_mem_set_alias.c
    src/oskar_mem_set_element.c
    src/oskar_mem_set_value_real.c
    src/oskar_mem_stats.c
    src/oskar_mem_write_fits_cube.c
    src/oskar_mem_write_healpix_fits.c
    src/oskar_mem_cpu.cl
    src/oskar_mem_gpu.cl
    src/oskar_mem.cl
)

if (CUDA_FOUND)
    list(APPEND mem_SRC
        src/oskar_mem_random_gaussian_cuda.cu
        src/oskar_mem_random_uniform_cuda.cu
        src/oskar_mem.cu
    )
endif()

if (OpenCL_FOUND)
    list(APPEND mem_SRC
        src/oskar_mem_random.cl
        src/oskar_random_generators.cl
    )
endif()

set(mem_SRC "${mem_SRC}" PARENT_SCOPE)

# === Recurse into test directory.
if (BUILD_TESTING OR NOT DEFINED BUILD_TESTING)
    add_subdirectory(test)
endif()

