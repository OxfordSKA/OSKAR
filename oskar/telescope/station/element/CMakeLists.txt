#
# oskar/telescope/station/element/CMakeLists.txt
#

set(element_SRC
    src/oskar_apply_element_taper_cosine.c
    src/oskar_apply_element_taper_gaussian.c
    src/oskar_element_accessors.c
    src/oskar_element_copy.c
    src/oskar_element_create.c
    src/oskar_element_different.c
    src/oskar_element_evaluate.c
    src/oskar_element_free.c
    src/oskar_element_load.c
    src/oskar_element_load_cst.c
    src/oskar_element_load_scalar.c
    src/oskar_element_read.c
    src/oskar_element_resize_freq_data.c
    src/oskar_element_save.c
    src/oskar_element_write.c
    src/oskar_evaluate_dipole_pattern.c
    src/oskar_evaluate_geometric_dipole_pattern.c)

if (CUDA_FOUND)
    list(APPEND element_SRC
        src/oskar_apply_element_taper_cosine_cuda.cu
        src/oskar_apply_element_taper_gaussian_cuda.cu
        src/oskar_evaluate_dipole_pattern_cuda.cu
        src/oskar_evaluate_geometric_dipole_pattern_cuda.cu)
endif()

set(element_SRC "${element_SRC}" PARENT_SCOPE)
