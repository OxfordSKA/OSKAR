#
# oskar/telescope/station/element/CMakeLists.txt
#

set(element_SRC
    define_apply_element_taper_cosine.h
    define_apply_element_taper_gaussian.h
    define_evaluate_dipole_pattern.h
    #define_evaluate_geometric_dipole_pattern.h
    define_evaluate_spherical_wave.h
    define_evaluate_spherical_wave_feko.h
    define_evaluate_spherical_wave_galileo.h
    src/oskar_apply_element_taper_cosine.c
    src/oskar_apply_element_taper_gaussian.c
    src/oskar_element_accessors.c
    src/oskar_element_copy.c
    src/oskar_element_create.c
    src/oskar_element_different.c
    src/oskar_element_evaluate.c
    src/oskar_element_free.c
    src/oskar_element_load_spherical_wave_coeff.c
    src/oskar_element_load_spherical_wave_coeff_feko.c
    src/oskar_element_load_spherical_wave_coeff_galileo.c
    src/oskar_element_resize_freq_data.c
    src/oskar_element.cl
    src/oskar_element.cu
    src/oskar_evaluate_dipole_pattern.c
    #src/oskar_evaluate_geometric_dipole_pattern.c
    src/oskar_evaluate_spherical_wave_sum.c
    src/oskar_evaluate_spherical_wave_sum_feko.c
    src/oskar_evaluate_spherical_wave_sum_galileo.c
)

set(element_SRC "${element_SRC}" PARENT_SCOPE)
