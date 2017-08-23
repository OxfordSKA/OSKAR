#
# oskar/interferometer/CMakeLists.txt
#

set(interferometer_SRC
    src/oskar_evaluate_jones_E.c
    src/oskar_evaluate_jones_K.c
    src/oskar_evaluate_jones_R.c
    src/oskar_evaluate_jones_Z.c
    src/oskar_interferometer.c
    src/oskar_jones_accessors.c
    src/oskar_jones_create.c
    src/oskar_jones_create_copy.c
    src/oskar_jones_free.c
    src/oskar_jones_get_station_pointer.c
    src/oskar_jones_join.c
    src/oskar_jones_set_size.c
    src/oskar_jones_set_real_scalar.c
    src/oskar_WorkJonesZ.c
)

if (CUDA_FOUND)
    list(APPEND interferometer_SRC
        src/oskar_evaluate_jones_K_cuda.cu
        src/oskar_evaluate_jones_R_cuda.cu
    )
endif()

set(interferometer_SRC "${interferometer_SRC}" PARENT_SCOPE)

add_subdirectory(test)
