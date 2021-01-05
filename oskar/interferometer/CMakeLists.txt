#
# oskar/interferometer/CMakeLists.txt
#

set(interferometer_SRC
    define_jones_apply_station_gains.h
    define_evaluate_jones_K.h
    define_evaluate_jones_R.h
    src/oskar_evaluate_jones_E.c
    src/oskar_evaluate_jones_K.c
    src/oskar_evaluate_jones_R.c
    #src/oskar_evaluate_jones_Z.c
    src/oskar_interferometer_accessors.c
    src/oskar_interferometer_check_init.c
    src/oskar_interferometer_create.c
    src/oskar_interferometer_finalise_block.c
    src/oskar_interferometer_finalise.c
    src/oskar_interferometer_free.c
    src/oskar_interferometer_run_block.c
    src/oskar_interferometer_run.c
    src/oskar_interferometer_write_block.c
    src/oskar_interferometer_cpu.cl
    src/oskar_interferometer_gpu.cl
    src/oskar_interferometer.cl
    src/oskar_jones_accessors.c
    src/oskar_jones_apply_station_gains.c
    src/oskar_jones_create.c
    src/oskar_jones_create_copy.c
    src/oskar_jones_free.c
    src/oskar_jones_join.c
    src/oskar_jones_set_size.c
    #src/oskar_WorkJonesZ.c
)

if (CUDA_FOUND)
    list(APPEND interferometer_SRC src/oskar_interferometer.cu)
endif()

set(interferometer_SRC "${interferometer_SRC}" PARENT_SCOPE)

if (BUILD_TESTING OR NOT DEFINED BUILD_TESTING)
    add_subdirectory(test)
endif()
