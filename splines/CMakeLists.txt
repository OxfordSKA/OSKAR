#
# src/splines/CMakeLists.txt
#

# Add the source files in this directory.
set(splines_SRC
    src/oskar_dierckx_bispev.c
    src/oskar_dierckx_fpback.c
    src/oskar_dierckx_fpbisp.c
    src/oskar_dierckx_fpbspl.c
    src/oskar_dierckx_fpdisc.c
    src/oskar_dierckx_fpgivs.c
    src/oskar_dierckx_fporde.c
    src/oskar_dierckx_fprank.c
    src/oskar_dierckx_fprati.c
    src/oskar_dierckx_fprota.c
    src/oskar_dierckx_fprpsp.c
    src/oskar_dierckx_fpsphe.c
    src/oskar_dierckx_fpsurf.c
    src/oskar_dierckx_sphere.c
    src/oskar_dierckx_surfit.c
    src/oskar_splines_accessors.c
    src/oskar_splines_copy.c
    src/oskar_splines_create.c
    src/oskar_splines_evaluate.c
    src/oskar_splines_fit.c
    src/oskar_splines_free.c
)

if (CUDA_FOUND)
    list(APPEND splines_SRC
        src/oskar_dierckx_bispev_cuda.cu
        src/oskar_dierckx_bispev_bicubic_cuda.cu
    )
endif()

set(splines_SRC "${splines_SRC}" PARENT_SCOPE)
