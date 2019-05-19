#
# oskar/splines/CMakeLists.txt
#

set(splines_SRC
    define_dierckx_bispev_bicubic.h
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
    src/oskar_splines.c
    src/oskar_splines_evaluate.c
    src/oskar_splines_fit.c
    src/oskar_splines.cl
)

if (CUDA_FOUND)
    list(APPEND splines_SRC src/oskar_splines.cu)
endif()

set(splines_SRC "${splines_SRC}" PARENT_SCOPE)
