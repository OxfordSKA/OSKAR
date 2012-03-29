#
# src/extern/dierckx/CMakeLists.txt
#

set(dierckx_SRC
    src/bispev.c
    src/fpback.c
    src/fpbisp.c
    src/fpbspl.c
    src/fpdisc.c
    src/fpgivs.c
    src/fporde.c
    src/fprank.c
    src/fprati.c
    src/fprota.c
    src/fprpsp.c
    src/fpsphe.c
    src/fpsurf.c
    src/sphere.c
    src/surfit.c

    src/bispev.f
    src/fpback.f
    src/fpbisp.f
    src/fpbspl.f
    src/fpchec.f # Deprecated
    src/fpdisc.f
    src/fpgivs.f
    src/fpgrre.f # Deprecated
    src/fpknot.f # Deprecated
    src/fporde.f
    src/fprank.f
    src/fprati.f
    src/fpregr.f # Deprecated
    src/fprota.f
    src/fprpsp.f
    src/fpsphe.f
    src/fpsurf.f
    src/regrid.f # Deprecated
    src/sphere.f
    src/surfit.f
)

add_library(dierckx SHARED ${dierckx_SRC})
install(TARGETS dierckx DESTINATION ${OSKAR_LIB_INSTALL_DIR})

add_subdirectory(test)
