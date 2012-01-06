#
# src/extern/dierckx/CMakeLists.txt
#

set(dierckx_SRC
    src/bispev.c
    src/bispev.f
    src/fpback.f
    src/fpbisp.c
    src/fpbisp.f
    src/fpbspl.c
    src/fpbspl.f
    src/fpchec.f
    src/fpdisc.f
    src/fpgivs.f
    src/fpgrre.f
    src/fpknot.f
    src/fporde.f
    src/fprank.f
    src/fprati.f
    src/fpregr.f
    src/fprota.f
    src/fprpsp.f
    src/fpsphe.c
    src/fpsphe.f
    src/regrid.f
    src/sphere.c
    src/sphere.f
)

add_library(dierckx SHARED ${dierckx_SRC})
install(TARGETS dierckx DESTINATION ${OSKAR_LIB_INSTALL_DIR})

add_subdirectory(test)

