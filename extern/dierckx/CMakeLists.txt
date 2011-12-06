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
    src/fprati.f
    src/fpregr.f
    src/fprota.f
    src/regrid.f
)

add_library(dierckx SHARED ${dierckx_SRC})
install(TARGETS dierckx DESTINATION ${OSKAR_LIB_INSTALL_DIR})

add_executable(oskar_fit_surface src/oskar_fit_surface.cpp)
target_link_libraries(oskar_fit_surface dierckx)
install(TARGETS oskar_fit_surface DESTINATION ${OSKAR_BIN_INSTALL_DIR})

add_subdirectory(test)

