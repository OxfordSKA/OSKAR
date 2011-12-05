project(dierckx Fortran C CXX)
#cmake_minimum_required(VERSION 2.8)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE release)
endif()

set(dierckx_SRC
    bispev.c
    bispev.f
    fpback.f
    fpbisp.c
    fpbisp.f
    fpbspl.c
    fpbspl.f
    fpchec.f
    fpdisc.f
    fpgivs.f
    fpgrre.f
    fpknot.f
    fprati.f
    fpregr.f
    fprota.f
    regrid.f
)

set(CMAKE_Fortran_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_Fortran_FLAGS_DEBUG "-g -O0")
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_C_FLAGS_DEBUG "-g -O0")

add_library(dierckx SHARED ${dierckx_SRC})

add_executable(test dierckx_test.cpp)
target_link_libraries(test dierckx)

