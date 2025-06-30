
set(cfitsio_SRC
    buffers.c cfileio.c checksum.c
    drvrfile.c
    drvrmem.c
    #drvrnet.c
    #drvrsmem.c
    #drvrgsiftp.c
    editcol.c edithdu.c eval_f.c eval_l.c eval_y.c
    #f77_wrap1.c f77_wrap2.c f77_wrap3.c f77_wrap4.c
    fits_hcompress.c fits_hdecompress.c fitscore.c
    getcol.c getcolb.c getcold.c getcole.c getcoli.c getcolj.c getcolk.c
    getcoll.c getcols.c getcolsb.c getcolui.c getcoluj.c getcoluk.c
    getkey.c group.c grparser.c histo.c imcompress.c iraffits.c modkey.c
    pliocomp.c
    putcol.c putcolb.c putcold.c putcole.c putcoli.c putcolj.c putcolk.c
    putcoll.c putcols.c putcolsb.c putcolu.c putcolui.c putcoluj.c putcoluk.c
    putkey.c quantize.c region.c ricecomp.c scalnull.c simplerng.c swapproc.c
    wcssub.c wcsutil.c zcompress.c zuncompress.c
)

find_package(ZLIB REQUIRED)

if (NOT WIN32)
    # We need a thread-safe version of this library!
    find_package(Threads REQUIRED)
    include_directories(${PTHREADS_INCLUDE_DIR})
    add_definitions(-D_REENTRANT)

    set(CFITSIO_FLAGS_RELEASE "-O3 -DNDEBUG -fPIC -w")
    set(CFITSIO_FLAGS_DEBUG "-g -fPIC -w")
else()
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
        set(CFITSIO_FLAGS_RELEASE "")
        set(CFITSIO_FLAGS_DEBUG "")
    endif()
endif ()

if (${CMAKE_BUILD_TYPE} MATCHES release)
    set(CFITSIO_FLAGS ${CFITSIO_FLAGS_RELEASE})
else()
    set(CFITSIO_FLAGS ${CFITSIO_FLAGS_DEBUG})
endif()

unset(CMAKE_C_FLAGS)
set(CMAKE_C_FLAGS_DEBUG ${CFITSIO_FLAGS_DEBUG})
set(CMAKE_C_FLAGS_RELEASE ${CFITSIO_FLAGS_RELEASE})
set(CMAKE_C_FLAGS_MINSIZEREL ${CFITSIO_FLAGS_RELEASE})
set(CMAKE_C_FLAGS_RELWITHDEBINFO ${CFITSIO_FLAGS_RELEASE})

add_library(cfitsio STATIC ${cfitsio_SRC})

# Math:
target_link_libraries(cfitsio ${M_LIB})
# zlib:
if (ZLIB_FOUND)
    target_include_directories(cfitsio PUBLIC ${ZLIB_INCLUDE_DIR})
    target_link_libraries(cfitsio ${ZLIB_LIBRARIES})
    # For cfitsio.pc.cmake:
    set(PKG_CONFIG_LIBS "${PKG_CONFIG_LIBS} -lz")
endif()
# PThreads:
if (Threads_FOUND)
    target_link_libraries(cfitsio Threads::Threads)
endif()

if (WIN32)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS -DFF_NO_UNISTD_H)
endif()
