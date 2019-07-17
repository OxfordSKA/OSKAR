
set(cfitsio_SRC
    buffers.c cfileio.c checksum.c
    drvrfile.c
    drvrmem.c
    #drvrnet.c
    #drvrsmem.c
    #drvrgsiftp.c
    editcol.c edithdu.c eval_l.c
    eval_y.c eval_f.c fitscore.c getcol.c getcolb.c getcold.c getcole.c
    getcoli.c getcolj.c getcolk.c getcoll.c getcols.c getcolsb.c
    getcoluk.c getcolui.c getcoluj.c getkey.c group.c grparser.c
    histo.c iraffits.c
    modkey.c putcol.c putcolb.c putcold.c putcole.c putcoli.c
    putcolj.c putcolk.c putcoluk.c putcoll.c putcols.c putcolsb.c
    putcolu.c putcolui.c putcoluj.c putkey.c region.c scalnull.c
    swapproc.c wcssub.c wcsutil.c imcompress.c quantize.c ricecomp.c
    pliocomp.c fits_hcompress.c fits_hdecompress.c zlib/zuncompress.c
    zlib/zcompress.c zlib/adler32.c zlib/crc32.c zlib/inffast.c
    zlib/inftrees.c zlib/trees.c zlib/zutil.c zlib/deflate.c
    zlib/infback.c zlib/inflate.c zlib/uncompr.c simplerng.c
    f77_wrap1.c f77_wrap2.c f77_wrap3.c f77_wrap4.c
)

if (NOT WIN32)
    set(CFITSIO_FLAGS_RELEASE "-O3 -DNDEBUG -fPIC")
    set(CFITSIO_FLAGS_DEBUG "-g -fPIC")
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
if (WIN32)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS -DFF_NO_UNISTD_H)
endif()
