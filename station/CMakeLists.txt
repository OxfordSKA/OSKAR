#
# src/station/CMakeLists.txt
#

set(station_SRC
    src/oskar_beamPattern.cpp
    #src/oskar_bf_cuda_ant_sig_iso_2d.cu
)
set(station_SRC "${station_SRC}" PARENT_SCOPE)


# ==== Recurse into test subdirectory.
add_subdirectory(test)
