#
# src/beamforming/CMakeLists.txt
#

set(beamforming_SRC
    src/oskar_beamPattern.cpp
    #src/oskar_bf_cuda_ant_sig_iso_2d.cu
)
set(beamforming_SRC "${beamforming_SRC}" PARENT_SCOPE)


# ==== Recurse into test subdirectory.
add_subdirectory(test)
