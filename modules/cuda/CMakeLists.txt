#
# src/modules/cuda/CMakeLists.txt
#
#==============================================================================

# === Create list of source files for the oskar-lib cuda module.
set(modules_cuda_SRC
    src/oskar_modules_cuda_correlator_lm_bw.cu
)
set(modules_cuda_SRC "${modules_cuda_SRC}" PARENT_SCOPE)

# === Recurse into test subdirectory.
add_subdirectory(test)
