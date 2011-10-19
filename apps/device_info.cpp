#include <cuda_runtime_api.h>
#include <cstdlib>
#include <cstdio>

int main(int /*argc*/, char** /*argv*/)
{
    size_t free, total;
    cudaError_t error = cudaMemGetInfo(&free, &total);
    printf("err = %i, free = %lu, total = %lu\n", error, (unsigned long)free,
            (unsigned long)total);
    return EXIT_SUCCESS;
}



