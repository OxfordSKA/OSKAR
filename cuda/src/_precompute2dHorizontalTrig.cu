#include "cuda/_precompute2dHorizontalTrig.h"

/**
 * @details
 * This CUDA kernel precomputes the required trigonometry for the given
 * source positions in (azimuth, elevation).
 *
 * Each thread operates on a single source.
 *
 * The source positions are specified as (azimuth, elevation) pairs in the
 * \p spos array:
 *
 * spos.x = {azimuth}
 * spos.y = {elevation}
 *
 * The output \p strig array contains triplets of the following pre-computed
 * trigonometry:
 *
 * strig.x = {cosine azimuth}
 * strig.y = {sine azimuth}
 * strig.z = {cosine elevation}
 *
 * This kernel can be used to prepare a source distribution to generate
 * antenna signals for a 2D antenna array.
 *
 * @param[in] ns The number of source positions.
 * @param[in] spos The azimuth and elevation source coordinates in radians.
 * @param[out] strig The cosine and sine of the source azimuth and elevation
 *                   coordinates.
 */
__global__
void _precompute2dHorizontalTrig(const int ns, const float2* spos, float3* strig)
{
    // Get the source ID that this thread is working on.
    const int s = blockDim.x * blockIdx.x + threadIdx.x;
    if (s >= ns) return; // Return if the index is out of range.

    strig[s].x = cosf(spos[s].x);
    strig[s].y = sinf(spos[s].x);
    strig[s].z = cosf(spos[s].y);
}
