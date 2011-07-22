#include "oskar_random_bpl.h"
#include "Random.h"

extern "C"
void oskar_random_bpl(int n, double min, double max, double threshold, double power1,
        double power2, int seed, double * values)
{
    values[0] = Random::broken_power_law<double>(min, max, threshold, power1,
            power2, seed);

    for (int i = 1; i < n; ++i)
    {
        values[i] = Random::broken_power_law<double>(min, max,
                threshold, power1, power2);
    }
}
