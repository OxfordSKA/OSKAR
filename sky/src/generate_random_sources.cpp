#include "sky/generate_random_sources.h"

#include <cstdlib>
#include <iostream>
using namespace std;

void generate_random_sources(const unsigned num_sources, const double inner_radius,
        const double outer_radius, const double brightness_min,
        const double brightness_max, const double distribution_power,
        double * ra, double * dec, double * brightness, const unsigned seed)
{
    // seed the random number generator
    srand(seed);

    for (unsigned i = 0; i < num_sources; ++i)
    {
        const double r = (double)rand() / (double)RAND_MAX;
//        cout << i << " " <<  r << endl;


    }


//    x0 = min
//    x1 = max
//    x = brightness

//    x = [(x1^(n+1) - x0^(n+1))*y + x0^(n+1)]^(1/(n+1));


}
