#ifndef OSKAR__ROUND_ROBIN_H
#define OSKAR__ROUND_ROBIN_H

#include "cuda/CudaEclipse.h"

/**
 * @file _roundRobin.h
 */

__device__
void _roundRobin(unsigned items, unsigned resources, unsigned rank,
        unsigned& number, unsigned& start);

#endif // OSKAR__ROUND_ROBIN_H
