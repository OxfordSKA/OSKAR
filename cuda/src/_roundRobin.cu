#include "cuda/_roundRobin.h"

/**
 * @details
 * Distributes a number of \p items among a number of \p resources
 * (e.g. threads, or processes), and returns the \p number of items
 * and \p start item index (zero-based) for the current resource \p rank
 * (also zero-based).
 *
 * @param[in] items The number of items to distribute.
 * @param[in] resources The number of resources available.
 * @param[in] rank The index of this resource.
 * @param[out] number The number of items that this resource must process.
 * @param[out] start The start item index that this resource must process.
 */
__device__
void _roundRobin(unsigned items, unsigned resources, unsigned rank,
        unsigned& number, unsigned& start)
{
    number = items / resources;
    unsigned remainder = items % resources;
    if (rank < remainder) number++;
    start = number * rank;
    if (rank >= remainder) start += remainder;
}
