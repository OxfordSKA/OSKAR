/*
 * Copyright (c) 2011, The University of Oxford
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the University of Oxford nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */


#ifndef OSKAR_WORKE_H_
#define OSKAR_WORKE_H_

/**
 * @file oskar_WorkE.h
 */


/**
 * @struct oskar_WorkE
 *
 * @brief Structure holding work buffers used in the evaluating E-Jones
 * matrices.
 */

struct oskar_WorkE
{
    oskar_Mem hor_l;   /**< Source horizontal l */
    oskar_Mem hor_m;   /**< Source horizontal m */
    oskar_Mem hor_n;   /**< Source horizontal n */
    oskar_Mem weights; /**< Beamforming weights */
    oskar_Mem signal;  /**< Element signals (if required for multi-level processing) */
    double beam_hor_l; /**< Beam phase centre horizontal l */
    double beam_hor_m; /**< Beam phase centre horizontal m */
    double beam_hor_n; /**< Beam phase centre horizontal n */
};
typedef struct oskar_WorkE oskar_WorkE;

#endif /* OSKAR_WORKE_H_ */
