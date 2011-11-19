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

#ifndef OSKAR_WORK_H_
#define OSKAR_WORK_H_

/**
 * @file oskar_Work.h
 */

#include "oskar_global.h"
#include "utility/oskar_Mem.h"

#ifdef __cplusplus
extern "C"
#endif
struct oskar_Work
{
    oskar_Mem integer;
	/* These are all either double or single. */
    oskar_Mem real;
    oskar_Mem complex;
    oskar_Mem matrix;

#ifdef __cplusplus
    /**
     * @brief Constructor.
     *
     * @param[in] type     OSKAR memory type ID (Accepted values: OSKAR_SINGLE,
     *                     OSKAR_DOUBLE).
     * @param[in] location OSKAR memory location ID.
     */
    oskar_Work(int type, int location);

    /**
     * @brief Constructs an oskar_Work structure as a copy of another oskar_Work
     * structure.
     *
     * @param other     oskar_Work structure to copy.
     * @param location  Memory location to copy to.
     * @param owner     Bool flag specifying if the structure should
     *                  take ownership of the memory.
     */
    oskar_Work(const oskar_Work* other, int location, int owner = 1);

    /**
     * @brief Destructor.
     */
    ~oskar_Work();
#endif
};

typedef struct oskar_Work oskar_Work;

#endif /* OSKAR_WORK_H_ */
