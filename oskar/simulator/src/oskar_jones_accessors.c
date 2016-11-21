/*
 * Copyright (c) 2013, The University of Oxford
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

#include <private_jones.h>

#include <oskar_jones_accessors.h>
#include <oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_jones_num_sources(const oskar_Jones* jones)
{
    return jones->num_sources;
}

int oskar_jones_num_stations(const oskar_Jones* jones)
{
    return jones->num_stations;
}

int oskar_jones_type(const oskar_Jones* jones)
{
    return oskar_mem_type(jones->data);
}

int oskar_jones_mem_location(const oskar_Jones* jones)
{
    return oskar_mem_location(jones->data);
}

oskar_Mem* oskar_jones_mem(oskar_Jones* jones)
{
    return jones->data;
}

const oskar_Mem* oskar_jones_mem_const(const oskar_Jones* jones)
{
    return jones->data;
}

/* Single precision. */

float2* oskar_jones_float2(oskar_Jones* jones, int* status)
{
    return oskar_mem_float2(jones->data, status);
}

const float2* oskar_jones_float2_const(const oskar_Jones* jones, int* status)
{
    return oskar_mem_float2_const(jones->data, status);
}

float4c* oskar_jones_float4c(oskar_Jones* jones, int* status)
{
    return oskar_mem_float4c(jones->data, status);
}

const float4c* oskar_jones_float4c_const(const oskar_Jones* jones, int* status)
{
    return oskar_mem_float4c_const(jones->data, status);
}

/* Double precision. */

double2* oskar_jones_double2(oskar_Jones* jones, int* status)
{
    return oskar_mem_double2(jones->data, status);
}

const double2* oskar_jones_double2_const(const oskar_Jones* jones, int* status)
{
    return oskar_mem_double2_const(jones->data, status);
}

double4c* oskar_jones_double4c(oskar_Jones* jones, int* status)
{
    return oskar_mem_double4c(jones->data, status);
}

const double4c* oskar_jones_double4c_const(const oskar_Jones* jones,
        int* status)
{
    return oskar_mem_double4c_const(jones->data, status);
}

#ifdef __cplusplus
}
#endif
