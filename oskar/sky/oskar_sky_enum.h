/*
 * Copyright (c) 2025-2026, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_SKY_ENUM_H_
#define OSKAR_SKY_ENUM_H_

/**
 * @file oskar_sky_enum.h
 */

/* Column types. */
enum oskar_SkyColumn
{
    OSKAR_SKY_CUSTOM = 0, /* Reserved. Currently for anything unknown. */
    /* Fix numbers for original columns. */
    OSKAR_SKY_RA_RAD = 1,
    OSKAR_SKY_DEC_RAD = 2,
    OSKAR_SKY_I_JY = 3,
    OSKAR_SKY_Q_JY = 4,
    OSKAR_SKY_U_JY = 5,
    OSKAR_SKY_V_JY = 6,
    OSKAR_SKY_REF_HZ = 7,
    OSKAR_SKY_SPEC_IDX = 8,
    OSKAR_SKY_RM_RAD = 9,
    OSKAR_SKY_MAJOR_RAD = 10,
    OSKAR_SKY_MINOR_RAD = 11,
    OSKAR_SKY_PA_RAD = 12,
    OSKAR_SKY_LIN_SI, /* Linear spectral index (opposite to LogarithmicSI). */
    OSKAR_SKY_POLA_RAD,
    OSKAR_SKY_POLF,
    OSKAR_SKY_REF_WAVE_M,
    OSKAR_SKY_SPEC_CURV,
    OSKAR_SKY_LINE_WIDTH_HZ,
    /* Keep track of the number of fixed ("real") column types. */
    OSKAR_SKY_NUM_FIXED_COLUMN_TYPES,
    /* The following are "phantom" column types, used only for loading. */
    OSKAR_SKY_RA_DEG,
    OSKAR_SKY_DEC_DEG,
    OSKAR_SKY_SEMI_MAJOR,
    OSKAR_SKY_SEMI_MINOR,
    /* Scratch columns start at 0x100 (256). */
    OSKAR_SKY_SCRATCH_START = 0x100,
    OSKAR_SKY_SCRATCH_EXT_A = OSKAR_SKY_SCRATCH_START,
    OSKAR_SKY_SCRATCH_EXT_B,
    OSKAR_SKY_SCRATCH_EXT_C,
    OSKAR_SKY_SCRATCH_L,
    OSKAR_SKY_SCRATCH_M,
    OSKAR_SKY_SCRATCH_N,
    OSKAR_SKY_SCRATCH_I_JY,
    OSKAR_SKY_SCRATCH_Q_JY,
    OSKAR_SKY_SCRATCH_U_JY,
    OSKAR_SKY_SCRATCH_V_JY,
    OSKAR_SKY_SCRATCH_END
};
typedef enum oskar_SkyColumn oskar_SkyColumn;


/* Static integer attributes. */
enum oskar_SkyAttribInt
{
    OSKAR_SKY_PRECISION,
    OSKAR_SKY_MEM_LOCATION,
    OSKAR_SKY_CAPACITY,
    OSKAR_SKY_NUM_SOURCES,
    OSKAR_SKY_NUM_COLUMNS,
    OSKAR_SKY_USE_EXTENDED,
    /* Last value is the number of attributes. */
    OSKAR_SKY_NUM_ATTRIBUTES_INT
};
typedef enum oskar_SkyAttribInt oskar_SkyAttribInt;


/* Static double attributes. */
enum oskar_SkyAttribDouble
{
    OSKAR_SKY_REF_RA_RAD,
    OSKAR_SKY_REF_DEC_RAD,
    /* Last value is the number of attributes. */
    OSKAR_SKY_NUM_ATTRIBUTES_DOUBLE
};
typedef enum oskar_SkyAttribDouble oskar_SkyAttribDouble;

#endif /* include guard */
