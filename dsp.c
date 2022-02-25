#include <inttypes.h>
#include <math.h>
#include <stdlib.h>

#include "dsp.h"
#include "tables.h"

#include <arm_neon.h>
#include <stdio.h>

static void transpose_block(float *in_data, float *out_data)
{
  float32x4x4_t line01, line23, line45, line67;
  float32x4x2_t tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
  float32x4x2_t c0, c1, c2, c3, c4, c5, c6, c7;

  // Read 2 rows at a time, each row covering 2 vectors. The data read is interleaved
  // between the 4 vectors in each variable
  line01 = vld4q_f32(in_data);
  line23 = vld4q_f32(in_data + 16);
  line45 = vld4q_f32(in_data + 32);
  line67 = vld4q_f32(in_data + 48);

  tmp0 = vuzpq_f32(line01.val[0], line45.val[0]);
  tmp1 = vuzpq_f32(line01.val[1], line45.val[1]);
  tmp2 = vuzpq_f32(line01.val[2], line45.val[2]);
  tmp3 = vuzpq_f32(line01.val[3], line45.val[3]);

  tmp4 = vuzpq_f32(line23.val[0], line67.val[0]);
  tmp5 = vuzpq_f32(line23.val[1], line67.val[1]);
  tmp6 = vuzpq_f32(line23.val[2], line67.val[2]);
  tmp7 = vuzpq_f32(line23.val[3], line67.val[3]);

  c0 = vtrnq_f32(tmp0.val[0], tmp4.val[0]);
  c1 = vtrnq_f32(tmp1.val[0], tmp5.val[0]);
  c2 = vtrnq_f32(tmp2.val[0], tmp6.val[0]);
  c3 = vtrnq_f32(tmp3.val[0], tmp7.val[0]);
  c4 = vtrnq_f32(tmp0.val[1], tmp4.val[1]);
  c5 = vtrnq_f32(tmp1.val[1], tmp5.val[1]);
  c6 = vtrnq_f32(tmp2.val[1], tmp6.val[1]);
  c7 = vtrnq_f32(tmp3.val[1], tmp7.val[1]);

  vst2q_f32(out_data, c0);
  vst2q_f32(out_data + 8, c1);
  vst2q_f32(out_data + 16, c2);
  vst2q_f32(out_data + 24, c3);
  vst2q_f32(out_data + 32, c4);
  vst2q_f32(out_data + 40, c5);
  vst2q_f32(out_data + 48, c6);
  vst2q_f32(out_data + 56, c7);
}

static void dct_1d(float *in_data, float *out_data)
{
  float32x4x2_t datav, dctv;
  float32x4_t mul1, mul2;

  // Load the 8 in_data elements
  datav = vld2q_f32(in_data); 

  // Load the dct values from the transposed version of the dctlookup (from tables.h)
  dctv = vld2q_f32(dctlookup_T);

  /* Multiply the in_data with the dctlookup. 
  First vector of first 4 elements then second vector of last 4 elements */
  mul1 = vmulq_f32(datav.val[0], dctv.val[0]);
  mul2 = vmulq_f32(datav.val[1], dctv.val[1]);

  /* We add mul1 and mul2 before doing across vector sum because this way
  we only need to do across vector sum once as opposed to twice.
  This is better because across vector sum is slow (or atleast slower than
  vector add) */
  mul1 = vaddq_f32(mul1, mul2);
  out_data[0] = vaddvq_f32(mul1);

  // Repeat 7 times. The loop is unrolled
  dctv = vld2q_f32(dctlookup_T + 8);
  mul1 = vmulq_f32(datav.val[0], dctv.val[0]);
  mul2 = vmulq_f32(datav.val[1], dctv.val[1]);
  mul1 = vaddq_f32(mul1, mul2);
  out_data[1] = vaddvq_f32(mul1);

  dctv = vld2q_f32(dctlookup_T + 16);
  mul1 = vmulq_f32(datav.val[0], dctv.val[0]);
  mul2 = vmulq_f32(datav.val[1], dctv.val[1]);
  mul1 = vaddq_f32(mul1, mul2);
  out_data[2] = vaddvq_f32(mul1);

  dctv = vld2q_f32(dctlookup_T + 24);
  mul1 = vmulq_f32(datav.val[0], dctv.val[0]);
  mul2 = vmulq_f32(datav.val[1], dctv.val[1]);
  mul1 = vaddq_f32(mul1, mul2);
  out_data[3] = vaddvq_f32(mul1);

  dctv = vld2q_f32(dctlookup_T + 32);
  mul1 = vmulq_f32(datav.val[0], dctv.val[0]);
  mul2 = vmulq_f32(datav.val[1], dctv.val[1]);
  mul1 = vaddq_f32(mul1, mul2);
  out_data[4] = vaddvq_f32(mul1);

  dctv = vld2q_f32(dctlookup_T + 40);
  mul1 = vmulq_f32(datav.val[0], dctv.val[0]);
  mul2 = vmulq_f32(datav.val[1], dctv.val[1]);
  mul1 = vaddq_f32(mul1, mul2);
  out_data[5] = vaddvq_f32(mul1);

  dctv = vld2q_f32(dctlookup_T + 48);
  mul1 = vmulq_f32(datav.val[0], dctv.val[0]);
  mul2 = vmulq_f32(datav.val[1], dctv.val[1]);
  mul1 = vaddq_f32(mul1, mul2);
  out_data[6] = vaddvq_f32(mul1);

  dctv = vld2q_f32(dctlookup_T + 56);
  mul1 = vmulq_f32(datav.val[0], dctv.val[0]);
  mul2 = vmulq_f32(datav.val[1], dctv.val[1]);
  mul1 = vaddq_f32(mul1, mul2);
  out_data[7] = vaddvq_f32(mul1);
}

static void idct_1d(float *in_data, float *out_data)
{
  float32x4x2_t datav, dctv;
  float32x4_t mul1, mul2;

  // Load the 8 in_data elements
  datav = vld2q_f32(in_data); 

  // Load the dct values from the dctlookup (from tables.h)
  dctv = vld2q_f32(dctlookup);

  /* Multiply the in_data with the dctlookup. 
  First vector of first 4 elements then second vector of last 4 elements */
  mul1 = vmulq_f32(datav.val[0], dctv.val[0]);
  mul2 = vmulq_f32(datav.val[1], dctv.val[1]);

  // We add mul1 and mul2 before doing across vector sum because this way
  // we only need to do across vector sum once as opposed to twice.
  // This is better because across vector sum is slow (or atleast slower than
  // vector add)
  mul1 = vaddq_f32(mul1, mul2);
  out_data[0] = vaddvq_f32(mul1);

  dctv = vld2q_f32(dctlookup + 8);
  mul1 = vmulq_f32(datav.val[0], dctv.val[0]);
  mul2 = vmulq_f32(datav.val[1], dctv.val[1]);
  mul1 = vaddq_f32(mul1, mul2);
  out_data[1] = vaddvq_f32(mul1);

  dctv = vld2q_f32(dctlookup + 16);
  mul1 = vmulq_f32(datav.val[0], dctv.val[0]);
  mul2 = vmulq_f32(datav.val[1], dctv.val[1]);
  mul1 = vaddq_f32(mul1, mul2);
  out_data[2] = vaddvq_f32(mul1);

  dctv = vld2q_f32(dctlookup + 24);
  mul1 = vmulq_f32(datav.val[0], dctv.val[0]);
  mul2 = vmulq_f32(datav.val[1], dctv.val[1]);
  mul1 = vaddq_f32(mul1, mul2);
  out_data[3] = vaddvq_f32(mul1);

  dctv = vld2q_f32(dctlookup + 32);
  mul1 = vmulq_f32(datav.val[0], dctv.val[0]);
  mul2 = vmulq_f32(datav.val[1], dctv.val[1]);
  mul1 = vaddq_f32(mul1, mul2);
  out_data[4] = vaddvq_f32(mul1);

  dctv = vld2q_f32(dctlookup + 40);
  mul1 = vmulq_f32(datav.val[0], dctv.val[0]);
  mul2 = vmulq_f32(datav.val[1], dctv.val[1]);
  mul1 = vaddq_f32(mul1, mul2);
  out_data[5] = vaddvq_f32(mul1);

  dctv = vld2q_f32(dctlookup + 48);
  mul1 = vmulq_f32(datav.val[0], dctv.val[0]);
  mul2 = vmulq_f32(datav.val[1], dctv.val[1]);
  mul1 = vaddq_f32(mul1, mul2);
  out_data[6] = vaddvq_f32(mul1);

  dctv = vld2q_f32(dctlookup + 56);
  mul1 = vmulq_f32(datav.val[0], dctv.val[0]);
  mul2 = vmulq_f32(datav.val[1], dctv.val[1]);
  mul1 = vaddq_f32(mul1, mul2);
  out_data[7] = vaddvq_f32(mul1);
}

static void scale_block_inplace(float *in_data)
{
  /* The original scale block did a lot of redundant computation. Most of the
  elements were being multiplied by 1.0f, which is the same as doing nothing.
  This is faster since we only do computation on the indices that are multiplied
  by 0.7f. This includes the first row and the first column, with index [0][0] being
  multiplied by 0.7 twice (0.7^2).

  The reason to do this inplace is to avoid having to copy over all the unchanged
  elements to the out_data block.
  The column is not aligned in memory and must therefore be processed sequentially. */

  // Load the first row 4x2 float32 elements
  float32x4x2_t datav = vld2q_f32(in_data);
  // Create a vector {0.7f 0.7f 0.7f 0.7f}
  float32x4_t isqrt2v = vdupq_n_f32(ISQRT2);

  // Multiply both vectors of the first row by 0.7f
  datav.val[0] = vmulq_f32(datav.val[0], isqrt2v);
  datav.val[1] = vmulq_f32(datav.val[1], isqrt2v);

  // Write the updated row back into the array
  vst2q_f32(in_data, datav);
 
  // Process the first column sequentially
  in_data[0] *= ISQRT2;
  in_data[8] *= ISQRT2;
  in_data[16] *= ISQRT2;
  in_data[24] *= ISQRT2;
  in_data[32] *= ISQRT2;
  in_data[40] *= ISQRT2;
  in_data[48] *= ISQRT2;
  in_data[56] *= ISQRT2;
}

static void scale_block_row(float *in_data)
{
  /* This function multiplies each element in the first row of the in_data block
  by 0.7. In dct_quant_block_8x8, the mb data is transposed before being scaled,
  so we can then scale the first row first, then do the transpose, then scale the
  first row again in place. This way we get to scale both the row and column using
  SIMD, since the transpose makes the row we scale first into the column. */

  // Create a vector with the isqrt2 value in each lane
  float32x4_t isqrt2v = vdupq_n_f32(ISQRT2);

  // Load the in_data row
  float32x4x2_t datav = vld2q_f32(in_data);

  // Multiply both vectors containing the first 4 and last 4 elements making up the row
  datav.val[0] = vmulq_f32(datav.val[0], isqrt2v);
  datav.val[1] = vmulq_f32(datav.val[1], isqrt2v);

  // Store the processed data back into the array
  vst2q_f32(in_data, datav);
}

static void quantize_block(float *in_data, float *out_data, uint8_t *quant_tbl)
{
  int zigzag;

  for (zigzag = 0; zigzag < 64; zigzag++)
  {
    uint8_t u = zigzag_U[zigzag];
    uint8_t v = zigzag_V[zigzag];

    float dct = in_data[v*8+u];

    // Zig-zag and quantize
    out_data[zigzag] = (float) round((dct / 4.0) / quant_tbl[zigzag]);
  }
}

static void dequantize_block(float *in_data, float *out_data,
    uint8_t *quant_tbl)
{
  int zigzag;

  for (zigzag = 0; zigzag < 64; zigzag++)
  {
    uint8_t u = zigzag_U[zigzag];
    uint8_t v = zigzag_V[zigzag];

    float dct = in_data[zigzag];

    /* Zig-zag and de-quantize */
    out_data[v*8+u] = (float) round((dct * quant_tbl[zigzag]) / 4.0);
  }
}

void int16_to_float32(int16_t *in_data, float *out_data) {
  float32x4x4_t st;
  for (int i = 0; i < 64; i+=16) {
    int16x4x4_t ld = vld4_s16(in_data + i);
    int32x4_t v1 = vmovl_s16(ld.val[0]);
    int32x4_t v2 = vmovl_s16(ld.val[1]);
    int32x4_t v3 = vmovl_s16(ld.val[2]);
    int32x4_t v4 = vmovl_s16(ld.val[3]);

    st.val[0] = vcvtq_f32_s32(v1);
    st.val[1] = vcvtq_f32_s32(v2);
    st.val[2] = vcvtq_f32_s32(v3);
    st.val[3] = vcvtq_f32_s32(v4);
    vst4q_f32(out_data + i, st);
  }
}

void float32_to_int16(float *in_data, int16_t *out_data) {
  int16x4x4_t st;
  for (int i = 0; i < 64; i+=16) {
    float32x4x4_t ld = vld4q_f32(in_data + i);
    int32x4_t v1 = vcvtq_s32_f32(ld.val[0]);
    int32x4_t v2 = vcvtq_s32_f32(ld.val[1]);
    int32x4_t v3 = vcvtq_s32_f32(ld.val[2]);
    int32x4_t v4 = vcvtq_s32_f32(ld.val[3]);

    st.val[0] = vmovn_s32(v1);
    st.val[1] = vmovn_s32(v2);
    st.val[2] = vmovn_s32(v3);
    st.val[3] = vmovn_s32(v4);
    vst4_s16(out_data + i, st);
  }
}

void dct_quant_block_8x8(int16_t *in_data, int16_t *out_data,
    uint8_t *quant_tbl)
{
  float mb[8*8] __attribute((aligned(16)));
  float mb2[8*8] __attribute((aligned(16)));

  // Convert the in_data from int16 to float32
  int16_to_float32(in_data, mb2);  

  /* Two 1D DCT operations with transpose */
  dct_1d(mb2, mb);
  dct_1d(mb2 + 8, mb + 8);
  dct_1d(mb2 + 16, mb + 16);
  dct_1d(mb2 + 24, mb + 24);
  dct_1d(mb2 + 32, mb + 32);
  dct_1d(mb2 + 40, mb + 40);
  dct_1d(mb2 + 48, mb + 48);
  dct_1d(mb2 + 56, mb + 56);

  transpose_block(mb, mb2);

  dct_1d(mb2, mb);
  dct_1d(mb2 + 8, mb + 8);
  dct_1d(mb2 + 16, mb + 16);
  dct_1d(mb2 + 24, mb + 24);
  dct_1d(mb2 + 32, mb + 32);
  dct_1d(mb2 + 40, mb + 40);
  dct_1d(mb2 + 48, mb + 48);
  dct_1d(mb2 + 56, mb + 56);

  // Directly do a2 scaling, then a1 scaling after transpose.
  // This way we only scale one float32x8 row two times, allowing us to
  // use intrinsics for both rows, and save redundant computation
  // by eliminating all the other elements that are just being multiplied by 1
  scale_block_row(mb);
  transpose_block(mb, mb2);
  scale_block_row(mb2);

  quantize_block(mb2, mb, quant_tbl);

  // Convert back from float32 to int16 
  float32_to_int16(mb, out_data);
}

void dequant_idct_block_8x8(int16_t *in_data, int16_t *out_data,
    uint8_t *quant_tbl)
{
  float mb[8*8] __attribute((aligned(16)));
  float mb2[8*8] __attribute((aligned(16)));

  // Convert the in_data from int16 to float32
  int16_to_float32(in_data, mb);  

  dequantize_block(mb, mb2, quant_tbl);

  /* We do not have a convenient lossless transpose here, so we have to scale
  both the row and the column here. We do it inplace to avoid having to copy over
  most of the data, since only one row and one column is being changed */
  scale_block_inplace(mb2);

  /* Two 1D inverse DCT operations with transpose */
  idct_1d(mb2, mb);
  idct_1d(mb2 + 8, mb + 8);
  idct_1d(mb2 + 16, mb + 16);
  idct_1d(mb2 + 24, mb + 24);
  idct_1d(mb2 + 32, mb + 32);
  idct_1d(mb2 + 40, mb + 40);
  idct_1d(mb2 + 48, mb + 48);
  idct_1d(mb2 + 56, mb + 56);

  transpose_block(mb, mb2);

  idct_1d(mb2, mb);
  idct_1d(mb2 + 8, mb + 8);
  idct_1d(mb2 + 16, mb + 16);
  idct_1d(mb2 + 24, mb + 24);
  idct_1d(mb2 + 32, mb + 32);
  idct_1d(mb2 + 40, mb + 40);
  idct_1d(mb2 + 48, mb + 48);
  idct_1d(mb2 + 56, mb + 56);

  transpose_block(mb, mb2);

  // Convert back from float32 to int16 
  float32_to_int16(mb2, out_data);
}
