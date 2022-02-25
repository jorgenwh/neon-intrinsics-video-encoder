#ifndef VEC_PRINT_H_
#define VEC_PRINT_H_

#include <stdio.h>
#include <arm_neon.h>

// --------- INT --------- 

void print_int8x8_t(int8x8_t v) {
  int elems = 8, i;
  int8_t A[elems];
  vst1_s8(A, v);
  printf("int8x%d_t( ", elems);
  for (i = 0; i < elems; i++) { printf("%d ", A[i]); }
  printf(")\n");
}

void print_int8x16_t(int8x16_t v) {
  int elems = 16, i;
  int8_t A[elems];
  vst1q_s8(A, v);
  printf("int8x%d_t( ", elems);
  for (i = 0; i < elems; i++) { printf("%d ", A[i]); }
  printf(")\n");
}

void print_int16x4_t(int16x4_t v) {
  int elems = 4, i;
  int16_t A[elems];
  vst1_s16(A, v);
  printf("int8x%d_t( ", elems);
  for (i = 0; i < elems; i++) { printf("%d ", A[i]); }
  printf(")\n");
}

void print_int16x8_t(int16x8_t v) {
  int elems = 8, i;
  int16_t A[elems];
  vst1q_s16(A, v);
  printf("int8x%d_t( ", elems);
  for (i = 0; i < elems; i++) { printf("%d ", A[i]); }
  printf(")\n");
}

void print_int32x2_t(int32x2_t v) {
  int elems = 2, i;
  int32_t A[elems];
  vst1_s32(A, v);
  printf("int8x%d_t( ", elems);
  for (i = 0; i < elems; i++) { printf("%d ", A[i]); }
  printf(")\n");
}

void print_int32x4_t(int32x4_t v) {
  int elems = 4, i;
  int32_t A[elems];
  vst1q_s32(A, v);
  printf("int8x%d_t( ", elems);
  for (i = 0; i < elems; i++) { printf("%d ", A[i]); }
  printf(")\n");
}

void print_int64x1_t(int64x1_t v) {
  int elems = 1, i;
  int64_t A[elems];
  vst1_s64(A, v);
  printf("int8x%d_t( ", elems);
  for (i = 0; i < elems; i++) { printf("%ld ", A[i]); }
  printf(")\n");
}

void print_int64x2_t(int64x2_t v) {
  int elems = 2, i;
  int64_t A[elems];
  vst1q_s64(A, v);
  printf("int8x%d_t( ", elems);
  for (i = 0; i < elems; i++) { printf("%ld ", A[i]); }
  printf(")\n");
}

// --------- INT END ---------

// --------- UINT --------- 

void print_uint8x8_t(uint8x8_t v) {
  int elems = 8, i;
  uint8_t A[elems];
  vst1_u8(A, v);
  printf("uint8x%d_t( ", elems);
  for (i = 0; i < elems; i++) { printf("%d ", A[i]); }
  printf(")\n");
}

void print_uint8x16_t(uint8x16_t v) {
  int elems = 16, i;
  uint8_t A[elems];
  vst1q_u8(A, v);
  printf("uint8x%d_t( ", elems);
  for (i = 0; i < elems; i++) { printf("%d ", A[i]); }
  printf(")\n");
}

void print_uint16x4_t(uint16x4_t v) {
  int elems = 4, i;
  uint16_t A[elems];
  vst1_u16(A, v);
  printf("uint8x%d_t( ", elems);
  for (i = 0; i < elems; i++) { printf("%d ", A[i]); }
  printf(")\n");
}

void print_uint16x8_t(uint16x8_t v) {
  int elems = 8, i;
  uint16_t A[elems];
  vst1q_u16(A, v);
  printf("uint8x%d_t( ", elems);
  for (i = 0; i < elems; i++) { printf("%d ", A[i]); }
  printf(")\n");
}

void print_uint32x2_t(uint32x2_t v) {
  int elems = 2, i;
  uint32_t A[elems];
  vst1_u32(A, v);
  printf("uint8x%d_t( ", elems);
  for (i = 0; i < elems; i++) { printf("%d ", A[i]); }
  printf(")\n");
}

void print_uint32x4_t(uint32x4_t v) {
  int elems = 4, i;
  uint32_t A[elems];
  vst1q_u32(A, v);
  printf("uint8x%d_t( ", elems);
  for (i = 0; i < elems; i++) { printf("%d ", A[i]); }
  printf(")\n");
}

void print_uint64x1_t(uint64x1_t v) {
  int elems = 1, i;
  uint64_t A[elems];
  vst1_u64(A, v);
  printf("uint8x%d_t( ", elems);
  for (i = 0; i < elems; i++) { printf("%ld ", A[i]); }
  printf(")\n");
}

void print_uint64x2_t(uint64x2_t v) {
  int elems = 2, i;
  uint64_t A[elems];
  vst1q_u64(A, v);
  printf("uint8x%d_t( ", elems);
  for (i = 0; i < elems; i++) { printf("%ld ", A[i]); }
  printf(")\n");
}

// --------- UINT END --------- 

// --------- FLOAT --------- 

void print_float32x2_t(float32x2_t v) {
  int elems = 2, i;
  float A[elems];
  vst1_f32(A, v);
  printf("float32x%d_t( ", elems);
  for (i = 0; i < elems; i++) { printf("%.1f ", A[i]); }
  printf(")\n");
}

void print_float32x4_t(float32x4_t v) {
  int elems = 4, i;
  float A[elems];
  vst1q_f32(A, v);
  printf("float32x%d_t( ", elems);
  for (i = 0; i < elems; i++) { printf("%.1f ", A[i]); }
  printf(")\n");
}

void print_float32x4x2_t(float32x4x2_t v) {
  int elems = 8, i;
  float A[elems];
  vst2q_f32(A, v);
  printf("float32x4x2_t( ");
  for (i = 0; i < elems; i++) { printf("%.1f ", A[i]); }
  printf(")\n");
}

// --------- FLOAT END ---------


#endif // VEC_PRINT_H_
