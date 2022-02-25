#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsp.h"
#include "me.h"

/* Motion estimation for 8x8 block */
static void me_block_8x8(struct c63_common *cm, int mb_x, int mb_y,
    uint8_t *orig, uint8_t *ref, int color_component)
{
  struct macroblock *mb =
    &cm->curframe->mbs[color_component][mb_y*cm->padw[color_component]/8+mb_x];

  int range = cm->me_search_range;

  /* Quarter resolution for chroma channels. */
  if (color_component > 0) { range /= 2; }

  int left = mb_x * 8 - range;
  int top = mb_y * 8 - range;
  int right = mb_x * 8 + range;
  int bottom = mb_y * 8 + range;

  int w = cm->padw[color_component];
  int h = cm->padh[color_component];

  /* Make sure we are within bounds of reference frame. TODO: Support partial
     frame bounds. */
  if (left < 0) { left = 0; }
  if (top < 0) { top = 0; }
  if (right > (w - 8)) { right = w - 8; }
  if (bottom > (h - 8)) { bottom = h - 8; }

  int x, y, row;

  int mx = mb_x * 8;
  int my = mb_y * 8;

  int best_sad = INT_MAX;

  /* Since the original image's block is constant for all the calls to sad_block_8x8,
  we load the 8x8 block into 8 x 8x8 vectors once here. */
  uint8x8_t orig_vectors[8];
  for (row = 0; row < 8; row++) {
    // Load in the next vector from the original block
    orig_vectors[row] = vld1_u8(orig + my*w+mx + w*row);
  }

  uint8x8_t refv, absv1, absv2;
  uint16x8_t sumv1, sumv2, sumv3, sumv4;

  // sad_block_8x8 have been inlined since it was called so many times
  for (y = top; y < bottom; y++) {
    for (x = left; x < right; x++) {
      int sad = 0;
      
      // Load the next row of the reference block
      refv = vld1_u8(ref + y*w+x);
      
      /* Compute the absolute differences between the row of the 
      original and the reference block */
      absv1 = vabd_u8(orig_vectors[0], refv);

      // Load the next row
      refv = vld1_u8(ref + (y+1)*w+x);

      // Compute the absolute differences again
      absv2 = vabd_u8(orig_vectors[1], refv);

      /* Use widening addition to sum the differences for the last 2 rows. We
      sum all the differences into one 16x8 row before doing across-vector sum, since
      across vector sum is slow */
      sumv1 = vaddl_u8(absv1, absv2);

      // Repeat the above for the remaining 6 rows
      refv = vld1_u8(ref + (y+2)*w+x);
      absv1 = vabd_u8(orig_vectors[2], refv);

      refv = vld1_u8(ref + (y+3)*w+x);
      absv2 = vabd_u8(orig_vectors[3], refv);

      sumv2 = vaddl_u8(absv1, absv2);

      refv = vld1_u8(ref + (y+4)*w+x);
      absv1 = vabd_u8(orig_vectors[4], refv);

      refv = vld1_u8(ref + (y+5)*w+x);
      absv2 = vabd_u8(orig_vectors[5], refv);

      sumv3 = vaddl_u8(absv1, absv2);

      refv = vld1_u8(ref + (y+6)*w+x);
      absv1 = vabd_u8(orig_vectors[6], refv);

      refv = vld1_u8(ref + (y+7)*w+x);
      absv2 = vabd_u8(orig_vectors[7], refv);

      sumv4 = vaddl_u8(absv1, absv2);

      // Sum up all the absolute differences into sumv1
      sumv1 = vaddq_u16(sumv1, sumv2);
      sumv3 = vaddq_u16(sumv3, sumv4);
      sumv1 = vaddq_u16(sumv1, sumv3);

      // Perform across-vector sum on sumv1 to get the final result
      sad = vaddvq_u16(sumv1);

      if (sad < best_sad) {
        mb->mv_x = x - mx;
        mb->mv_y = y - my;
        best_sad = sad;
      }
    }
  }

  /* Here, there should be a threshold on SAD that checks if the motion vector
     is cheaper than intraprediction. We always assume MV to be beneficial */

  /* printf("Using motion vector (%d, %d) with SAD %d\n", mb->mv_x, mb->mv_y,
     best_sad); */

  mb->use_mv = 1;
}

void c63_motion_estimate(struct c63_common *cm)
{
  /* Compare this frame with previous reconstructed frame */
  int mb_x, mb_y;

  // Loops have been merged
  for (mb_y = 0; mb_y < cm->mb_rows; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols; ++mb_x)
    {
      // Luma
      me_block_8x8(cm, mb_x, mb_y, cm->curframe->orig->Y,
          cm->refframe->recons->Y, Y_COMPONENT);

      // Chroma
      if (mb_y < cm->mb_rows / 2 && mb_x < cm->mb_cols / 2) {
        me_block_8x8(cm, mb_x, mb_y, cm->curframe->orig->U,
            cm->refframe->recons->U, U_COMPONENT);
        me_block_8x8(cm, mb_x, mb_y, cm->curframe->orig->V,
            cm->refframe->recons->V, V_COMPONENT);
      }
    }
  }
}

/* Motion compensation for 8x8 block */
static void mc_block_8x8(struct c63_common *cm, int mb_x, int mb_y,
    uint8_t *predicted, uint8_t *ref, int color_component)
{
  struct macroblock *mb =
    &cm->curframe->mbs[color_component][mb_y*cm->padw[color_component]/8+mb_x];

  if (!mb->use_mv) { return; }

  int left = mb_x * 8;
  int top = mb_y * 8;
  int bottom = top + 8;

  int w = cm->padw[color_component];

  /* Copy block from ref mandated by MV */
  int y;
  uint8x8_t v;

  for (y = top; y < bottom; ++y)
  {
    // Load 8 uint8_t elements from the reference
    v = vld1_u8(ref + (y + mb->mv_y) * w + (left + mb->mv_x));

    // Store the elements in predicted
    vst1_u8(predicted + y*w + left, v);
  }
}

void c63_motion_compensate(struct c63_common *cm)
{
  int mb_x, mb_y;

  // Loops have been merged
  for (mb_y = 0; mb_y < cm->mb_rows; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols; ++mb_x)
    {
      // Luma
      mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->Y,
          cm->refframe->recons->Y, Y_COMPONENT);

      // Chroma
      if (mb_y < cm->mb_rows / 2 && mb_x < cm->mb_cols / 2) {
        mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->U,
          cm->refframe->recons->U, U_COMPONENT);
        mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->V,
          cm->refframe->recons->V, V_COMPONENT);
      }
    }
  }
}
