#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "miner.h"

#include <stdio.h>
#include <stdint.h>
#include <memory.h>

#include "cuda_helper.h"
#include "vector_functions.h"
static __forceinline__ __device__
int4 operator+ (int4 a, int4 b) { return make_int4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }

static __forceinline__ __device__
int4 operator- (int4 a, int4 b) { return make_int4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }

#define BLOCKSZ 32

typedef signed char swift_int8_t;
typedef unsigned char swift_uint8_t;

typedef short swift_int16_t;
typedef unsigned short swift_uint16_t;

typedef int swift_int32_t;
typedef unsigned int swift_uint32_t;

typedef long long swift_int64_t;
typedef unsigned long long swift_uint64_t;


#define SWIFFTX_INPUT_BLOCK_SIZE 256
#define SWIFFTX_OUTPUT_BLOCK_SIZE 65
#define FIELD_SIZE 257
#define N 64
#define EIGHTH_N (N / 8)
#define M (SWIFFTX_INPUT_BLOCK_SIZE / 8)   // 32
#define M_2 25 // 3 * (N/8) + 1
#define W 8

#define ADD_SUB(A, B) {int temp = (B); B = ((A) - (B)); A = ((A) + (temp));}
#define Q_REDUCE(A) (((A) & 0xff) - ((A) >> 8))

#include "x22/cuda_swifftx512.h"

#define BYTE(x, n) __byte_perm(x, 0, 0x4440 + (n))


__device__ // __forceinline__
swift_int16_t TranslateToBase256(swift_int32_t input[EIGHTH_N], unsigned char output[EIGHTH_N]) {
  swift_int32_t pairs[EIGHTH_N / 2];

  #pragma unroll
  for (int i = 0; i < EIGHTH_N; i += 2) {
    pairs[i >> 1] = input[i] + input[i + 1] + (input[i + 1] << 8);
  }

  #pragma unroll
  for (int i = (EIGHTH_N / 2) - 1; i > 0; --i) {

    #pragma unroll
    for (int j = i - 1; j < (EIGHTH_N / 2) - 1; ++j) {
      swift_int32_t temp = pairs[j] + pairs[j + 1] + (pairs[j + 1] << 9);
      pairs[j] = temp & 0xffff;
      pairs[j + 1] += (temp >> 16);
    }
  }

  #pragma unroll
  for (int i = 0; i < EIGHTH_N; i += 2) {
    output[i] = BYTE(pairs[i >> 1], 0);
    output[i + 1] = BYTE(pairs[i >> 1], 1);
  }

  return (pairs[EIGHTH_N/2 - 1] >> 16);
}


void h_InitializeSWIFFTX() {
}

__device__
void e_FFT_staged(const unsigned char input[EIGHTH_N], swift_int32_t *output,
		  const swift_int16_t *fftTable,
		  const swift_int16_t *multipliers,
		  int i /* stage */) {

  swift_int32_t F0,F1,F2,F3,F4,F5,F6,F7;

  F0  = multipliers[0 + (i << 3)] * *(&fftTable[input[0] << 3] + i);
  F1  = multipliers[1 + (i << 3)] * *(&fftTable[input[1] << 3] + i);
  F2  = multipliers[2 + (i << 3)] * *(&fftTable[input[2] << 3] + i);
  F3  = multipliers[3 + (i << 3)] * *(&fftTable[input[3] << 3] + i);
  F4  = multipliers[4 + (i << 3)] * *(&fftTable[input[4] << 3] + i);
  F5  = multipliers[5 + (i << 3)] * *(&fftTable[input[5] << 3] + i);
  F6  = multipliers[6 + (i << 3)] * *(&fftTable[input[6] << 3] + i);
  F7  = multipliers[7 + (i << 3)] * *(&fftTable[input[7] << 3] + i);

  ADD_SUB(F0, F1);
  ADD_SUB(F2, F3);
  ADD_SUB(F4, F5);
  ADD_SUB(F6, F7);

  F3 <<= 4;
  F7 <<= 4;

  ADD_SUB(F0, F2);
  ADD_SUB(F1, F3);
  ADD_SUB(F4, F6);
  ADD_SUB(F5, F7);

  F5 <<= 2;
  F6 <<= 4;
  F7 <<= 6;

  ADD_SUB(F0, F4);
  ADD_SUB(F1, F5);
  ADD_SUB(F2, F6);
  ADD_SUB(F3, F7);

  output[0] = Q_REDUCE(F0);
  output[1] = Q_REDUCE(F1);
  output[2] = Q_REDUCE(F2);
  output[3] = Q_REDUCE(F3);
  output[4] = Q_REDUCE(F4);
  output[5] = Q_REDUCE(F5);
  output[6] = Q_REDUCE(F6);
  output[7] = Q_REDUCE(F7);
}


__device__
void e_FFT_staged_int4(const unsigned char input[EIGHTH_N], swift_int32_t *output,
		       const swift_int16_t *fftTable,
		       const swift_int16_t *multipliers,
		       int i /* stage */) {

  swift_int32_t F0,F1,F2,F3,F4,F5,F6,F7;

  F0  = multipliers[0 + (i << 3)] * *(&fftTable[input[0] << 3] + i);
  F1  = multipliers[1 + (i << 3)] * *(&fftTable[input[1] << 3] + i);
  F2  = multipliers[2 + (i << 3)] * *(&fftTable[input[2] << 3] + i);
  F3  = multipliers[3 + (i << 3)] * *(&fftTable[input[3] << 3] + i);
  F4  = multipliers[4 + (i << 3)] * *(&fftTable[input[4] << 3] + i);
  F5  = multipliers[5 + (i << 3)] * *(&fftTable[input[5] << 3] + i);
  F6  = multipliers[6 + (i << 3)] * *(&fftTable[input[6] << 3] + i);
  F7  = multipliers[7 + (i << 3)] * *(&fftTable[input[7] << 3] + i);

  int4 a0 = make_int4(F0, F2, F4, F6);
  int4 a1 = make_int4(F1, F3, F5, F7);

#define ADD_SUB4(A, B) { int4 temp = (B); B = ((A) - (B)); A = ((A) + (temp)); }

  ADD_SUB4(a0, a1);

  a1.y <<= 4;
  a1.w <<= 4;

  int4 b0 = make_int4(a0.x, a1.x, a0.z, a1.z);
  int4 b1 = make_int4(a0.y, a1.y, a0.w, a1.w);

  ADD_SUB4(b0, b1);

  b0.w <<= 2;
  b1.z <<= 4;
  b1.w <<= 6;

  int4 c0 = make_int4(b0.x, b0.y, b1.x, b1.y);
  int4 c1 = make_int4(b0.z, b0.w, b1.z, b1.w);

  ADD_SUB4(c0, c1);

  output[0] = Q_REDUCE(c0.x);
  output[1] = Q_REDUCE(c0.y);
  output[2] = Q_REDUCE(c0.z);
  output[3] = Q_REDUCE(c0.w);
  output[4] = Q_REDUCE(c1.x);
  output[5] = Q_REDUCE(c1.y);
  output[6] = Q_REDUCE(c1.z);
  output[7] = Q_REDUCE(c1.w);
}

//__shared__ swift_int32_t __FIELD_SIZE_22__;
#define __FIELD_SIZE_22__ (FIELD_SIZE << 22)

__device__
void setzero(void *a, int size) {
  swift_int64_t *ptr = (swift_int64_t *)a;
  #pragma unroll
  for (int i=0; i<(size/8); i++) ptr[i] = 0;
}

__device__
void e_ComputeSingleSWIFFTX(unsigned char input[SWIFFTX_INPUT_BLOCK_SIZE],
			    unsigned char output[SWIFFTX_OUTPUT_BLOCK_SIZE],
			    const unsigned char SBox[256],
			    const swift_int16_t As[3*M*N],
			    const swift_int16_t fftTable[256 * EIGHTH_N],
			    const swift_int16_t multipliers[64]) {
  swift_int32_t sum[3*N];
  setzero(sum, 3*N*sizeof(swift_int32_t));

  #pragma nounroll
  for (int i=0; i<M; ++i) {
    #pragma unroll
    for (int stride=0; stride<8; stride++) { // 0 8 16 24 32 40 48 56
      swift_int32_t fftOut[8];
      e_FFT_staged_int4(input + (i << 3), fftOut, fftTable, multipliers, stride);
      const swift_int16_t *As_i = As + (i*N);

      #pragma unroll
      for (int j=0; j<N/8; j++) {
	const int jj = stride + (j << 3);
	const swift_int16_t *As_j = As_i + jj;
	const swift_int32_t *f = fftOut + j;

	#pragma unroll
	for (int k=0; k<3; ++k) {
	  const swift_int16_t *a = As_j + (k << 11); //As + (k * M * N) + (i * N) + j;
	  sum[k*N + jj] += (*f) * (*a);
	}
      }
    }
  }

  unsigned char intermediate[N*3 + 8];
  setzero(intermediate, 24);

  #pragma unroll
  for (int k=0; k<3; ++k) {

    #pragma unroll
    for (int j=0; j<N; ++j) {
      sum[k*N + j] = (__FIELD_SIZE_22__ + sum[k*N + j]) % FIELD_SIZE;
    }

    int carry=0;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      int carryBit = TranslateToBase256(sum + (k*N) + (j << 3), intermediate + (k*N) + (j << 3));
      carry |= carryBit << j;
    }

    intermediate[3*N+k] = carry;
  }

  #pragma unroll
  for (int i = 0; i < (3 * N) + 3; ++i)
    intermediate[i] = SBox[intermediate[i]];

  #pragma unroll
  for (int i = (3 * N) + 3; i < (3 * N) + 8; ++i)
    intermediate[i] = 0x7d;

  setzero(sum, N*sizeof(swift_int32_t));

  #pragma nounroll
  for (int i=0; i<M_2; ++i) {
    swift_int32_t fftOut[8];
    #pragma unroll
    for (int stride=0; stride<8; stride++) {
      e_FFT_staged_int4(intermediate + (i << 3), fftOut, fftTable, multipliers, stride);
      #pragma unroll
      for (int j=0; j<N/8; ++j) {
	const int jj = stride + (j << 3);
	const swift_int16_t *a = As + (i * N) + jj;
	const swift_int32_t *f = fftOut + j;
	sum[jj] += (*f) * (*a);
      }
    }
  }
  #pragma unroll
  for (int j=0; j<N; ++j) {
    sum[j] = (__FIELD_SIZE_22__ + sum[j]) % FIELD_SIZE;
  }

  #pragma unroll
  for (int j = 0; j < 8; ++j) {
    TranslateToBase256(sum + (j << 3), output + (j << 3));
  }
}

#define __E64__ 64

__global__  __launch_bounds__(BLOCKSZ, 1)
void swifftx512_gpu_hash_64(int threads,
			    uint32_t *g_hash, uint32_t *g_hash1, uint32_t *g_hash2, uint32_t *g_hash3) {

  const int thread = (blockDim.x * blockIdx.x + threadIdx.x);
  const int tid = threadIdx.x;

  __shared__ unsigned char S_SBox[256];
  __shared__ swift_int16_t S_fftTable[256 * EIGHTH_N];

  uint32_t in[64];

  const int blockSize = min(256, BLOCKSZ); //blockDim.x;

  if (tid < 256) {
  #pragma unroll
  for (int i=0; i<(256/blockSize); i++) {
    S_SBox[tid + i*blockSize] = SBox[tid + i*blockSize];
  }

  #pragma unroll
  for (int i=0; i<(256 * EIGHTH_N)/blockSize; i++) {
    S_fftTable[tid + i*blockSize] = fftTable[tid + i*blockSize];
  }
  }

  if (thread < threads) {
    uint32_t* inout = &g_hash [thread<<4];
    uint32_t* in1   = &g_hash1[thread<<4];
    uint32_t* in2   = &g_hash2[thread<<4];
    uint32_t* in3   = &g_hash3[thread<<4];

    #pragma unroll
    for (int i = 0; i < 16; i++) {
      in[i     ] = inout[i];
      in[i + 16] = in1  [i];
      in[i + 32] = in2  [i];
      in[i + 48] = in3  [i];
    }

    e_ComputeSingleSWIFFTX((unsigned char*)in, (unsigned char*)in, S_SBox, As, S_fftTable, multipliers);

    #pragma unroll
    for (int i = 0; i < 16; i++)
      inout[i] = in[i];
   }
}

__host__
void swifftx512_cpu_hash_64(int thr_id, int threads,
			    uint32_t *d_hash, uint32_t *d_hash1, uint32_t *d_hash2, uint32_t *d_hash3) {

  const int threadsperblock = BLOCKSZ; //128;
  dim3 grid(threads/threadsperblock);
  dim3 block(threadsperblock);
  swifftx512_gpu_hash_64<<<grid, block>>>(threads, d_hash, d_hash1, d_hash2, d_hash3);
}

/*
 * SWIFFTX CUDA Implementation Optimization
 *
 * Date  : October 2018
 * Author: eburimu <urusan80@gmail.com>
 *
 */
