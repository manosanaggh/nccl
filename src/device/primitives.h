/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_PRIMITIVES_H_
#define NCCL_PRIMITIVES_H_

#include <type_traits>
#include "reduce_kernel.h" // for reduction funcs
#include "common_kernel.h"
#include "common.h"

#define NCCL_SPINS_BEFORE_CHECK_ABORT 10000

#ifndef NCCL_DEVICE_MEASURE_RING_PRIMS
#define NCCL_DEVICE_MEASURE_RING_PRIMS 1
#endif

#if NCCL_DEVICE_MEASURE_RING_PRIMS
#define NCCL_RING_KERNEL_CHANNEL_MEASURE_START(_tid) \
  (((_tid) == 0 && ncclShmem.comm.measureRingPrims && ncclShmem.comm.measureRingPrimsStats != nullptr) ? globaltimer() : 0ULL)
#define NCCL_RING_OP_MEASURE_VALID(_work) \
  ((_work) != nullptr && (_work)->measureOpSlot < NCCL_RING_PRIM_OP_STATS_MAX)
#define NCCL_RING_OP_MEASURE_META(_work) do { \
  if (NCCL_RING_OP_MEASURE_VALID(_work)) { \
    int __ncclRingOpMetaBase = NCCL_RING_PRIM_OP_META_BASE + (int)((_work)->measureOpSlot) * NCCL_RING_PRIM_OP_META_FIELDS; \
    atomicExch((unsigned long long*)(ncclShmem.comm.measureRingPrimsStats + __ncclRingOpMetaBase + NCCL_RING_PRIM_OP_META_VALID), 1ULL); \
    atomicExch((unsigned long long*)(ncclShmem.comm.measureRingPrimsStats + __ncclRingOpMetaBase + NCCL_RING_PRIM_OP_META_FUNC), (unsigned long long)((_work)->measureFunc)); \
    atomicExch((unsigned long long*)(ncclShmem.comm.measureRingPrimsStats + __ncclRingOpMetaBase + NCCL_RING_PRIM_OP_META_BYTES), (unsigned long long)((_work)->measureCollBytes)); \
    atomicExch((unsigned long long*)(ncclShmem.comm.measureRingPrimsStats + __ncclRingOpMetaBase + NCCL_RING_PRIM_OP_META_CHANNEL_LO), (unsigned long long)((_work)->channelLo)); \
    atomicExch((unsigned long long*)(ncclShmem.comm.measureRingPrimsStats + __ncclRingOpMetaBase + NCCL_RING_PRIM_OP_META_CHANNEL_HI), (unsigned long long)((_work)->channelHi)); \
  } \
} while (0)
#define NCCL_RING_KERNEL_CHANNEL_MEASURE_END(_tid, _start, _work) do { \
  if ((_tid) == 0 && ncclShmem.comm.measureRingPrims && ncclShmem.comm.measureRingPrimsStats != nullptr && (_start) != 0ULL) { \
    unsigned long long __ncclRingKernelChannelEnd = globaltimer(); \
    atomicAdd((unsigned long long*)(ncclShmem.comm.measureRingPrimsStats + NCCL_RING_PRIM_STATS_KERNEL_CHANNEL_COUNT), 1ULL); \
    atomicAdd((unsigned long long*)(ncclShmem.comm.measureRingPrimsStats + NCCL_RING_PRIM_STATS_KERNEL_CHANNEL_NS), __ncclRingKernelChannelEnd - (unsigned long long)(_start)); \
    NCCL_RING_OP_MEASURE_META(_work); \
  } \
} while (0)
#define NCCL_RING_SYNC_MEASURE_START(_enabled) \
  (((_enabled) && ncclShmem.comm.measureRingPrims && ncclShmem.comm.measureRingPrimsStats != nullptr) ? globaltimer() : 0ULL)
#define NCCL_RING_SYNC_MEASURE_END(_enabled, _syncId, _start, _work) do { \
  if ((_enabled) && ncclShmem.comm.measureRingPrims && ncclShmem.comm.measureRingPrimsStats != nullptr && (_start) != 0ULL) { \
    unsigned long long __ncclRingSyncEnd = globaltimer(); \
    unsigned long long __ncclRingSyncNs = __ncclRingSyncEnd - (unsigned long long)(_start); \
    int __ncclRingSyncBase = NCCL_RING_PRIM_STATS_SYNC_BASE + ((int)(_syncId)) * NCCL_RING_PRIM_STATS_SYNC_FIELDS; \
    atomicAdd((unsigned long long*)(ncclShmem.comm.measureRingPrimsStats + __ncclRingSyncBase + NCCL_RING_PRIM_STATS_SYNC_COUNT), 1ULL); \
    atomicAdd((unsigned long long*)(ncclShmem.comm.measureRingPrimsStats + __ncclRingSyncBase + NCCL_RING_PRIM_STATS_SYNC_NS), __ncclRingSyncNs); \
    int __ncclRingSyncChannel = ncclShmem.channelId; \
    if (__ncclRingSyncChannel >= 0 && __ncclRingSyncChannel < MAXCHANNELS) { \
      int __ncclRingSyncChannelBase = NCCL_RING_PRIM_STATS_SYNC_CHANNEL_BASE + __ncclRingSyncChannel * NCCL_RING_PRIM_STATS_SYNC_CHANNEL_STRIDE + ((int)(_syncId)) * NCCL_RING_PRIM_STATS_SYNC_FIELDS; \
      atomicAdd((unsigned long long*)(ncclShmem.comm.measureRingPrimsStats + __ncclRingSyncChannelBase + NCCL_RING_PRIM_STATS_SYNC_COUNT), 1ULL); \
      atomicAdd((unsigned long long*)(ncclShmem.comm.measureRingPrimsStats + __ncclRingSyncChannelBase + NCCL_RING_PRIM_STATS_SYNC_NS), __ncclRingSyncNs); \
      if (NCCL_RING_OP_MEASURE_VALID(_work)) { \
        int __ncclRingOpSyncBase = NCCL_RING_PRIM_OP_SYNC_CHANNEL_BASE + (int)((_work)->measureOpSlot) * MAXCHANNELS * NCCL_RING_PRIM_OP_SYNC_CHANNEL_STRIDE + __ncclRingSyncChannel * NCCL_RING_PRIM_OP_SYNC_CHANNEL_STRIDE + ((int)(_syncId)) * NCCL_RING_PRIM_STATS_SYNC_FIELDS; \
        atomicAdd((unsigned long long*)(ncclShmem.comm.measureRingPrimsStats + __ncclRingOpSyncBase + NCCL_RING_PRIM_STATS_SYNC_COUNT), 1ULL); \
        atomicAdd((unsigned long long*)(ncclShmem.comm.measureRingPrimsStats + __ncclRingOpSyncBase + NCCL_RING_PRIM_STATS_SYNC_NS), __ncclRingSyncNs); \
      } \
    } \
  } \
} while (0)
#define NCCL_RING_PRIM_MEASURE_START(_tid) \
  (((_tid) == 0 && ncclShmem.comm.measureRingPrims && ncclShmem.comm.measureRingPrimsStats != nullptr) ? globaltimer() : 0ULL)
#define NCCL_RING_PRIM_MEASURE_END_IF(_net, _tid, _primId, _elemOffset, _chunkCount, _nelem, _typeSize, _start) do { \
  if ((_tid) == 0 && ncclShmem.comm.measureRingPrims && ncclShmem.comm.measureRingPrimsStats != nullptr && (_start) != 0ULL) { \
    unsigned long long __ncclRingPrimBytes = (unsigned long long)(_nelem) * (unsigned long long)(_typeSize); \
    if (__ncclRingPrimBytes >= ncclShmem.comm.measureRingPrimsMinBytes) { \
      unsigned long long __ncclRingPrimEnd = globaltimer(); \
      int __ncclRingPrimBase = NCCL_RING_PRIM_STATS_PRIM_BASE + ((int)(_primId)) * NCCL_RING_PRIM_STATS_PRIM_FIELDS; \
      unsigned long long __ncclRingPrimNs = __ncclRingPrimEnd - (unsigned long long)(_start); \
      atomicAdd((unsigned long long*)(ncclShmem.comm.measureRingPrimsStats + __ncclRingPrimBase + NCCL_RING_PRIM_STATS_PRIM_COUNT), 1ULL); \
      atomicAdd((unsigned long long*)(ncclShmem.comm.measureRingPrimsStats + __ncclRingPrimBase + NCCL_RING_PRIM_STATS_PRIM_BYTES), __ncclRingPrimBytes); \
      atomicAdd((unsigned long long*)(ncclShmem.comm.measureRingPrimsStats + __ncclRingPrimBase + NCCL_RING_PRIM_STATS_PRIM_NS), __ncclRingPrimNs); \
      if (_net) { \
        atomicAdd((unsigned long long*)(ncclShmem.comm.measureRingPrimsStats + __ncclRingPrimBase + NCCL_RING_PRIM_STATS_PRIM_NET_COUNT), 1ULL); \
        atomicAdd((unsigned long long*)(ncclShmem.comm.measureRingPrimsStats + __ncclRingPrimBase + NCCL_RING_PRIM_STATS_PRIM_NET_BYTES), __ncclRingPrimBytes); \
        atomicAdd((unsigned long long*)(ncclShmem.comm.measureRingPrimsStats + __ncclRingPrimBase + NCCL_RING_PRIM_STATS_PRIM_NET_NS), __ncclRingPrimNs); \
      } \
    } \
  } \
} while (0)
#define NCCL_RING_PRIM_MEASURE_END(_tid, _primId, _elemOffset, _chunkCount, _nelem, _typeSize, _start) \
  NCCL_RING_PRIM_MEASURE_END_IF(0, _tid, _primId, _elemOffset, _chunkCount, _nelem, _typeSize, _start)
#define NCCL_RING_IB_STAGING_COPY_MEASURE_START(_enabled, _tid) \
  (((_enabled) && (_tid) == 0 && ncclShmem.comm.measureRingPrims && ncclShmem.comm.measureRingPrimsStats != nullptr) ? globaltimer() : 0ULL)
#define NCCL_RING_IB_STAGING_COPY_MEASURE_END(_enabled, _tid, _prim, _nelem, _typeSize, _start, _work) do { \
  if ((_enabled) && (_tid) == 0 && ncclShmem.comm.measureRingPrims && ncclShmem.comm.measureRingPrimsStats != nullptr && (_start) != 0ULL) { \
    unsigned long long __ncclRingStagingBytes = (unsigned long long)(_nelem) * (unsigned long long)(_typeSize); \
    if (__ncclRingStagingBytes >= ncclShmem.comm.measureRingPrimsMinBytes) { \
      unsigned long long __ncclRingStagingEnd = globaltimer(); \
      unsigned long long __ncclRingStagingNs = __ncclRingStagingEnd - (unsigned long long)(_start); \
      atomicAdd((unsigned long long*)(ncclShmem.comm.measureRingPrimsStats + NCCL_RING_PRIM_STATS_STAGING_COUNT), 1ULL); \
      atomicAdd((unsigned long long*)(ncclShmem.comm.measureRingPrimsStats + NCCL_RING_PRIM_STATS_STAGING_BYTES), __ncclRingStagingBytes); \
      atomicAdd((unsigned long long*)(ncclShmem.comm.measureRingPrimsStats + NCCL_RING_PRIM_STATS_STAGING_NS), __ncclRingStagingNs); \
      int __ncclRingStagingChannel = ncclShmem.channelId; \
      if (__ncclRingStagingChannel >= 0 && __ncclRingStagingChannel < MAXCHANNELS) { \
        int __ncclRingStagingChannelBase = NCCL_RING_PRIM_STATS_STAGING_CHANNEL_BASE + __ncclRingStagingChannel * NCCL_RING_PRIM_STATS_STAGING_CHANNEL_FIELDS; \
        atomicAdd((unsigned long long*)(ncclShmem.comm.measureRingPrimsStats + __ncclRingStagingChannelBase + NCCL_RING_PRIM_STATS_STAGING_COUNT), 1ULL); \
        atomicAdd((unsigned long long*)(ncclShmem.comm.measureRingPrimsStats + __ncclRingStagingChannelBase + NCCL_RING_PRIM_STATS_STAGING_BYTES), __ncclRingStagingBytes); \
        atomicAdd((unsigned long long*)(ncclShmem.comm.measureRingPrimsStats + __ncclRingStagingChannelBase + NCCL_RING_PRIM_STATS_STAGING_NS), __ncclRingStagingNs); \
        if (NCCL_RING_OP_MEASURE_VALID(_work)) { \
          int __ncclRingOpStagingBase = NCCL_RING_PRIM_OP_STAGING_CHANNEL_BASE + (int)((_work)->measureOpSlot) * MAXCHANNELS * NCCL_RING_PRIM_OP_STAGING_CHANNEL_STRIDE + __ncclRingStagingChannel * NCCL_RING_PRIM_OP_STAGING_CHANNEL_STRIDE; \
          atomicAdd((unsigned long long*)(ncclShmem.comm.measureRingPrimsStats + __ncclRingOpStagingBase + NCCL_RING_PRIM_STATS_STAGING_COUNT), 1ULL); \
          atomicAdd((unsigned long long*)(ncclShmem.comm.measureRingPrimsStats + __ncclRingOpStagingBase + NCCL_RING_PRIM_STATS_STAGING_BYTES), __ncclRingStagingBytes); \
          atomicAdd((unsigned long long*)(ncclShmem.comm.measureRingPrimsStats + __ncclRingOpStagingBase + NCCL_RING_PRIM_STATS_STAGING_NS), __ncclRingStagingNs); \
        } \
      } \
    } \
  } \
} while (0)
#define NCCL_RING_SEND_ROUNDTRIP_MEASURE_END(_enabled, _start) do { \
  if ((_enabled) && ncclShmem.comm.measureRingPrims && ncclShmem.comm.measureRingPrimsStats != nullptr && (_start) != 0ULL) { \
    unsigned long long __ncclRingRtEnd = globaltimer(); \
    unsigned long long __ncclRingRtNs = __ncclRingRtEnd - (unsigned long long)(_start); \
    int __ncclRingRtChannel = ncclShmem.channelId; \
    if (__ncclRingRtChannel >= 0 && __ncclRingRtChannel < MAXCHANNELS) { \
      int __ncclRingRtBase = NCCL_RING_PRIM_STATS_ROUNDTRIP_CHANNEL_BASE + __ncclRingRtChannel * NCCL_RING_PRIM_STATS_ROUNDTRIP_CHANNEL_FIELDS; \
      atomicAdd((unsigned long long*)(ncclShmem.comm.measureRingPrimsStats + __ncclRingRtBase + NCCL_RING_PRIM_STATS_ROUNDTRIP_COUNT), 1ULL); \
      atomicAdd((unsigned long long*)(ncclShmem.comm.measureRingPrimsStats + __ncclRingRtBase + NCCL_RING_PRIM_STATS_ROUNDTRIP_NS), __ncclRingRtNs); \
      atomicMax((unsigned long long*)(ncclShmem.comm.measureRingPrimsStats + __ncclRingRtBase + NCCL_RING_PRIM_STATS_ROUNDTRIP_MAX_NS), __ncclRingRtNs); \
    } \
  } \
} while (0)
#else
#define NCCL_RING_KERNEL_CHANNEL_MEASURE_START(_tid) 0ULL
#define NCCL_RING_OP_MEASURE_VALID(_work) 0
#define NCCL_RING_OP_MEASURE_META(_work) do {} while (0)
#define NCCL_RING_KERNEL_CHANNEL_MEASURE_END(_tid, _start, _work) do {} while (0)
#define NCCL_RING_SYNC_MEASURE_START(_enabled) 0ULL
#define NCCL_RING_SYNC_MEASURE_END(_enabled, _syncId, _start, _work) do {} while (0)
#define NCCL_RING_PRIM_MEASURE_START(_tid) 0ULL
#define NCCL_RING_PRIM_MEASURE_END_IF(_net, _tid, _primId, _elemOffset, _chunkCount, _nelem, _typeSize, _start) do {} while (0)
#define NCCL_RING_PRIM_MEASURE_END(_tid, _primId, _elemOffset, _chunkCount, _nelem, _typeSize, _start) do {} while (0)
#define NCCL_RING_IB_STAGING_COPY_MEASURE_START(_enabled, _tid) 0ULL
#define NCCL_RING_IB_STAGING_COPY_MEASURE_END(_enabled, _tid, _prim, _nelem, _typeSize, _start, _work) do {} while (0)
#define NCCL_RING_SEND_ROUNDTRIP_MEASURE_END(_enabled, _start) do {} while (0)
#endif
/* Protocol classes: ProtoSimple, ProtoLL, ProtoLL128
 * We use these as template args to the Primtiives class instead of integral
 * enums (e.g. NCCL_PROTO_LL) because for SIMPLE we need to carry a few extra
 * numbers. Also these types hold methods which let us compute numbers important
 * to how that protocol operates with a consistent interface so that our
 * algorithm code can operate protocol parametrically.
 */
template<int SlicePerChunk_1, int StepPerSlice_1, int Unroll_1 = COLL_UNROLL, int MultimemSrcs_1 = 0, int MultimemDsts_1 = 0>
struct ProtoSimple {
  static constexpr int Id = NCCL_PROTO_SIMPLE;
  static constexpr int SlicePerChunk = SlicePerChunk_1;
  static constexpr int StepPerSlice = StepPerSlice_1;
  static constexpr int Unroll = Unroll_1;
  static constexpr int MultimemSrcs = MultimemSrcs_1;
  static constexpr int MultimemDsts = MultimemDsts_1;

  // Data bytes (no flags etc) in one step of the fifo queue.
  __device__ static int calcBytePerStep() {
    return ncclShmem.comm.buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS;
  }
  // Granularity of data bytes transferred per thread.
  __device__ static int calcBytePerGrain() {
    return sizeof(uint64_t); // Bogus value? Nobody queries this metric for simple.
  }
  // Group width is how many consecutive group values a subchannel occupies.
  static constexpr int MaxGroupWidth = 2;
};

struct ProtoLL {
  static constexpr int Id = NCCL_PROTO_LL;

  // Data bytes (no flags etc) in one step of the fifo queue.
  __device__ static int calcBytePerStep() {
    return ncclShmem.comm.buffSizes[NCCL_PROTO_LL]/NCCL_STEPS/2; // Half is data
  }
  // Granularity of data bytes transferred per thread.
  __device__ static int calcBytePerGrain() {
    return sizeof(uint64_t); // One 16-byte line has 8-bytes of data
  }
  // Group width is how many consecutive group values a subchannel occupies.
  static constexpr int MaxGroupWidth = 1;
};

struct ProtoLL128 {
  static constexpr int Id = NCCL_PROTO_LL128;

  // Data bytes (no flags etc) in one step of the fifo queue.
  __device__ static int calcBytePerStep() {
    return (ncclShmem.comm.buffSizes[NCCL_PROTO_LL128]/NCCL_STEPS)*NCCL_LL128_DATAELEMS/NCCL_LL128_LINEELEMS;
  }
  // Granularity of data bytes transferred per thread.
  __device__ static int calcBytePerGrain() {
    return NCCL_LL128_SHMEM_ELEMS_PER_THREAD*NCCL_LL128_DATAELEMS*sizeof(uint64_t)/NCCL_LL128_LINEELEMS;
  }
  // Group width is how many consecutive group values a subchannel occupies.
  static constexpr int MaxGroupWidth = 1;
};

/* Fan (as in fan-in & fan-out) classes hold recv and send counts. The template
 * arguments are static bounds on the maximum values. Asymmetric counts are
 * independent. Symmetric is a static guarantee that nrecv==nsend, so it only
 * stores one value at runtime. This optimization save 32-bit register, but more
 * importantly uses fewer predicate registers when unrolling loops.
 */
template<int MaxRecv_, int MaxSend_>
struct FanAsymmetric {
  static constexpr int MaxRecv = MaxRecv_, MaxSend = MaxSend_;
  int nr, ns;
  FanAsymmetric() = default;
  __device__ FanAsymmetric(int nrecv, int nsend): nr(nrecv), ns(nsend) {
    // assert(nrecv <= MaxRecv && nsend <= MaxSend);
  }
  __device__ int nrecv() const { return MaxRecv ? nr : 0; }
  __device__ int nsend() const { return MaxSend ? ns : 0; }
};

template<int MaxArity>
struct FanSymmetric {
  static constexpr int MaxRecv = MaxArity, MaxSend = MaxArity;
  int n;
  FanSymmetric() = default;
  __device__ FanSymmetric(int nrecv, int nsend): n(nrecv) {
    // assert(nrecv == nsend && nrecv <= MaxArity);
  }
  __device__ int nrecv() const { return n; }
  __device__ int nsend() const { return n; }
};

// The primitives class. Specialized per protocol in the other headers.
template<typename T, typename RedOp, typename Fan, int Direct, typename Proto, int P2p, bool isNetOffload = false>
class Primitives;

// Used by LL & LL128 to implement direct members in the naive way.
template<typename RealPrimitives>
struct PrimitivesWithoutDirect {
  __device__ void directSend(intptr_t inpIx, intptr_t outIx, int eltN) {
    static_cast<RealPrimitives*>(this)->send(inpIx, eltN);
  }
  __device__ void directSendFromOutput(intptr_t outIx, int eltN) {
    static_cast<RealPrimitives*>(this)->sendFromOutput(outIx, eltN);
  }
  __device__ void directRecv(intptr_t outIx, int eltN) {
    static_cast<RealPrimitives*>(this)->recv(outIx, eltN, /*postOp=*/false);
  }
  __device__ void directCopySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    static_cast<RealPrimitives*>(this)->copySend(inpIx, outIx, eltN, postOp);
  }
  __device__ void directRecvCopyDirectSend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    static_cast<RealPrimitives*>(this)->recvCopySend(outIx, eltN, /*postOp=*/false);
  }
  __device__ void directRecvDirectSend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    return;
  }
  __device__ void recvReduceCopyDirectSend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    // Direct is only for the send part
    static_cast<RealPrimitives*>(this)->recvReduceCopySend(inpIx, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvReduceDirectSend(intptr_t inpIx, intptr_t outIx, ssize_t eltN, bool postOp=false) {
    static_cast<RealPrimitives*>(this)->recvReduceSend(inpIx, eltN);
  }
  __device__ __forceinline__ void directRecvReduceCopyDirectSend(intptr_t inpIx, intptr_t outIx, ssize_t eltN, bool postOp=false) {
    static_cast<RealPrimitives*>(this)->recvReduceCopySend(inpIx, outIx, eltN, postOp);
  }
};

__device__ inline int checkAbort(int &abortCache, const int abortValue, int &spins) {
  if (abortCache & abortValue) return 1;
  if (++spins < NCCL_SPINS_BEFORE_CHECK_ABORT) return 0;
  spins = 0;
  int abort = *ncclShmem.comm.abortFlag;
  if (abort) {
    ncclShmem.aborted = abort;
    abortCache |= abortValue;
  }
  return abort;
}

#include "prims_simple.h"
#include "prims_ll.h"
#include "prims_ll128.h"
#endif
