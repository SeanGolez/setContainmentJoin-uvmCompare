#include "params.h"
#include "CPU.h"
#include "kernel.h"
#include "omp.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <cstdint>

//thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/system/cuda/execution_policy.h> //for streams for thrust (added with Thrust v1.8)

#include <parallel/algorithm>


void setContainmentJoinGPUBatched(int * R_data, int * R_offsets, int R_size, int * S_data, int * S_offsets, int S_size, int S_elementCount, int largestElement, int2 ** resultSet, unsigned long long int * resultSetSize);

void setContainmentJoinGPUUVM(int * R_data, int * R_offsets, int R_size, int * S_data, int * S_offsets, int S_size, int S_elementCount, int largestElement, int2 ** resultSet, unsigned long long int * resultSetSize);

void probeAndStore(int2 * array, unsigned long long int * cnt, cudaEvent_t * kernelStop);

void checkGPUMem();