#include "params.h"
#include <stdio.h>
#include <cstdint>

__global__ void kernelCountResultSetSize(int * R_data, int * R_offsets, int R_offsetRate, int R_size, int * S_data, int * S_offsets, int S_size, unsigned long long int * resultSetSize);

__global__ void kernelFillResultSet(int * R_data, int * R_offsets, int R_batchOffset, int R_batchSize, int * S_data, int * S_offsets, int S_size, int2 * resultSet, unsigned long long int * resultSetSize);

__device__ bool isContainedIn(int * R_data, int R_start, int R_end, int * S_data, int S_start, int S_end);