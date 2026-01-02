#include "params.h"
#include "CPU.h"
#include "kernel.h"
#include "omp.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <cstdint>

void setContainmentJoinGPUBatched(int * R_data, int * R_offsets, int R_size, int * S_data, int * S_offsets, int S_size, int S_elementCount, int largestElement, vector<int2> * resultSet);

void setContainmentJoinGPUUVM(int * R_data, int * R_offsets, int R_size, int * S_data, int * S_offsets, int S_size, int S_elementCount, int largestElement, vector<int2> * resultSet);

void checkGPUMem();