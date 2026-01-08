#include "kernel.h"

__global__ void kernelCountResultSetSize(int * R_data, int * R_offsets, int R_offsetRate, int R_size, int * S_data, int * S_offsets, int S_size, unsigned long long int * resultSetSize)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int i = tid * R_offsetRate;

	if (i>=R_size){
		return;
	}

    for (int j=0; j<S_size; j++)
    {
        if (isContainedIn(R_data, R_offsets[i], R_offsets[i+1], S_data, S_offsets[j], S_offsets[j+1]))
        {
            atomicAdd(resultSetSize, 1);
        }
    }
}

__global__ void kernelFillResultSet(int * R_data, int * R_offsets, int R_batchOffset, int R_batchSize, int * S_data, int * S_offsets, int S_size, int2 * resultSet, unsigned long long int * resultSetSize)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid>=R_batchSize){
		return;
	}

    int i = R_batchOffset + tid;

    for (int j=0; j<S_size; j++)
    {
        if (isContainedIn(R_data, R_offsets[i], R_offsets[i+1], S_data, S_offsets[j], S_offsets[j+1]))
        {
            unsigned long long int resultIndex = atomicAdd(resultSetSize, 1);
            resultSet[resultIndex] = make_int2(i, j);
        }
    }
}

__device__ bool isContainedIn(int * R_data, int R_start, int R_end, int * S_data, int S_start, int S_end)
{
    for (int i=R_start; i<R_end; i++)
    {
        int r_elem = R_data[i];

        bool found = false;

        for (int j=S_start; j<S_end && !found; j++)
        {
            int s_elem = S_data[j];

            found = (r_elem == s_elem);
        }
        
        if (!found)
        {
            return false;
        }
    }

    return true;
}
