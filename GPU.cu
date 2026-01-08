#include "GPU.h"

using namespace std;

#define checkError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void setContainmentJoinGPUBatched(int * R_data, int * R_offsets, int R_size, int * S_data, int * S_offsets, int S_size, int S_elementCount, int largestElement, int2 ** resultSet, unsigned long long int * resultSetSize)
{
    ////////////////////////////////////////////////////////////////////////////
    // Allocate memory
    int R_totalElements = R_offsets[R_size];
    int * dev_R_data;
    int * dev_R_offsets;
    checkError(cudaMalloc((void**)&dev_R_data, R_totalElements * sizeof(int)));
    checkError(cudaMalloc((void**)&dev_R_offsets, (R_size + 1) * sizeof(int)));
    checkError(cudaMemcpy(dev_R_data, R_data, R_totalElements * sizeof(int), cudaMemcpyHostToDevice));
    checkError(cudaMemcpy(dev_R_offsets, R_offsets, (R_size + 1) * sizeof(int), cudaMemcpyHostToDevice));

    int S_totalElements = S_offsets[S_size];
    int * dev_S_data;
    int * dev_S_offsets;
    checkError(cudaMalloc((void**)&dev_S_data, S_totalElements * sizeof(int)));
    checkError(cudaMalloc((void**)&dev_S_offsets, (S_size + 1) * sizeof(int)));
    checkError(cudaMemcpy(dev_S_data, S_data, S_totalElements * sizeof(int), cudaMemcpyHostToDevice));
    checkError(cudaMemcpy(dev_S_offsets, S_offsets, (S_size + 1) * sizeof(int), cudaMemcpyHostToDevice));

    // checkGPUMem();
    ////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////
    // Estimate result set size
    unsigned long long int sampleResultSetSize;

    unsigned long long int * dev_resultSetSize;
    checkError(cudaMalloc((void**)&dev_resultSetSize, sizeof(unsigned long long int)));
    checkError(cudaMemset(dev_resultSetSize, 0, sizeof(unsigned long long int)));

    int S_offsetRate=1.0 / SAMPLERATE;

    const int TOTALBLOCKSBATCHEST = (R_size + BLOCKSIZE - 1) / BLOCKSIZE;
    printf("\ntotal blocks: %d\n",TOTALBLOCKSBATCHEST);

    kernelCountResultSetSize<<<TOTALBLOCKSBATCHEST, BLOCKSIZE>>>(dev_R_data, dev_R_offsets, R_size, dev_S_data, dev_S_offsets, S_offsetRate, S_size, dev_resultSetSize);
    cout<<"** ERROR FROM KERNEL LAUNCH OF BATCH ESTIMATOR: "<<cudaGetLastError()<<endl;
    checkError(cudaDeviceSynchronize());

    checkError(cudaMemcpy(&sampleResultSetSize, dev_resultSetSize, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));

    cudaFree(dev_resultSetSize);

    unsigned long long int estimatedResultSetSize = sampleResultSetSize * S_offsetRate * 1.15;
    printf("estimatedResultSetSize: %llu\n", estimatedResultSetSize);
    ////////////////////////////////////////////////////////////////////////////
    
    ////////////////////////////////////////////////////////////////////////////
    // Estimate batches needed
    int numBatches = (estimatedResultSetSize) / GPUBUFFERSIZE;
    if (numBatches < 3)
    {
        numBatches = 3;
    }

    // batch on S
    unsigned long long int S_batchSize = S_size / numBatches;

    printf("numBatches: %d\n", numBatches);

    int batchesThatHaveOneMore = S_size - (S_batchSize * numBatches); //batch number 0-
    ////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////
    // Generate result set

    cudaStream_t stream[GPUSTREAMS];
	for (int i=0; i<GPUSTREAMS; i++){
	    cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking);
	}	

    unsigned long long int * dev_batchResultSetSize[GPUSTREAMS];
    int2 * dev_batchResultSet[GPUSTREAMS];
    int2 * batchResultSet[GPUSTREAMS]; // pinned memory
	for (int i=0; i<GPUSTREAMS; i++)
	{
        checkError(cudaMalloc((void **)&dev_batchResultSetSize[i], sizeof(unsigned long long int)));
		checkError(cudaMalloc((void **)&dev_batchResultSet[i], sizeof(int2)*GPUBUFFERSIZE));
        
        checkError(cudaMallocHost((void **) &batchResultSet[i], sizeof(int2)*GPUBUFFERSIZE));
	}

    *resultSet = new int2[estimatedResultSetSize];
    *resultSetSize = 0;

    #pragma omp parallel for schedule(static,1) num_threads(GPUSTREAMS)
    for (int i=0; i<numBatches; i++)
    {
        unsigned int tid = omp_get_thread_num();

        int thread_batchOffset = (i * S_batchSize);
        int thread_batchSize = S_batchSize;
        if (i<batchesThatHaveOneMore)
        {
            thread_batchOffset += i;
            thread_batchSize += 1;
        }
        else
        {
            thread_batchOffset += batchesThatHaveOneMore;
        }

        checkError(cudaMemsetAsync(dev_batchResultSetSize[tid], 0, sizeof(unsigned long long int), stream[tid]));

        const int TOTALBLOCKS = (R_size + BLOCKSIZE - 1) / BLOCKSIZE;
        printf("\ntotal blocks: %d\n",TOTALBLOCKS);

        kernelFillResultSet<<<TOTALBLOCKS, BLOCKSIZE, 0, stream[tid]>>>(dev_R_data, dev_R_offsets, R_size, dev_S_data, dev_S_offsets, thread_batchOffset, thread_batchSize, dev_batchResultSet[tid], dev_batchResultSetSize[tid]);
        cout<<"** ERROR FROM KERNEL LAUNCH OF MAIN KERNEL: "<<cudaGetLastError()<<endl;
        checkError(cudaStreamSynchronize(stream[tid]));

        unsigned long long int batchResultSetSize;
        checkError(cudaMemcpy(&batchResultSetSize, dev_batchResultSetSize[tid], sizeof(unsigned long long int), cudaMemcpyDeviceToHost));
        checkError(cudaMemcpy(batchResultSet[tid], dev_batchResultSet[tid], sizeof(int2) * batchResultSetSize, cudaMemcpyDeviceToHost)); 

        printf("Result set size of batch %d: %llu\n", i, batchResultSetSize);
        
        unsigned long long int resultSetStart;
        #pragma omp critical
        {
            resultSetStart = *resultSetSize;
            *resultSetSize += batchResultSetSize;
        }

        #pragma parallel for num_threads(8)
        for (unsigned long long int j=0; j<batchResultSetSize; j++)
        {
            (*resultSet)[resultSetStart+j] = batchResultSet[tid][j];
        }
    }

    for (int i=0; i<GPUSTREAMS; i++)
    {
        cudaStreamDestroy(stream[i]);
        cudaFree(dev_batchResultSet[i]);
        cudaFree(dev_batchResultSetSize[i]);
        cudaFree(batchResultSet[i]);
	}

    ////////////////////////////////////////////////////////////////////////////

    cudaFree(dev_R_data);
    cudaFree(dev_R_offsets);
    cudaFree(dev_S_data);
    cudaFree(dev_S_offsets);
}

void setContainmentJoinGPUUVM(int * R_data, int * R_offsets, int R_size, int * S_data, int * S_offsets, int S_size, int S_elementCount, int largestElement, int2 ** resultSet, unsigned long long int * resultSetSize)
{
    ////////////////////////////////////////////////////////////////////////////
    // Allocate memory
    int R_totalElements = R_offsets[R_size];
    int * dev_R_data;
    int * dev_R_offsets;
    checkError(cudaMalloc((void**)&dev_R_data, R_totalElements * sizeof(int)));
    checkError(cudaMalloc((void**)&dev_R_offsets, (R_size + 1) * sizeof(int)));
    checkError(cudaMemcpy(dev_R_data, R_data, R_totalElements * sizeof(int), cudaMemcpyHostToDevice));
    checkError(cudaMemcpy(dev_R_offsets, R_offsets, (R_size + 1) * sizeof(int), cudaMemcpyHostToDevice));

    int S_totalElements = S_offsets[S_size];
    int * dev_S_data;
    int * dev_S_offsets;
    checkError(cudaMalloc((void**)&dev_S_data, S_totalElements * sizeof(int)));
    checkError(cudaMalloc((void**)&dev_S_offsets, (S_size + 1) * sizeof(int)));
    checkError(cudaMemcpy(dev_S_data, S_data, S_totalElements * sizeof(int), cudaMemcpyHostToDevice));
    checkError(cudaMemcpy(dev_S_offsets, S_offsets, (S_size + 1) * sizeof(int), cudaMemcpyHostToDevice));

    unsigned long long int * dev_resultSetSize;
    checkError(cudaMalloc((void**)&dev_resultSetSize, sizeof(unsigned long long int)));
    checkError(cudaMemset(dev_resultSetSize, 0, sizeof(unsigned long long int)));

    unsigned long long int allocatedElements = ((unsigned long long int)OUTPUTUVMBUFFERSIZE * (1024 * 1024 * 1024)) / sizeof(int2);
	printf("\nNumber of allocated result set elements in managed memory: %llu\n", allocatedElements);

    checkError(cudaMallocManaged((void **)resultSet, sizeof(int2)*allocatedElements));

    // checkGPUMem();
    ////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////
    // Generate result set
    const int TOTALBLOCKS = (R_size + BLOCKSIZE - 1) / BLOCKSIZE;
    printf("\ntotal blocks: %d\n",TOTALBLOCKS);

    kernelFillResultSet<<<TOTALBLOCKS, BLOCKSIZE>>>(dev_R_data, dev_R_offsets, R_size, dev_S_data, dev_S_offsets, 0, S_size, *resultSet, dev_resultSetSize);
    cout<<"** ERROR FROM KERNEL LAUNCH OF MAIN KERNEL: "<<cudaGetLastError()<<endl;
    checkError(cudaDeviceSynchronize());

    checkError(cudaMemcpy(resultSetSize, dev_resultSetSize, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));

    ////////////////////////////////////////////////////////////////////////////

    cudaFree(dev_R_data);
    cudaFree(dev_R_offsets);
    cudaFree(dev_S_data);
    cudaFree(dev_S_offsets);
    cudaFree(dev_resultSetSize);
}

void checkGPUMem()
{
    size_t freeMem, totalMem;
    cudaError_t err = cudaMemGetInfo(&freeMem, &totalMem);
    if (err != cudaSuccess) 
    {
        printf("cudaMemGetInfo failed: %s\n", cudaGetErrorString(err));
    } 
    else 
    {
        printf("GPU memory: free = %.2f GB, total = %.2f GB\n",
            freeMem / 1e9, totalMem / 1e9);
    }
}