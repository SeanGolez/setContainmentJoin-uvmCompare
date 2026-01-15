#include "GPU.h"

struct compare_int2{
	__host__ __device__ bool operator()(int2 a,int2 b){return (a.x!=b.x) ? (a.x<b.x):(a.y<b.y);}
};

bool compareInt2(const int2& a, const int2& b)
{
    if (a.x != b.x)
    {
        return a.x < b.x;
    }
    return a.y < b.y;
}

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

    int R_offsetRate=1.0 / SAMPLERATE;

    const int TOTALBLOCKSBATCHEST = ((R_size * SAMPLERATE) + BLOCKSIZE - 1) / BLOCKSIZE;
    printf("\ntotal blocks: %d\n",TOTALBLOCKSBATCHEST);

    kernelCountResultSetSize<<<TOTALBLOCKSBATCHEST, BLOCKSIZE>>>(dev_R_data, dev_R_offsets, R_offsetRate, R_size, dev_S_data, dev_S_offsets, S_size, dev_resultSetSize);
    cout<<"** ERROR FROM KERNEL LAUNCH OF BATCH ESTIMATOR: "<<cudaGetLastError()<<endl;
    checkError(cudaDeviceSynchronize());

    checkError(cudaMemcpy(&sampleResultSetSize, dev_resultSetSize, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));

    cudaFree(dev_resultSetSize);

    unsigned long long int estimatedResultSetSize = sampleResultSetSize * R_offsetRate * 1.15;
    printf("estimatedResultSetSize: %llu\n", estimatedResultSetSize);
    ////////////////////////////////////////////////////////////////////////////
    
    ////////////////////////////////////////////////////////////////////////////
    // Estimate batches needed
    int numBatches = (estimatedResultSetSize) / GPUBUFFERSIZE;
    if (numBatches < 3)
    {
        numBatches = 3;
    }

    // batch on R
    unsigned long long int R_batchSize = R_size / numBatches;

    printf("numBatches: %d\n", numBatches);

    int batchesThatHaveOneMore = R_size - (R_batchSize * numBatches); //batch number 0-
    ////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////
    // Generate result set

    cudaStream_t stream[GPUSTREAMS];
	for (int i=0; i<GPUSTREAMS; i++){
	    cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking);
	}	

    unsigned long long int * dev_batchResultSetSize[GPUSTREAMS];
    int2 * dev_batchResultSet[GPUSTREAMS];
	int2 * batchResultSet[GPUSTREAMS];  // pinned memory
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

        int thread_batchOffset = (i * R_batchSize);
        int thread_batchSize = R_batchSize;
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

        const int TOTALBLOCKS = (thread_batchSize + BLOCKSIZE - 1) / BLOCKSIZE;
        printf("\ntotal blocks: %d\n",TOTALBLOCKS);

        kernelFillResultSet<<<TOTALBLOCKS, BLOCKSIZE, 0, stream[tid]>>>(dev_R_data, dev_R_offsets, thread_batchOffset, thread_batchSize, dev_S_data, dev_S_offsets, S_size, dev_batchResultSet[tid], dev_batchResultSetSize[tid]);
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
    checkError(cudaMallocManaged((void**)&dev_resultSetSize, sizeof(unsigned long long int)));
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

    cudaEvent_t kernelStart;
	cudaEvent_t kernelStop;
    cudaEventCreate(&kernelStart);
	cudaEventCreate(&kernelStop);

	cudaEventRecord(kernelStart);
    kernelFillResultSet<<<TOTALBLOCKS, BLOCKSIZE>>>(dev_R_data, dev_R_offsets, 0, R_size, dev_S_data, dev_S_offsets, S_size, *resultSet, dev_resultSetSize);
    cout<<"** ERROR FROM KERNEL LAUNCH OF MAIN KERNEL: "<<cudaGetLastError()<<endl;
    cudaEventRecord(kernelStop);

#if PROBEANDSTORE == 1
    probeAndStore(*resultSet, dev_resultSetSize, &kernelStop);
#endif

    cudaEventSynchronize(kernelStop);

	float milliseconds;
	cudaEventElapsedTime(&milliseconds, kernelStart, kernelStop);
	printf("\nKernel execution time: %f\n", (milliseconds / 1000));
	cudaEventDestroy(kernelStart);
	cudaEventDestroy(kernelStop);

    *resultSetSize = *dev_resultSetSize;

    ////////////////////////////////////////////////////////////////////////////

    cudaFree(dev_R_data);
    cudaFree(dev_R_offsets);
    cudaFree(dev_S_data);
    cudaFree(dev_S_offsets);
    cudaFree(dev_resultSetSize);
}

void probeAndStore(int2 * array, unsigned long long int * cnt, cudaEvent_t * kernelStop)
{
	// set bounds
	uint64_t lowerBound = 0;
	uint64_t upperBound = 0;

	// get the number of elements in one page
	unsigned int elemsPerPage = ((PAGESIZE) * (1024)) / sizeof(int2);
	printf("\nelemsPerPage: %u", elemsPerPage);

	// keep track of the number of pages stored
	unsigned long long int pagesStored = 0;

	unsigned long long int localCnt;
	unsigned long long int localPagesIdx;

	// loop while kernel is not complete
	while( cudaEventQuery(*kernelStop) == cudaErrorNotReady )
	{
		// sleep for SLEEPSEC seconds
		usleep(SLEEPSEC*1000000);

		// read cnt
		localCnt = *cnt;
		// printf("\nNum elems generated in array on GPU: %llu", localCnt);

		// convert count to pages
		localPagesIdx = localCnt  / elemsPerPage;
		// printf("\nPage index being updated on GPU: %llu", localPagesIdx);

		// set bounds
		lowerBound = pagesStored * elemsPerPage;
		upperBound = localPagesIdx * elemsPerPage;

		// store
		// printf("\nStoring %lu elements in bounds [%lu, %lu)", (upperBound-lowerBound), lowerBound, upperBound);
        touchArray(array + lowerBound, upperBound - lowerBound);

		// update pages sorted count
		pagesStored = localPagesIdx;
	}

	// read cnt
	localCnt = *cnt;
	// printf("\n[Leftover] Num elems generated in array on GPU: %llu", localCnt);

	// set bounds
	lowerBound = pagesStored * elemsPerPage;
	upperBound = localCnt;

	// store
    // printf("\nStoring %lu elements in bounds [%lu, %lu)", (upperBound-lowerBound), lowerBound, upperBound);
    touchArray(array + lowerBound, upperBound - lowerBound);
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