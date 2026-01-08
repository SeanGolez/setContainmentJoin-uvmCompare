#include "main.h"

using namespace std;

bool compareInt2(const int2& a, const int2& b)
{
    if (a.x != b.x)
    {
        return a.x < b.x;
    }
    return a.y < b.y;
}


int main(int argc, char *argv[])
{
    omp_set_max_active_levels(3);

    ////////////////////////////////////////////////////////////////////////////
    // read input
    if (argc!=3)
	{
	cout << "\nIncorrect number of input parameters.\nShould be R dataset file and S data set file\n";
	return 0;
	}

    char R_Filename[500];
    char S_Filename[500];
	strcpy(R_Filename,argv[1]);
    strcpy(S_Filename,argv[2]);
    ////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////
    // initialize datasets
    int R_elementCount;
    int R_smallestElement;
    int R_largestElement;
    int S_elementCount;
    int S_smallestElement;
    int S_largestElement;
    vector<vector<int>> R_sets;
    vector<vector<int>> S_sets;

    importDataset(R_Filename, &R_sets, &R_elementCount, &R_smallestElement, &R_largestElement);
    printf("Imported %lu sets and %d total elements in R\n", R_sets.size(), R_elementCount);

    importDataset(S_Filename, &S_sets, &S_elementCount, &S_smallestElement, &S_largestElement);
    printf("Imported %lu sets %d total elements in S\n", S_sets.size(), S_elementCount);

    int* R_data = new int[R_elementCount];
    int* R_offsets = new int[R_sets.size()+1];
    int* S_data = new int[S_elementCount];
    int* S_offsets = new int[S_sets.size()+1];

    int largestElement = S_largestElement > R_largestElement ? S_largestElement : R_largestElement;
    int smallestElement = S_smallestElement < R_smallestElement ? S_smallestElement : R_smallestElement;

    fillData(&R_sets, smallestElement, R_data, R_offsets);
    fillData(&S_sets, smallestElement, S_data, S_offsets);
    ////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////
    // Run set containment join (RI-Join)

    int largestElementReindexed = (largestElement-smallestElement);

    // compute result set
    printf("Computing result set...\n");

    double total_tstart = omp_get_wtime();

#if MODE == 0
    vector<int2> resultSet;

    // run algorithm
    double algorithm_tstart = omp_get_wtime();
    setContainmentJoinCPU(R_data, R_offsets, R_sets.size(), S_data, S_offsets, S_sets.size(), S_elementCount, largestElementReindexed, &resultSet);
    printf("CPU Result set has %lu elements\n", resultSet.size());
    double algorithm_tend = omp_get_wtime();
    printf("Generation time: %f\n", algorithm_tend - algorithm_tstart);
    
    // sort result set
    double sort_tstart = omp_get_wtime();
	fprintf(stderr, "Sorting pairs...\n");
	__gnu_parallel::sort(resultSet.begin(), resultSet.end(), compareInt2);
	double sort_tend = omp_get_wtime();
	printf("Sort time: %f\n", (sort_tend - sort_tstart));

#elif MODE == 1
    int2 * resultSet; // C++ style array
    unsigned long long int resultSetSize;

    // run algorithm
    double algorithm_tstart = omp_get_wtime();
    setContainmentJoinGPUBatched(R_data, R_offsets, R_sets.size(), S_data, S_offsets, S_sets.size(), S_elementCount, largestElementReindexed, &resultSet, &resultSetSize);
    printf("GPU Result set has %llu elements\n", resultSetSize);
    double algorithm_tend = omp_get_wtime();
    printf("Generation time: %f\n", algorithm_tend - algorithm_tstart);

    // sort result set
    double sort_tstart = omp_get_wtime();
	fprintf(stderr, "Sorting pairs...\n");
	__gnu_parallel::sort(resultSet, resultSet+resultSetSize, compareInt2);
	double sort_tend = omp_get_wtime();
	printf("Sort time: %f\n", (sort_tend - sort_tstart));

    delete[] resultSet;

#elif MODE == 2
    int2 * resultSet; // CUDA UVM buffer
    unsigned long long int resultSetSize;

    // run algorithm
    double algorithm_tstart = omp_get_wtime();
    setContainmentJoinGPUUVM(R_data, R_offsets, R_sets.size(), S_data, S_offsets, S_sets.size(), S_elementCount, largestElementReindexed, &resultSet, &resultSetSize);
    printf("GPU Result set has %llu elements\n", resultSetSize);
    double algorithm_tend = omp_get_wtime();
    printf("Generation time: %f\n", algorithm_tend - algorithm_tstart);

    // sort result set
    double sort_tstart = omp_get_wtime();
	fprintf(stderr, "Sorting pairs...\n");
	__gnu_parallel::sort(resultSet, resultSet+resultSetSize, compareInt2);
	double sort_tend = omp_get_wtime();
	printf("Sort time: %f\n", (sort_tend - sort_tstart));
    
    cudaFree(resultSet);

#endif

    double total_tend = omp_get_wtime();
    printf("Total time: %f\n", total_tend - total_tstart);

    ////////////////////////////////////////////////////////////////////////////

    delete[] R_data;
    delete[] R_offsets;
    delete[] S_data;
    delete[] S_offsets;

    return 0;
}

void importDataset(char * filename, vector<vector<int>> * sets, int * totalElements, int * smallestElement, int * largestElement )
{
	ifstream in(filename);
    if (!in.is_open())
    {
        cout << "Failed to open file: " << filename << endl;
        return;
    }

    string line;
	*totalElements=0;
    *smallestElement=numeric_limits<int>::max();
    *largestElement=numeric_limits<int>::min();

	while (getline(in, line))
    {
        vector<int> tmpSetData;
	    int i;
		stringstream ss(line);
	    while (ss >> i)
	    {
	        tmpSetData.push_back(i);

            if (i < *smallestElement)
            {
                *smallestElement = i;
            }
            if (i > *largestElement)
            {
                *largestElement = i;
            }
	    }

        sets->push_back(tmpSetData);
        *totalElements += tmpSetData.size();
  	}
}

void fillData(vector<vector<int>> * sets, int smallestElement, int * data, int * offsets)
{
    int index = 0;
    for (int i=0; i<sets->size(); i++)
    {
        offsets[i] = index;

        for (int j=0; j<(*sets)[i].size(); j++)
        {
            int elem = (*sets)[i][j] - smallestElement;
            data[index] = elem;
            index++;
        }
    }

    offsets[sets->size()] = index;
}