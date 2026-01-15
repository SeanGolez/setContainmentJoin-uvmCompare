#include "CPU.h"

void constructInvertedListCPU(int * S_data, int * S_offsets, int S_start, int S_end, int largestElement, int * I_data, int * I_offsets)
{
    int* I_counts = new int[largestElement+1]();
    for (int i=S_start; i<S_end; i++)
    {
        for (int j=S_offsets[i]; j<S_offsets[i+1]; j++)
        {
            int elem = S_data[j];
            I_counts[elem]++;
        }
    }

    I_offsets[0] = 0;
    for (int i=1; i<=largestElement; i++)
    {
        I_offsets[i] = I_offsets[i-1] + I_counts[i-1];
    }
    I_offsets[largestElement+1] = I_offsets[largestElement] + I_counts[largestElement];

    int* I_indexes = new int[largestElement+1]();
    for (int i=S_start; i<S_end; i++)
    {
        for (int j=S_offsets[i]; j<S_offsets[i+1]; j++)
        {
            int elem = S_data[j];
            I_data[I_offsets[elem] + I_indexes[elem]] = i;
            I_indexes[elem]++;
        }
    }

    delete[] I_counts;
    delete[] I_indexes;
}

void computeResultSetCPU(int * R_data, int * R_offsets, int R_size, int S_size, int * I_data, int * I_offsets, vector<int2> * resultSet)
{
    int * count = new int[S_size]();
    int * touched = new int[S_size];

    for (int i=0; i<R_size; i++)
    {
        int touchedLength = 0;

        for (int j=R_offsets[i]; j<R_offsets[i+1]; j++)
        {
            int elem = R_data[j];

            for (int k=I_offsets[elem]; k<I_offsets[elem+1]; k++)
            {
                int s_index = I_data[k];

                if (count[s_index] == 0)
                {
                    touched[touchedLength] = s_index;
                    touchedLength++;
                }

                count[s_index]++;
            }
        }
        
        // update result set
        int rLength = R_offsets[i+1] - R_offsets[i];

        for (int j=0; j<touchedLength; j++) 
        {
            int s_index = touched[j];
            if (count[s_index] == rLength)
            {
                int2 tmp;
                tmp.x = i;
                tmp.y = s_index;
                resultSet->push_back(tmp);
            }
            count[s_index] = 0;
        }
    }

    delete[] count;
    delete[] touched;
}

void setContainmentJoinCPU(int * R_data, int * R_offsets, int R_size, int * S_data, int * S_offsets, int S_size, int S_elementCount, int largestElement, vector<int2> * resultSet)
{
    // construct inverted list
    int* I_data = new int[S_elementCount];
    int* I_offsets = new int[largestElement+2];
    constructInvertedListCPU(S_data, S_offsets, 0, S_size, largestElement, I_data, I_offsets);

    computeResultSetCPU(R_data, R_offsets, R_size, S_size, I_data, I_offsets, resultSet);
}

void touchArray(int2 *array, unsigned long long int length)
{
    volatile int sink_x, sink_y;

    for (unsigned long long int i = 0; i < length; i++)
    {
        sink_x = array[i].x;
        sink_y = array[i].y;
    }
}