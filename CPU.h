#include <vector>

using namespace std;

void constructInvertedListCPU(int * S_data, int * S_offsets, int S_start, int S_end, int largestElement, int * I_data, int * I_offsets);

void computeResultSetCPU(int * R_data, int * R_offsets, int R_size, int S_size, int * I_data, int * I_offsets, vector<int2> * resultSet);

void setContainmentJoinCPU(int * R_data, int * R_offsets, int R_size, int * S_data, int * S_offsets, int S_size, int S_elementCount, int largestElement, vector<int2> * resultSet);

void touchArray(int2 *array, unsigned long long int length);