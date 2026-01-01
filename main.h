#include "GPU.h"
#include "kernel.h"
#include "params.h"

#include <vector>
#include <string>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cstdint>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

void importDataset(char * filename, vector<vector<int>> * sets, int * totalElements, int * smallestElement, int * largestElement );

void fillData(vector<vector<int>> * sets, int smallestElement, int * data, int * offsets);