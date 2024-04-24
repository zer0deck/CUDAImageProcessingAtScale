// :MARK: Includes

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
// #include <numeric>
// #include <iterator>
#include <curl/curl.h>
#include <curl/easy.h>
#include <sys/stat.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

// #include <chrono>

#include "indicators.hpp"

// :MARK: Space setup

using namespace std;
using namespace cv;

// :MARK: Constants defenition

__constant__ int baseBlockSize;
__constant__ int dataLen;
#if defined TEST_MODE || defined DEBUG_MODE
const string dataPath = "./data/";
const string sitePermaLink = "https://sipi.usc.edu/database/download.php?vol=aerials&img=";
const string extension = ".tiff";
const string imageList[] = {
    "2.1.02",
    "2.1.03",
    "2.1.04",
    "2.1.05",
    "2.1.06",
    "2.1.07",
    "2.1.08",
    "2.1.09",
    "2.1.10",
    "2.1.11",
    "2.1.12",
    "2.2.01",
    "2.2.02",
    "2.2.03",
    "2.2.04",
    "2.2.05",
    "2.2.06",
    "2.2.07",
    "2.2.08",
    "2.2.09",
    "2.2.10",
    "2.2.11",
    "2.2.12",
    "2.2.13",
    "2.2.14",
    "2.2.15",
    "2.2.16",
    "2.2.17",
    "2.2.18",
    "2.2.19",
    "2.2.20",
    "2.2.21",
    "2.2.22",
    "2.2.23",
    "2.2.24",
    "3.2.25",
    "wash-ir"
};
#endif

// :MARK: Functions declaration

__host__ inline bool fileExistTest(const string& name);
__host__ inline void checkCUDAerror(cudaError_t err);

__host__ bool downloadSingleFile(string fName, string Url);
__host__ tuple<int, uchar *, uchar *, uchar *> readImageFile(string fileName, int blockSize);
__host__ void SaveReadyBins(uchar *R, uchar *G, uchar *B, int *originalImgSizes, int numImages, const char* outputPath);

__device__ int mergeColor(uchar a1, uchar a2, uchar b1, uchar b2);
__device__ int calculateCurrentShift();
__device__ int calculateId(int Shift);
__device__ int calculateFirstInx();
__device__ int calculateSecondInx(int FirstInx);
__global__ void mipMapping(
    uchar *deviceInputR, 
    uchar *deviceInputG, 
    uchar *deviceInputB, 
    uchar *deviceOutputR, 
    uchar *deviceOutputG, 
    uchar *deviceOutputB
    );

__host__ void cleanUpDevice(
    uchar *deviceInputR, 
    uchar *deviceInputG, 
    uchar *deviceInputB, 
    uchar *deviceOutputR, 
    uchar *deviceOutputG, 
    uchar *deviceOutputB
    );


#ifdef DEBUG_MODE
__host__ void saveImageBatch(uchar *R, uchar *G, uchar *B, string Name, int batchSize);
__host__ void createCPUBatch(uchar *R, uchar *G, uchar *B, string Name, int batchSize);
#endif

#ifdef TEST_MODE
__host__ bool loadTestData();
#endif
