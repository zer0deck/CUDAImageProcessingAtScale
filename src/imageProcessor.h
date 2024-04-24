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

// :MARK: Functions declaration




