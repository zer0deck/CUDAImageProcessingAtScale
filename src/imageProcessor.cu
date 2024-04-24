/*
* Provided by Aleksei Grandilevskii (zer0deck) in 2024;
* Please, refer to me if you are using this code in your work.
* /usr/local/cuda-12.4/bin/nvcc
* nvcc imageProcessor.cu -o imageProcessor.exe -lcurl -I /usr/local/include/opencv4
*/

#include "imageProcessor.h"

// :MARK: Inline System Functions

__host__ inline bool fileExistTest(const string& name) 
{
  struct stat buffer;   
  return (stat (name.c_str(), &buffer) == 0); 
}

__host__ inline void checkCUDAerror(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// :MARK: Processing Data
// loadTestData() function is based on: https://leimao.github.io/blog/Download-Using-LibCur

__host__ bool downloadSingleFile(string fName, string Url)
{
    CURL* curlHandler;
    FILE* file;
    CURLcode res;
    curlHandler = curl_easy_init();

    if (!curlHandler)
    {
        curl_easy_cleanup(curlHandler);
        return false;
    }

    file = fopen(fName.c_str(), "wb");
    curl_easy_setopt(curlHandler, CURLOPT_URL, Url.c_str());
    curl_easy_setopt(curlHandler, CURLOPT_VERBOSE, 0L);
    curl_easy_setopt(curlHandler, CURLOPT_NOPROGRESS, 1L);
    curl_easy_setopt(curlHandler, CURLOPT_WRITEDATA, file);
    res = curl_easy_perform(curlHandler);
    fclose(file);

    if (!res == CURLE_OK)
        return false;

    curl_easy_cleanup(curlHandler);
    return true;  
}

#ifdef TEST_MODE
__host__ bool loadTestData()
{
    int imageCount = sizeof(imageList)/sizeof(string);

    bool flag = false;
    for (int i=0; i < imageCount; i++)
    {
        flag += !fileExistTest(dataPath + imageList[i] + extension);
    }
    if (!flag) 
    {
        cout << "Found all test data." <<endl;
        return true;
    }
    
    indicators::show_console_cursor(false);

    CURL* curlHandler;
    FILE* file;
    CURLcode res;
    curlHandler = curl_easy_init();

    indicators::ProgressBar progressBar{
        indicators::option::BarWidth{30}, indicators::option::Start{" ["},
        indicators::option::Fill{"█"}, indicators::option::Lead{"█"},
        indicators::option::Remainder{"-"}, indicators::option::End{"]"},
        indicators::option::PrefixText{"Downloading data"},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true},
    };

    for (int i=0; i < imageCount; i++)
    {

        if (!curlHandler)
        {
            curl_easy_cleanup(curlHandler);
            indicators::show_console_cursor(true);
            return false;
        }

        file = fopen((dataPath + imageList[i]+ extension).c_str(), "wb");
        curl_easy_setopt(curlHandler, CURLOPT_URL, (sitePermaLink + imageList[i]).c_str());
        curl_easy_setopt(curlHandler, CURLOPT_VERBOSE, 0L);
        curl_easy_setopt(curlHandler, CURLOPT_NOPROGRESS, 1L);
        curl_easy_setopt(curlHandler, CURLOPT_WRITEDATA, file);
        res = curl_easy_perform(curlHandler);
        fclose(file);

        if (!res == CURLE_OK)
            return false;
        
        progressBar.set_option(indicators::option::PostfixText{"Loaded " + to_string(i) + "/" + to_string(imageCount) + ". Now downloading " + imageList[i] + extension}); 
        progressBar.set_progress(100 * i / imageCount);
    }

    progressBar.set_option(indicators::option::PostfixText{"Download completed!"});
    progressBar.set_progress(100);
    curl_easy_cleanup(curlHandler);
    indicators::show_console_cursor(true);

    return true;
}
#endif

__host__ tuple<int, uchar *, uchar *, uchar *> readImageFile(string fileName, int blockSize)
{
    Mat img = imread(fileName, IMREAD_COLOR);
    int numRows = img.rows;
    if (numRows != img.cols)
    {
        fprintf(stderr, "Failed to process image. Width and Height are not the same.\n");
        exit(EXIT_FAILURE);
    }

    size_t sizeToAllocate = sizeof(uchar) * numRows * numRows;
    uchar *hostInputR = (uchar *)malloc(sizeToAllocate);
    uchar *hostInputG = (uchar *)malloc(sizeToAllocate);
    uchar *hostInputB = (uchar *)malloc(sizeToAllocate);

    vector <int> indexList;
    indexList.reserve(sizeToAllocate);
    
    for (int i1 = 0; i1 < numRows*numRows; i1 += numRows*blockSize)
    {
        for (int i2 = 0; i2 < numRows; i2 += blockSize)
        {
            for (int r = 0; r < blockSize; ++r)
            {
                for (int c = 0; c < blockSize; ++c)
                {
                    indexList.push_back(r*numRows + c + i2 + i1);
                }
            }
        }
    }

    int count = 0;
    for (int& inx : indexList) {
        Vec3b intensity = img.at<Vec3b>(inx);
        hostInputR[count] = intensity.val[0];
        hostInputG[count] = intensity.val[1];
        hostInputB[count] = intensity.val[2];
        count++;
    }

    return {numRows, hostInputR, hostInputG, hostInputB};
}

__host__ void SaveReadyBins(uchar *R, uchar *G, uchar *B, int *originalImgSizes, int numImages, const char* outputPath)
{
    // int newBatchSize = originalBatchSize * .5f;
    int shift = 0;
    for (int i=0; i<numImages; ++i)
    {
        int newImgSize = originalImgSizes[i] * .5f;
        size_t sizeToAlloc = sizeof(uchar)*newImgSize*newImgSize;
        uchar *imageData = (uchar*)malloc(sizeToAlloc*4);
        for (int z=0; z<sizeToAlloc*4; ++z)
        {
            imageData[z] = static_cast<uchar>(77);
        }
        for (int j=0; j<sizeToAlloc; ++j)
        {
            imageData[j] = R[shift + j];
            imageData[j+sizeToAlloc] = G[shift + j];
            imageData[j+2*sizeToAlloc] = B[shift + j];
            imageData[j+3*sizeToAlloc] = static_cast<uchar>(0);
        }
        string buf(outputPath);
        buf.append(to_string(i));
        buf.append(".bin");
        ofstream fout(buf.c_str(), ios_base::binary);

        fout.write(reinterpret_cast<const char*>(imageData), sizeToAlloc*4);
        fout.close();

        free(imageData);
        shift += sizeToAlloc;
    }
}

// :MARK: CUDA funtions

__device__ int mergeColor(uchar a1, uchar a2, uchar b1, uchar b2)
{
    return (a1+a2+b1+b2) * .25f;
}

__device__ int calculateCurrentShift()
{
    return blockIdx.x * baseBlockSize * baseBlockSize;
}

__device__ int calculateId(int Shift)
{
    return .25f * Shift + .5f * baseBlockSize * threadIdx.y + threadIdx.x;
}

__device__ int calculateFirstInx()
{
    return 2*(baseBlockSize * threadIdx.y + threadIdx.x);
}

__device__ int calculateSecondInx(int FirstInx)
{
    return FirstInx + baseBlockSize;
}

__global__ void mipMapping(uchar *deviceInputR, uchar *deviceInputG, uchar *deviceInputB, uchar *deviceOutputR, uchar *deviceOutputG, uchar *deviceOutputB)
{
    // есть blockIdx.x - индекс блока
    // есть threadIdx.x и threadIdx.y - индексы тредов

    int Shift = calculateCurrentShift();
    int id = calculateId(Shift);
    int FirstRowInx = calculateFirstInx();
    int SecondRowInx = calculateSecondInx(FirstRowInx);

    if (SecondRowInx + 1 < dataLen)
    {
        deviceOutputR[id] = mergeColor(deviceInputR[FirstRowInx], deviceInputR[FirstRowInx + 1], deviceInputR[SecondRowInx], deviceInputR[SecondRowInx + 1]);
        deviceOutputG[id] = mergeColor(deviceInputG[FirstRowInx], deviceInputG[FirstRowInx + 1], deviceInputG[SecondRowInx], deviceInputG[SecondRowInx + 1]);
        deviceOutputB[id] = mergeColor(deviceInputB[FirstRowInx], deviceInputB[FirstRowInx + 1], deviceInputB[SecondRowInx], deviceInputB[SecondRowInx + 1]);
    }

}

// :MARK: Memory cleanup

__host__ void cleanUpDevice(uchar *deviceInputR, uchar *deviceInputG, uchar *deviceInputB, uchar *deviceOutputR, uchar *deviceOutputG, uchar *deviceOutputB)
{
    cudaError_t err = cudaFree(deviceInputR);
    checkCUDAerror(err);
    err = cudaFree(deviceInputG);
    checkCUDAerror(err);
    err = cudaFree(deviceInputB);
    checkCUDAerror(err);
    err = cudaFree(deviceOutputR);
    checkCUDAerror(err);
    err = cudaFree(deviceOutputG);
    checkCUDAerror(err);
    err = cudaFree(deviceOutputB);
    checkCUDAerror(err);
    err = cudaDeviceReset();
    checkCUDAerror(err);
}

#ifdef DEBUG_MODE
// :MARK: TESTS

__host__ void saveImageBatch(uchar *R, uchar *G, uchar *B, string Name, int batchSize)
{
    Mat compressedImageMat(batchSize, batchSize, CV_8UC3);
    vector<int> compression_params;
    compression_params.push_back(IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);
    for(int r = 0; r < batchSize; ++r)
    {
        for (int c = 0; c < batchSize; ++c)
        {
            Vec3b& bgr = compressedImageMat.at<Vec3b>(r,c);
            bgr[0] = R[r*batchSize+c];
            bgr[1] = G[r*batchSize+c];
            bgr[2] = B[r*batchSize+c];
        }
    }
    string dName = "./.debug/";
    string ext = ".png";
    imwrite((dName + Name + to_string(batchSize) + ext).c_str(), compressedImageMat, compression_params);

    Mat compressedMIPMat(batchSize*2, batchSize*2, CV_8UC3);
    for(int r = 0; r < batchSize; ++r)
    {
        for (int c = 0; c < batchSize; ++c)
        {
            Vec3b& bgr = compressedMIPMat.at<Vec3b>(r,c);
            bgr[0] = R[r*batchSize+c];
            bgr[1] = static_cast<uchar>(0);
            bgr[2] = static_cast<uchar>(0);
        }
    }

    for(int r = batchSize; r < 2*batchSize; ++r)
    {
        for (int c = 0; c < batchSize; ++c)
        {
            Vec3b& bgr = compressedMIPMat.at<Vec3b>(r,c);
            bgr[0] = static_cast<uchar>(0);
            bgr[1] = G[(r-batchSize)*batchSize+c];
            bgr[2] = static_cast<uchar>(0);
        }
    }

    for(int r = 0; r < batchSize; ++r)
    {
        for (int c = batchSize; c < 2*batchSize; ++c)
        {
            Vec3b& bgr = compressedMIPMat.at<Vec3b>(r,c);
            bgr[0] = static_cast<uchar>(0);
            bgr[1] = static_cast<uchar>(0);
            bgr[2] = B[r*batchSize+c-batchSize];
        }
    }
    for(int r = batchSize; r < 2*batchSize; ++r)
    {
        for (int c = batchSize; c < 2*batchSize; ++c)
        {
            Vec3b& bgr = compressedMIPMat.at<Vec3b>(r,c);
            bgr[0] = static_cast<uchar>(255);
            bgr[1] = static_cast<uchar>(255);
            bgr[2] = static_cast<uchar>(255);
        }
    }
    ext = "MIP.png";
    imwrite((dName + Name + to_string(batchSize) + ext).c_str(), compressedMIPMat, compression_params);
}

__host__ void createCPUBatch(uchar *R, uchar *G, uchar *B, string Name, int batchSize)
{
    uchar *nR = (uchar*)malloc(sizeof(uchar)*batchSize*batchSize);
    uchar *nG = (uchar*)malloc(sizeof(uchar)*batchSize*batchSize);
    uchar *nB = (uchar*)malloc(sizeof(uchar)*batchSize*batchSize);

    for(int r = 0; r < batchSize; ++r)
    {
        for (int c = 0; c < batchSize; ++c)
        {
            int inx1 = 2*(batchSize*2*r+c);
            int inx2 = inx1+1;
            int inx3 = inx1+batchSize*2;
            int inx4 = inx3+1;
            nR[r*batchSize + c] = (R[inx1] + R[inx2] + R[inx3] + R[inx4]) * .25f;
            nG[r*batchSize + c] = (G[inx1] + G[inx2] + G[inx3] + G[inx4]) * .25f;
            nB[r*batchSize + c] = (B[inx1] + B[inx2] + B[inx3] + B[inx4]) * .25f;
        }
    }
    saveImageBatch(nR, nG, nB, Name, batchSize);

}
#endif

// :MARK: Main executable fuction

__host__ int main(int argc, char *argv[])
{
    vector<string> filePaths;
    const char* outputPath = NULL;
    cudaError_t err;
    int blockSize = 64;

    // parsing args
    for(int i = 1; i < argc; i++)
    {
        string option(argv[i]);
        i++;
        string value(argv[i]);
        if(option.compare("-f") == 0)
        {
            filePaths.push_back(value);
        } else if (option.compare("-o") == 0)
        {
            outputPath = value.c_str();
        } else if (option.compare("-b") == 0)
        {
            int _t = atoi(value.c_str());
            if (_t>64)
            {
                cerr << "CUDA architecture does not allow batch size more than 64. You input: " << _t << endl;
                return EXIT_FAILURE;
            }
            blockSize = _t;
        }
    }
    if (!outputPath)
#ifdef DEBUG_MODE
        outputPath = "./.debug/";
#else
        outputPath = "./data/output/";
#endif

#ifdef TEST_MODE
    cout<< "TEST_MODE ACTIVE"<<endl;
#elif DEBUG_MODE
    cout<< "DEBUG_MODE ACTIVE"<<endl;
#endif

    bool err_c;
#ifdef TEST_MODE
    err_c = !loadTestData();
#else
    if (filePaths.empty())
    {
        cerr << "Error: No file paths specified. Use '-f' to specify path or enable test mode with flag '-t'" << endl;
        return EXIT_FAILURE;
    }

    for (string & path : filePaths)
        err_c += !fileExistTest(path);
#endif
#ifdef DEBUG_MODE
    if (err_c)
    {
        err_c = false;
        for (string & path : filePaths)
        {
            string File = path.substr(path.rfind("/")+1, path.rfind(".")-path.rfind("/")-1);
            err_c += !downloadSingleFile(path, sitePermaLink+File);
            cout << "Downloaded " << File << " by uri: " << sitePermaLink+File <<endl; 
        }
    }
    else
    {
        cout << "Found data for debugging." << endl;
    }
#endif

    if (err_c)
    {
        cerr << "Error: Catched error during data loading" << endl;
        return EXIT_FAILURE;
    }

    vector<uchar> hostInputR;
    vector<uchar> hostInputG;
    vector<uchar> hostInputB;

    uchar *deviceInputR; 
    uchar *deviceInputG;
    uchar *deviceInputB;
    uchar *deviceOutputR;
    uchar *deviceOutputG;
    uchar *deviceOutputB;


    // Read Image Data
#ifdef TEST_MODE
    int numFiles = sizeof(imageList)/sizeof(string);
#else
    int numFiles = filePaths.size();
#endif

    int *imgSizes = new int(numFiles);

#ifdef TEST_MODE 
    cout << "Images to process: " << numFiles << endl;
#endif

    for (int i = 0; i < numFiles; ++i)
    {
#ifdef TEST_MODE
        string imagePath = dataPath+imageList[i]+extension;
#else   
        string imagePath = filePaths[i];
#endif
        tuple<int, uchar *, uchar *, uchar *> imageData = readImageFile(imagePath, blockSize);
        imgSizes[i] = get<0>(imageData);
        int tempLen = pow(imgSizes[i], 2);

        hostInputR.insert(hostInputR.end(), get<1>(imageData), get<1>(imageData) + tempLen);
        hostInputG.insert(hostInputG.end(), get<2>(imageData), get<2>(imageData) + tempLen);
        hostInputB.insert(hostInputB.end(), get<3>(imageData), get<3>(imageData) + tempLen);
    }
    
#ifdef DEBUG_MODE
    saveImageBatch(hostInputR.data(), hostInputG.data(), hostInputB.data(), "INPUT", blockSize);
    cout << "Stored input image." << endl;
    createCPUBatch(hostInputR.data(), hostInputG.data(), hostInputB.data(), "CPU", blockSize*.5f);
    cout << "Stored CPU processed image." << endl;
#endif

    int uncompressedSize = hostInputR.size();
    int compressedSize = uncompressedSize * .25f;

    uchar *hostOutputR = (uchar *)malloc(sizeof(uchar) * compressedSize);
    uchar *hostOutputG = (uchar *)malloc(sizeof(uchar) * compressedSize);
    uchar *hostOutputB = (uchar *)malloc(sizeof(uchar) * compressedSize);

    // Allocate and fill CUDA memory
    cudaMemcpyToSymbol(baseBlockSize, &blockSize, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(dataLen, &uncompressedSize, sizeof(int), 0, cudaMemcpyHostToDevice);

    err = cudaMalloc(&deviceInputR, uncompressedSize);
    checkCUDAerror(err);
    err = cudaMalloc(&deviceInputG, uncompressedSize);
    checkCUDAerror(err);
    err = cudaMalloc(&deviceInputB, uncompressedSize);
    checkCUDAerror(err);
    err = cudaMalloc(&deviceOutputR, sizeof(uchar) * compressedSize);
    checkCUDAerror(err);
    err = cudaMalloc(&deviceOutputG, sizeof(uchar) * compressedSize);
    checkCUDAerror(err);
    err = cudaMalloc(&deviceOutputB, sizeof(uchar) * compressedSize);
    checkCUDAerror(err);

    err = cudaMemcpy(deviceInputR, hostInputR.data(), uncompressedSize, cudaMemcpyHostToDevice);
    checkCUDAerror(err);
    err = cudaMemcpy(deviceInputG, hostInputG.data(), uncompressedSize, cudaMemcpyHostToDevice);
    checkCUDAerror(err);
    err = cudaMemcpy(deviceInputB, hostInputB.data(), uncompressedSize, cudaMemcpyHostToDevice);
    checkCUDAerror(err);

    // We set gridWide to [0.5 of blockCount + 1] to ease the computation
    // In practice we will have 2-3% of more threads as we need, but easy cut with 'dataLen' var
    int gridWide = uncompressedSize/(blockSize*blockSize);
#ifdef DEBUG_MODE
    cout << "Compressed size: " << compressedSize << "." <<endl;
    cout << "Num of grid: " << gridWide << "." << endl;
#endif
    // dim3 grid(2, gridWide);
    // Block is set to 1/4 blockSize as this is MIP-mapping alghorithm 
    int bs = blockSize * .5f;
    dim3 block(bs, bs);
    mipMapping<<<gridWide, block>>>(deviceInputR, deviceInputG, deviceInputB, deviceOutputR, deviceOutputG, deviceOutputB);

    err = cudaMemcpy(hostOutputR, deviceOutputR, compressedSize, cudaMemcpyDeviceToHost);
    checkCUDAerror(err);
    err = cudaMemcpy(hostOutputG, deviceOutputG, compressedSize, cudaMemcpyDeviceToHost);
    checkCUDAerror(err);
    err = cudaMemcpy(hostOutputB, deviceOutputB, compressedSize, cudaMemcpyDeviceToHost);
    checkCUDAerror(err);

#ifdef DEBUG_MODE
    saveImageBatch(hostOutputR, hostOutputG, hostOutputB, "CUDA", blockSize*.5f);
    cout << "Stored CUDA processed image." << endl;
#endif

    cleanUpDevice(deviceInputR, deviceInputG, deviceInputB, deviceOutputR, deviceOutputG, deviceOutputB);

    vector<uchar>().swap(hostInputR);
    vector<uchar>().swap(hostInputG);
    vector<uchar>().swap(hostInputB);

    // for (int l=65000; l<65536; ++l)
    //     cout <<hostOutputR[l] << endl;
    SaveReadyBins(hostOutputR, hostOutputG, hostOutputB, imgSizes, numFiles, outputPath);
    cout << "Done: " << numFiles << " images compression." << endl;
    cout << numFiles << "bins saved to " << outputPath << endl;
    return EXIT_SUCCESS;
}
