#include <cuda.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <map>
#include <sys/time.h>

using namespace std;

#define TOTAL_DEGREES 180
#define BINS_PER_DEGREE 4
#define TOTAL_BINS (TOTAL_DEGREES * BINS_PER_DEGREE + 1) // + 1 for the last degree bin
#define PARAMETERS 2

const float ARCMIN_TO_RADIAN = M_PI / (180 * 60);

const int NUM_THREADS = 256;

__device__ float angleBetweenGalaxies(float rightAscension1, float declination1, float rightAscension2, float declination2)
{
    float expression = sin(declination1) * sin(declination2) + cos(declination1) * cos(declination2) * cos(rightAscension1 - rightAscension2);

    if (expression > 1)
        expression = 1.0;
    else if (expression < -1)
        expression = -1.0;

    return acos(expression);
}

__global__ void calculateHistogram(float *galaxy1, float *galaxy2, int n, unsigned int *histogramBins)
{
    // Each thread will cover n galaxies
    // Galaxy1 is fixed and galaxy2 is the n galaxies a thread will cover

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n)
    {
        float rightAscension1 = galaxy1[index * PARAMETERS];
        float declination1 = galaxy1[index * PARAMETERS + 1];
        float radianToDegree = 180 / M_PI;

        for (int i = 0; i < n; i++)
        {
            float rightAscension2 = galaxy2[i * PARAMETERS];
            float declination2 = galaxy2[i * PARAMETERS + 1];

            // Calculate the angle between the two galaxies
            float angleRadians = angleBetweenGalaxies(rightAscension1, declination1, rightAscension2, declination2);
            float angleDegrees = angleRadians * radianToDegree;

            // Calculate the bin
            int bin = floor(angleDegrees * BINS_PER_DEGREE);

            // Increment the bin
            atomicAdd(&histogramBins[bin], 1);
        }
    }
}

int main(int argc, char *argv[])
{
    // Functions
    float arcminToRadian(float arcmin);
    float *readGalaxies(ifstream & file, int n);
    void printHistogram(unsigned int *histogramBins, unsigned long long &sum, int n, string path);

    int getDevice(int deviceno);
    void checkError(cudaError_t error, string message);

    // Variables
    float *realGalaxies, *randomGalaxies;                  // CPU
    unsigned int *histogramDR, *histogramDD, *histogramRR; // CPU
    float *omega_values;                                   // CPU

    float *devRealGalaxies, *devRandomGalaxies;                     // GPU
    unsigned int *devHistogramDR, *devHistogramDD, *devHistogramRR; // GPU

    unsigned long long histogramDRsum, histogramDDsum, histogramRRsum;
    int nreal, nrandom, n;
    int numBlocks;

    double start, end, kerneltime;
    struct timeval _ttime;
    struct timezone _tzone;
    cudaError_t myError;

    // Check arguments
    if (argc != 4)
    {
        cout << "Usage: a.out realData randomData output_data" << endl;
        return (-1);
    }

    // Select device
    if (getDevice(0) != 0)
    {
        cout << "Error in selecting device" << endl;
        return (-1);
    }

    kerneltime = 0.0;
    gettimeofday(&_ttime, &_tzone);
    start = (double)_ttime.tv_sec + (double)_ttime.tv_usec / 1000000.;

    // Read data
    ifstream realData(argv[1]);
    if (!realData.is_open())
    {
        cout << "Error opening real data file" << endl;
        return (-1);
    }

    ifstream randomData(argv[2]);
    if (!randomData.is_open())
    {
        cout << "Error opening random data file" << endl;
        return (-1);
    }

    ofstream output_data(argv[3]);
    if (!output_data.is_open())
    {
        cout << "Error opening output data file" << endl;
        return (-1);
    }

    realData >> nreal;
    randomData >> nrandom;

    // Check if the number of real and random galaxies is the same
    if (nreal != nrandom)
    {
        cout << "Error: Number of real and random galaxies must be the same" << endl;
        return (-1);
    }

    n = nreal;

    realGalaxies = readGalaxies(realData, n);
    randomGalaxies = readGalaxies(randomData, n);

    // Close files
    realData.close();
    randomData.close();

    // Initialization
    histogramDR = new unsigned int[TOTAL_BINS];
    histogramDD = new unsigned int[TOTAL_BINS];
    histogramRR = new unsigned int[TOTAL_BINS];
    omega_values = new float[TOTAL_BINS];
    histogramDRsum = 0;
    histogramDDsum = 0;
    histogramRRsum = 0;

    // Allocate memory on the GPU
    myError = cudaMalloc((void **)&devRealGalaxies, n * PARAMETERS * sizeof(float));
    checkError(myError, "Error in allocating memory for real galaxies on the GPU");

    myError = cudaMalloc((void **)&devRandomGalaxies, n * PARAMETERS * sizeof(float));
    checkError(myError, "Error in allocating memory for random galaxies on the GPU");

    myError = cudaMalloc((void **)&devHistogramDR, TOTAL_BINS * sizeof(unsigned int));
    checkError(myError, "Error in allocating memory for histogram DR on the GPU");
    cudaMemset(devHistogramDR, 0, TOTAL_BINS * sizeof(unsigned int));

    myError = cudaMalloc((void **)&devHistogramDD, TOTAL_BINS * sizeof(unsigned int));
    checkError(myError, "Error in allocating memory for histogram DD on the GPU");
    cudaMemset(devHistogramDD, 0, TOTAL_BINS * sizeof(unsigned int));

    myError = cudaMalloc((void **)&devHistogramRR, TOTAL_BINS * sizeof(unsigned int));
    checkError(myError, "Error in allocating memory for histogram RR on the GPU");
    cudaMemset(devHistogramRR, 0, TOTAL_BINS * sizeof(unsigned int));

    // Copy data to the GPU
    myError = cudaMemcpy(devRealGalaxies, realGalaxies, n * PARAMETERS * sizeof(float), cudaMemcpyHostToDevice);
    checkError(myError, "Error in copying real galaxies to the GPU");

    myError = cudaMemcpy(devRandomGalaxies, randomGalaxies, n * PARAMETERS * sizeof(float), cudaMemcpyHostToDevice);
    checkError(myError, "Error in copying random galaxies to the GPU");

    // Run the kernels on the GPU
    numBlocks = (n + NUM_THREADS - 1) / NUM_THREADS;
    calculateHistogram<<<numBlocks, NUM_THREADS>>>(devRealGalaxies, devRandomGalaxies, n, devHistogramDR);
    calculateHistogram<<<numBlocks, NUM_THREADS>>>(devRealGalaxies, devRealGalaxies, n, devHistogramDD);
    calculateHistogram<<<numBlocks, NUM_THREADS>>>(devRandomGalaxies, devRandomGalaxies, n, devHistogramRR);

    // Copy the results back to the CPU
    myError = cudaMemcpy(histogramDR, devHistogramDR, TOTAL_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    checkError(myError, "Error in copying histogram DR to the CPU");

    myError = cudaMemcpy(histogramDD, devHistogramDD, TOTAL_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    checkError(myError, "Error in copying histogram DD to the CPU");

    myError = cudaMemcpy(histogramRR, devHistogramRR, TOTAL_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    checkError(myError, "Error in copying histogram RR to the CPU");

    // Print the histograms
    cout << "Histogram DR" << endl;
    printHistogram(histogramDR, histogramDRsum, n, "dr.txt");

    cout << "Histogram DD" << endl;
    printHistogram(histogramDD, histogramDDsum, n, "dd.txt");

    cout << "Histogram RR" << endl;
    printHistogram(histogramRR, histogramRRsum, n, "rr.txt");

    // Calculate omega values on the CPU
    for (int i = 0; i < TOTAL_BINS; i++)
    {
        float ddValue = histogramDD[i];
        float drValue = histogramDR[i];
        float rrValue = histogramRR[i];

        omega_values[i] = (ddValue - 2 * drValue + rrValue) / rrValue; // possible division by zero - expect nan value
    }

    // Write the omega values to the output file
    for (int i = 0; i < TOTAL_BINS; i++)
    {
        output_data << omega_values[i] << endl;
    }
    output_data.close();

    gettimeofday(&_ttime, &_tzone);
    end = (double)_ttime.tv_sec + (double)_ttime.tv_usec / 1000000.;
    kerneltime += end - start;
    cout << "Kernel time = " << kerneltime << " secs" << endl;

    // Delete memory
    delete[] realGalaxies;
    delete[] randomGalaxies;
    delete[] histogramDR;
    delete[] histogramDD;
    delete[] histogramRR;
    delete[] omega_values;

    cudaFree(devRealGalaxies);
    cudaFree(devRandomGalaxies);
    cudaFree(devHistogramDR);
    cudaFree(devHistogramDD);
    cudaFree(devHistogramRR);

    return (0);
}

void printHistogram(unsigned int *histogramBins, unsigned long long &sum, int n, string path)
{
    for (int i = 0; i < TOTAL_BINS; i++)
    {
        sum += histogramBins[i];
    }

    cout << "Histogram sum: " << sum;

    if (sum == pow(n, 2))
        cout << " - Correct!" << endl;
    else
        cout << " - Incorrect!" << endl;

    ofstream file(path);
    if (!file.is_open())
    {
        cout << "Error opening file" << endl;
        return;
    }

    for (int i = 0; i < TOTAL_BINS; i++)
    {
        file << histogramBins[i] << endl;
    }
}

void checkError(cudaError_t error, string message)
{
    if (error != cudaSuccess)
    {
        cout << message << endl;
        cout << "Error: " << cudaGetErrorString(error) << endl;
        exit(-1);
    }
}

float arcminToRadian(float arcmin)
{
    return arcmin * ARCMIN_TO_RADIAN;
}

float *readGalaxies(ifstream &file, int n)
{
    float *galaxies = new float[n * PARAMETERS]; // right ascension in arc minutes and declination in arc minutes - LINEARIZED
    for (int i = 0; i < n; i++)
    {
        float right_ascension, declination;
        file >> right_ascension >> declination;

        // Converting arc minutes to radians
        galaxies[i * PARAMETERS] = arcminToRadian(right_ascension);
        galaxies[i * PARAMETERS + 1] = arcminToRadian(declination);
    }
    return galaxies;
}

int getDevice(int deviceNo)
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    cout << "Number of CUDA devices = " << deviceCount << endl;

    if (deviceCount < 0 || deviceCount > 128)
        return (-1);

    int device;

    cudaSetDevice(deviceNo);
    cudaGetDevice(&device);
    if (device != deviceNo)
        cout << "Unable to set device " << deviceNo << ", using device " << device << " instead" << endl;
    else
        cout << "Device " << device << " selected" << endl;

    return (0);
}
