#define _USE_MATH_DEFINES

#include <iostream>
#include <fstream>
#include <math.h>
#include <map>

using namespace std;

#define TOTAL_DEGREES 180
#define BINS_PER_DEGREE 4
#define TOTAL_BINS (TOTAL_DEGREES * BINS_PER_DEGREE + 1) // + 1 for the last degree bin
#define PARAMETERS 2

const float ARCMIN_TO_RADIAN = M_PI / (180 * 60);

float arcminToRadian(float arcmin)
{
	return arcmin * ARCMIN_TO_RADIAN;
}

float angleBetweenGalaxies(float rightAscension1, float declination1, float rightAscension2, float declination2)
{
	float expression = sin(declination1) * sin(declination2) + cos(declination1) * cos(declination2) * cos(rightAscension1 - rightAscension2);

	if (expression > 1)
		expression = 1.0;
	else if (expression < -1)
		expression = -1.0;

	return acos(expression);
}

void calculateHistogram(float *galaxy1, float *galaxy2, int n, unsigned int *histogramBins)
{
	cout << "Calculating histogram..." << endl;
	float radianToDegree = 180 / M_PI;
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			float angleRadians = angleBetweenGalaxies(galaxy1[i * PARAMETERS], galaxy1[i * PARAMETERS + 1], galaxy2[j * PARAMETERS], galaxy2[j * PARAMETERS + 1]); // in radians
			float angleDegrees = angleRadians * radianToDegree;																									   // in degrees

			int bin = floor(angleDegrees * BINS_PER_DEGREE);
			histogramBins[bin]++;
		}
	}
}

float *readGalaxies(ifstream &file, int n)
{
	float *galaxies = new float[n * PARAMETERS]; // right ascension in arc minutes and declination in arc minutes - LINEARIZED
	for (int i = 0; i < n; i++)
	{
		float right_ascension, declination;
		file >> right_ascension >> declination;

		// Convert arc minutes to radians
		galaxies[i * PARAMETERS] = arcminToRadian(right_ascension);
		galaxies[i * PARAMETERS + 1] = arcminToRadian(declination);
	}
	return galaxies;
}

void printHistogram(unsigned int *histogramBins, int n, string path)
{
	unsigned long long sum = 0;

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

int main()
{
	// Files
	ifstream realData("C:/Users/masac/OneDrive/Desktop/Masa/College/GPU Programming/real-galaxies.txt");
	ifstream randomData("C:/Users/masac/OneDrive/Desktop/Masa/College/GPU Programming/random-galaxies.txt");
	ofstream omegaFile("C:/Users/masac/OneDrive/Desktop/Masa/College/GPU Programming/omega.txt");

	// File opening check
	if (!realData.is_open() || !randomData.is_open() || !omegaFile.is_open())
	{
		cout << "Error opening files." << endl;
		return -1;
	}

	// Read number of galaxies
	int n1, n2;
	realData >> n1;
	randomData >> n2;

	// Check if the number of galaxies is the same
	if (n1 != n2)
	{
		cout << "The number of galaxies in the two files is not the same." << endl;
		return -1;
	}

	int n = n1;

	// Read real galaxies
	float *realGalaxies = readGalaxies(realData, n);

	// Read random galaxies
	float *randomGalaxies = readGalaxies(randomData, n);

	// Close the files
	realData.close();
	randomData.close();

	// Histograms
	unsigned int *histogramDD = new unsigned int[TOTAL_BINS];
	fill(histogramDD, histogramDD + TOTAL_BINS, 0);
	unsigned int *histogramDR = new unsigned int[TOTAL_BINS];
	fill(histogramDR, histogramDR + TOTAL_BINS, 0);
	unsigned int *histogramRR = new unsigned int[TOTAL_BINS];
	fill(histogramRR, histogramRR + TOTAL_BINS, 0);

	// Calculate the histograms
	calculateHistogram(realGalaxies, realGalaxies, n, histogramDD);
	calculateHistogram(realGalaxies, randomGalaxies, n, histogramDR);
	calculateHistogram(randomGalaxies, randomGalaxies, n, histogramRR);

	// Print the histograms
	cout << "histogramDD Histogram:" << endl;
	printHistogram(histogramDD, n, "C:/Users/masac/OneDrive/Desktop/Masa/College/GPU Programming/histogramDD.txt");

	cout << "histogramDR Histogram:" << endl;
	printHistogram(histogramDR, n, "C:/Users/masac/OneDrive/Desktop/Masa/College/GPU Programming/histogramDR.txt");

	cout << "histogramRR Histogram:" << endl;
	printHistogram(histogramRR, n, "C:/Users/masac/OneDrive/Desktop/Masa/College/GPU Programming/histogramRR.txt");

	// Calculating omega values
	float *omega = new float[TOTAL_BINS];
	for (int i = 0; i < TOTAL_BINS; i++)
	{
		float histogramDDValue = histogramDD[i];
		float drValue = histogramDR[i];
		float rrValue = histogramRR[i];

		omega[i] = (histogramDDValue - 2 * drValue + rrValue) / rrValue;
	}

	// Write omega values to file
	for (int i = 0; i < TOTAL_BINS; i++)
	{
		omegaFile << omega[i] << endl;
	}
	omegaFile.close();

	// Delete dynamic memory
	delete[] realGalaxies;
	delete[] randomGalaxies;
	delete[] histogramDD;
	delete[] histogramDR;
	delete[] histogramRR;
	delete[] omega;

	return 0;
}
