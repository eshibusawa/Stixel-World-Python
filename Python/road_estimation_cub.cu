// This file is part of Stixel-World-Python
// Copyleft 2023, Eijiro Shibusawa <phd_kimberlite@yahoo.co.jp>
// This file is licensed under the GPL-3.0 license.

#include <cub/block/block_histogram.cuh>

extern "C" __global__ void computeHistogram(int* output,
		const float* __restrict__ inputDisparity)
{
	// ROAD_ESTIMATION_HEIGHT_DISPARITY = height of raw disparity
	// ROAD_ESTIMATION_WIDTH_DISPARITY = width of raw disparity
	// ROAD_ESTIMATION_MAX_DISPARITY = max disparity
	int indexY = blockIdx.x;
	int indexX = threadIdx.x;
	if (indexY >= ROAD_ESTIMATION_HEIGHT_DISPARITY)
	{
		return;
	}
	typedef cub::BlockHistogram<int, ROAD_ESTIMATION_HISTOGRAM_THREADS, ROAD_ESTIMATION_HISTOGRAM_ITEMS, ROAD_ESTIMATION_MAX_DISPARITY> BlockHistogram;
	__shared__ typename BlockHistogram::TempStorage tempStorage;
	int disparityValues[ROAD_ESTIMATION_HISTOGRAM_ITEMS];

	const float *pd = inputDisparity + indexY * (ROAD_ESTIMATION_WIDTH_DISPARITY) + indexX;
	#pragma unroll
	for (int k = 0; k < (ROAD_ESTIMATION_HISTOGRAM_ITEMS); k++)
	{
		disparityValues[k] = static_cast<int>(fminf(fmaxf(*pd, 0), ROAD_ESTIMATION_MAX_DISPARITY - 1));
		pd += (ROAD_ESTIMATION_HISTOGRAM_THREADS);
	}
	int *pOutput = output + indexY * (ROAD_ESTIMATION_MAX_DISPARITY);
	BlockHistogram(tempStorage).Histogram(disparityValues, pOutput);
}

extern "C" __global__ void thresholdingHistogram(unsigned char* output,
		const int* __restrict__ inputVDisparity,
		float maxValue)
{
	int indexL = blockIdx.x * blockDim.x + threadIdx.x;
	if (indexL >= ((ROAD_ESTIMATION_HEIGHT_DISPARITY) * (ROAD_ESTIMATION_MAX_DISPARITY)))
	{
		return;
	}
	if (indexL % (ROAD_ESTIMATION_MAX_DISPARITY) == 0)
	{
		output[indexL] = 0;
	}
	else
	{
		output[indexL] = ((inputVDisparity[indexL] > maxValue*(ROAD_ESTIMATION_HISTOGRAM_THRESHOLD)) ? 255 : 0);
	}
}
