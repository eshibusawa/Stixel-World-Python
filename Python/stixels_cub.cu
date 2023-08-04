// This file is part of Stixel-World-Python
// Copyleft 2023, Eijiro Shibusawa <phd_kimberlite@yahoo.co.jp>
// This file is licensed under the GPL-3.0 license.

// This file is a modified version of stixels <https://github.com/dhernandez0/stixels>,
// see GPL-3.0 license <http://www.gnu.org/licenses/>.

#include <cub/util_ptx.cuh>
#include <cub/warp/warp_scan.cuh>
#include <cub/block/block_scan.cuh>

extern "C" __global__ void computeObjectLUT(float* output,
		const float* __restrict__ inputDisparity, const float* __restrict__ inputObjectCostLUT)
{
	// STIXELS_HEIGHT_LUT = width of smoothed disparity
	// STIXELS_WIDTH_LUT = height of smoothed disparity ceiled to warp unit (32)
	// STIXELS_WIDTH_DISPARITY = height of raw disparity
	int indexY = blockIdx.x;
	// indexY = cub::ShuffleIndex<32>(indexY, 0, 0xffffffff); // may be needed
	int indexD = blockIdx.y * blockDim.y + threadIdx.y;
	// indexD = cub::ShuffleIndex<32>(indexD, 0, 0xffffffff); // may be needed
	if ((indexD >= (STIXELS_MAX_DISPARITY)) || (indexY >= (STIXELS_HEIGHT_LUT)))
	{
		return;
	}
	const int index = indexY * (STIXELS_MAX_DISPARITY) * (STIXELS_WIDTH_LUT) + indexD * (STIXELS_WIDTH_LUT);

	typedef cub::WarpScan<float> WarpScanT;
	__shared__ typename WarpScanT::TempStorage tempStorage;

	int indexXIn = threadIdx.x - 1, indexXOut = threadIdx.x;
	int disparityIn = 0;
	float cost = 0.f, aggregation_previous = 0.f, aggregation_current = 0.f;
	disparityIn = (threadIdx.x == 0) ? 0 : inputDisparity[(STIXELS_WIDTH_DISPARITY) * indexY + indexXIn];
	cost = (threadIdx.x == 0) ? 0 : inputObjectCostLUT[indexD * (STIXELS_MAX_DISPARITY) + disparityIn];
	WarpScanT(tempStorage).InclusiveSum(cost, cost, aggregation_current);
	output[index + indexXOut] = cost;
	aggregation_previous = aggregation_current;
	#pragma unroll
	for (int k = (CUDA_WARP_SIZE); k < (STIXELS_WIDTH_LUT); k += (CUDA_WARP_SIZE))
	{
		indexXIn += (CUDA_WARP_SIZE);
		indexXOut += (CUDA_WARP_SIZE);
		disparityIn = 0;
		if (indexXIn < (STIXELS_WIDTH_DISPARITY))
		{
			disparityIn = inputDisparity[(STIXELS_WIDTH_DISPARITY) * indexY + indexXIn];
		}
		float cost = inputObjectCostLUT[indexD * (STIXELS_MAX_DISPARITY) + disparityIn];
		WarpScanT(tempStorage).InclusiveSum(cost, cost, aggregation_current);
		output[index + indexXOut] = cost + aggregation_previous;
		aggregation_previous += aggregation_current;
	}
}

__device__ inline float getDataCostGround(int index, float disparity,
	const float3* __restrict__ groundModel)
{
	const float modelDiff = (disparity - groundModel[index].x);
	const float pGaussian = (groundModel[index].y) + modelDiff * modelDiff * (groundModel[index].z);
	const float pData = fminf((STIXELS_P_UNIFORM), pGaussian);
	return pData + (STIXELS_NO_P_EXISTS_GIVEN_GROUND_LOG);
}

__device__ inline float getDataCostSky(float disparity)
{
	const float pGaussian = (STIXELS_NORMALIZATION_SKY) + disparity * disparity * (STIXELS_INV_SIGMA2_SKY);
	const float pData = fminf((STIXELS_P_UNIFORM_SKY), pGaussian);
	return pData + (STIXELS_NO_P_EXISTS_GIVEN_SKY_LOG);
}

__device__ inline float computeMean(int bottom, int top, const float *sum)
{
	const float diff = sum[top + 1] - sum[bottom];
	return (diff > 0) ? (diff / (top + 1 - bottom)) : 0;
}

__device__ inline float getPriorCostObject(int bottom)
{
	return __logf((STIXELS_WIDTH_DISPARITY) - bottom);
}

__device__ inline float getPriorCostGround(float priorCost)
{
	return -(STIXELS_MATH_LOG_3E-1) + priorCost;
}

__device__ __inline__ float getPriorCostSkyFromObject(float disparityObject, float priorCost)
{
	float cost = (disparityObject < (STIXELS_PRIOR_COST_OBJECT_EPSILON)) ?
		(STIXELS_MAX_LOGPROB) : ((STIXELS_MATH_LOG_2) + priorCost);
	return cost;
}

__device__ __inline__ float getPriorCostSkyFromGround(int bottom, const float *groundModel0, float priorCost)
{
	float cost = (groundModel0[bottom] < 1) ? priorCost : (STIXELS_MAX_LOGPROB);
	return cost;
}

__device__ __inline__ float getPriorCostObjectFromGround(int bottom, float disparityObject,
		const float *groundModel0, float priorCost)
{
	float cost = priorCost - (STIXELS_MATH_LOG_7E-1);
	float disparityGround0 = groundModel0[bottom];
	if (disparityGround0 < 0.0f)
	{
		disparityGround0 = 0.0f;
	}

	if (disparityObject > (disparityGround0 + (STIXELS_PRIOR_COST_OBJECT_EPSILON)))
	{
		cost += __logf(((STIXELS_MAX_DISPARITY) - (STIXELS_PRIOR_COST_OBJECT_EPSILON) - disparityGround0) / (STIXELS_P_GRAVITY));
	}
	else if (disparityObject < (disparityGround0 - (STIXELS_PRIOR_COST_OBJECT_EPSILON)))
	{
		cost += __logf((disparityGround0 - (STIXELS_PRIOR_COST_OBJECT_EPSILON)) / (STIXELS_P_BELOW_GROUND));
	}
	else
	{
		cost += __logf((2 * (STIXELS_PRIOR_COST_OBJECT_EPSILON)) / (1 - (STIXELS_P_GRAVITY) - (STIXELS_P_BELOW_GROUND)));
	}
	return cost;
}

__device__ __inline__ float getPriorCostObjectFromObject(int bottom, float disparityObject,
		float disparityObject0, const float *objectDisparityRange,
		int vHorizon, const float priorCost)
{
	float cost = (bottom < vHorizon) ? -(STIXELS_MATH_LOG_7E-1) : (STIXELS_MATH_LOG_2);
	cost += priorCost;

	float disparityDiff = objectDisparityRange[static_cast<int>(disparityObject0)];
	if(disparityDiff < 0.0f) {
		disparityDiff = 0.0f;
	}

	if(disparityObject > (disparityObject0 + disparityDiff))
	{
		cost += __logf(((STIXELS_MAX_DISPARITY) - disparityObject0 - disparityDiff) / (STIXELS_P_ORDER));
	} else if(disparityObject < (disparityObject0 - disparityDiff))
	{
		cost += __logf((disparityObject0 - disparityDiff) / (1 - (STIXELS_P_ORDER)));
	}
	else
	{
		cost = (STIXELS_MAX_LOGPROB);
	}
	return cost;
}

__device__ __inline__ float getPriorCostObjectFromSky(float disparityObject, float priorCost)
{
	float cost = (disparityObject > (STIXELS_PRIOR_COST_OBJECT_EPSILON)) ?
		(__logf((STIXELS_MAX_DISPARITY) - (STIXELS_PRIOR_COST_OBJECT_EPSILON)) + priorCost)
		: (STIXELS_MAX_LOGPROB);
	return cost;
}

extern "C" __global__ void computeStixels(short3 *outputType, float *outputDisparity,
		const float* __restrict__ inputDisparity,
		const float3* __restrict__ intputGroundModel,
		const float* __restrict__ intputObjectLUT,
		const float* __restrict__ objectDisparityRange,
		int vHorizontal)
{
	enum {Ground = 0, Object, Sky};
	int indexY = blockIdx.x;
	int indexX = threadIdx.x;
	if ((indexY >= (STIXELS_HEIGHT_LUT)))
	{
		return;
	}
	typedef cub::BlockScan<float, (STIXELS_THREAD_BLOCKS), cub::BLOCK_SCAN_RAKING> BlockScanT;
	__shared__ typename BlockScanT::TempStorage tempStorage;
	__shared__ float disparitySum[(STIXELS_WIDTH_LUT)]; // for block cooperative opration
	__shared__ float skyCostSum[(STIXELS_WIDTH_LUT)]; // for block cooperative opration
	__shared__ float groundCostSum[(STIXELS_WIDTH_LUT)]; // for block cooperative opration
	__shared__ float groundModel0[(STIXELS_WIDTH_DISPARITY)];
	__shared__ float costTable[(STIXELS_WIDTH_DISPARITY)][3];
	__shared__ short indexTable[(STIXELS_WIDTH_DISPARITY)][3];

	float disparityIn = (indexX < STIXELS_WIDTH_DISPARITY) ? inputDisparity[(STIXELS_WIDTH_DISPARITY) * indexY + indexX] : 0;
	float skyCost = ((indexX < vHorizontal) ? (STIXELS_MAX_LOGPROB) : getDataCostSky(disparityIn));
	float groundCost = ((indexX >= vHorizontal) ? (STIXELS_MAX_LOGPROB) :
		getDataCostGround(indexX, disparityIn, intputGroundModel));

	BlockScanT(tempStorage).ExclusiveSum(disparityIn, disparitySum[indexX]);
	BlockScanT(tempStorage).ExclusiveSum(skyCost, skyCostSum[indexX]);
	BlockScanT(tempStorage).ExclusiveSum(groundCost, groundCostSum[indexX]);
	__syncthreads();

	// block cooperative opration is completed, so redundant threads are exited
	if (indexX >= (STIXELS_WIDTH_DISPARITY))
	{
		return;
	}
	groundModel0[indexX] = intputGroundModel[indexX].x;
	costTable[indexX][Ground] = costTable[indexX][Object] = costTable[indexX][Sky] = (STIXELS_MAX_LOGPROB);
	indexTable[indexX][Ground] = indexTable[indexX][Object] = indexTable[indexX][Sky] = -1;
	__syncthreads();

	const int vT = indexX;
	const float *objectLUTY = intputObjectLUT + indexY * (STIXELS_MAX_DISPARITY) * (STIXELS_WIDTH_LUT);
	{
		const int vB = 0;
		float disparityObject = computeMean(vB, vT, disparitySum);
		const int disparityObjectIndex = static_cast<int>(floorf(disparityObject));

		const float dataCostGround = groundCostSum[vT + 1] - groundCostSum[vB];
		// objectLUT[indexY][disparityObject][vT + 1] - objectLUT[indexY][disparityObject][vB]
		const float dataCostObject = objectLUTY[disparityObjectIndex*(STIXELS_WIDTH_LUT) + vT + 1] -
				objectLUTY[disparityObjectIndex*(STIXELS_WIDTH_LUT) + vB];

		float priorCostObject = 0.f;
		if (vT <= vHorizontal) // is current pixel vT below the horizon?
		{
			const float priorCostGround = (STIXELS_PRIOR_COST_GROUND_BOTTOM);
			const float currCostGround = costTable[vT][Ground];
			const float costGround = dataCostGround + priorCostGround;
			if(costGround < currCostGround)
			{
				costTable[vT][Ground] = costGround;
				indexTable[vT][Ground] = Ground;
			}
			priorCostObject = (STIXELS_PRIOR_COST_OBJECT_BOTTOM0);
		}
		else
		{
			priorCostObject = (STIXELS_PRIOR_COST_OBJECT_BOTTOM1);
		}

		const float currCostObject = costTable[vT][Object];
		const float costObject = dataCostObject + priorCostObject;
		if(costObject < currCostObject)
		{
			costTable[vT][Object] = costObject;
			indexTable[vT][Object] = Object;
		}
	}
	__syncthreads();

	#pragma unroll
	for(int vB = 1; vB < (STIXELS_WIDTH_DISPARITY); vB++) {
		__syncthreads();
		if(vT < vB)
		{
			continue;
		}

		const float disparityObject = computeMean(vB, vT, disparitySum);
		const int disparityObjectIndex = static_cast<int>(floorf(disparityObject));
		const float dataCostObject = objectLUTY[disparityObjectIndex*(STIXELS_WIDTH_LUT) + vT + 1] -
				objectLUTY[disparityObjectIndex*(STIXELS_WIDTH_LUT) + vB];
		const float priorCostObject = getPriorCostObject(vB);
		const int vT0 = vB - 1;
		const int vB0 = indexTable[vT0][Object] / 3;
		const float disparityObject0 = computeMean(vB0, vT0, disparitySum);
		if (vT0 < vHorizontal)
		{
			const float dataCostGround = groundCostSum[vT + 1] - groundCostSum[vB];
			const float priorCostGrond0 = getPriorCostGround(priorCostObject);
			const float priorCostGround1 = priorCostGrond0 + costTable[vT0][Ground];
			const float priorCostGround2 = priorCostGrond0 + costTable[vT0][Object];
			const float currCostGround = costTable[vT][Ground];
			const float costGround = dataCostGround + fminf(priorCostGround1, priorCostGround2);
			if (costGround < currCostGround)
			{
				costTable[vT][Ground] = costGround;
				const int min0 = (priorCostGround1 < priorCostGround2) ? Ground : Object;
				indexTable[vT][Ground] = vB * 3 + min0;
			}
		}
		else
		{
			const float dataCostSky = skyCostSum[vT + 1] - skyCostSum[vB];
			const float priorCostSky1 = getPriorCostSkyFromGround(vT0, groundModel0, priorCostObject)
					+ costTable[vT0][Ground];
			const float priorCostSky2 = getPriorCostSkyFromObject(disparityObject0, priorCostObject)
							+ costTable[vT0][Object];
			const float currCostSky = costTable[vT][Sky];
			const float costSky = dataCostSky + fminf(priorCostSky1, priorCostSky2);
			if(costSky < currCostSky)
			{
				costTable[vT][Sky] = costSky;
				const int min0 = (priorCostSky1 < priorCostSky2) ? Ground : Object;
				indexTable[vT][Sky] = vB * 3 + min0;
			}
		}

		const float priorCostObject1 = getPriorCostObjectFromGround(vT0, disparityObject,
			groundModel0, priorCostObject) + costTable[vT0][Ground];
		const float priorCostObject2 = getPriorCostObjectFromObject(vT0, disparityObject,
				disparityObject0, objectDisparityRange, vHorizontal, priorCostObject) + costTable[vT0][Object];
		const float priorCostObject3 = getPriorCostObjectFromSky(disparityObject, priorCostObject) + costTable[vT0][Sky];
		const float currCostObject = costTable[vT][Object];
		const float costObject = dataCostObject + fminf(fminf(priorCostObject1, priorCostObject2), priorCostObject3);
		if (costObject < currCostObject)
		{
			costTable[vT][Object] = costObject;
			int min0 = Object;
			if (priorCostObject1 < priorCostObject2)
			{
				min0 = Ground;
			}
			if (priorCostObject3 < fminf(priorCostObject1, priorCostObject2))
			{
				min0 = Sky;
			}
			indexTable[vT][Object] = vB * 3 + min0;
		}
	}
	__syncthreads();

	if (threadIdx.x == 0)
	{
		int vT = (STIXELS_WIDTH_DISPARITY) - 1;
		const float costGround = costTable[vT][Ground];
		const float costObject = costTable[vT][Object];
		const float costSky = costTable[vT][Sky];
		int type = Ground;
		if (costObject < costGround)
		{
			type = Object;
		}
		if (costSky < fminf(costGround, costObject)) {
			type = Sky;
		}

		int vT0 = vT, minVT = vT, minType = type;
		int k = 0;
		float disparity = 0.f;
		do
		{
			vT0 = (indexTable[minVT][minType] / 3) - 1;
			disparity = computeMean(vT0 + 1, vT, disparitySum);
			if ((type == Object) && (disparity < (STIXELS_MINIMUM_OBJECT_DISPARITY)))
			{
				type = Sky;
			}
			outputType[indexY * (STIXELS_MAX_SECTIONS) + k].x = vT0 + 1;
			outputType[indexY * (STIXELS_MAX_SECTIONS) + k].y = vT;
			outputType[indexY * (STIXELS_MAX_SECTIONS) + k].z = type;
			outputDisparity[indexY * (STIXELS_MAX_SECTIONS) + k] = disparity;

			type = indexTable[minVT][minType] % 3;
			vT = vT0;
			minVT = vT0;
			minType = type;
			k++;
		}
		while((vT0 != -1) && (k < (STIXELS_MAX_SECTIONS) - 1));
		outputType[indexY * (STIXELS_MAX_SECTIONS) + k].z = -1;
	}
}
