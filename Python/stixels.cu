// This file is part of Stixel-World-Python
// Copyleft 2023, Eijiro Shibusawa <phd_kimberlite@yahoo.co.jp>
// This file is licensed under the GPL-3.0 license.

// This file is a modified version of stixels <https://github.com/dhernandez0/stixels>,
// see GPL-3.0 license <http://www.gnu.org/licenses/>.

inline __device__ float groundFunction(int v, int vHorizontal, float alpha)
{
	return alpha * (vHorizontal - v);
}

extern "C" __global__ void updateGroundModel(float3 *output,
	int vHorizontal, float groundAlpha,
	float cameraHeight,	float cameraTilt,
	int height)
{
	const int indexY = blockIdx.x * blockDim.x + threadIdx.x;
	if (indexY >= height)
	{
		return;
	}
	const int &v = indexY;
	const float gF = groundFunction(v, vHorizontal, groundAlpha);
	const float fbH = (STIXELS_FOCAL_LENGTH) * (STIXELS_BASELINE) / cameraHeight;
	const float x = fmaf(1/(STIXELS_FOCAL_LENGTH), (vHorizontal-v), cameraTilt);
	const float s2R = fbH * fbH * ((STIXELS_SIGMA_CAMERA_HEIGHT) * (STIXELS_SIGMA_CAMERA_HEIGHT) * x * x / (cameraHeight*cameraHeight ) + (STIXELS_SIGMA_CAMERA_TILT)*(STIXELS_SIGMA_CAMERA_TILT));
	const float s = sqrtf((STIXELS_SIGMA_DISPARITY_GROUND)*(STIXELS_SIGMA_DISPARITY_GROUND) + s2R);
	const float aRange = 0.5f*(erf(((STIXELS_MAX_DISPARITY) - gF)/(s * (STIXELS_MATH_SQRT_2))) - erf((-gF) / (s * (STIXELS_MATH_SQRT_2))));
	output[indexY].x = gF;
	output[indexY].y = __logf(aRange * (s * (STIXELS_MATH_SQRT_2PI)) / (1.0f - (STIXELS_P_OUT)));
	output[indexY].z = 1 / (2 * s * s);
}

template<typename TYPE, int WIDTH>
struct Mean
{
	__device__ static inline TYPE get(const TYPE *raw)
	{
		TYPE mean = 0;
		#pragma unroll
		for (int k = 0; k < WIDTH; k++)
		{
			mean += raw[k];
		}
		return mean / WIDTH;
	}
};

template<typename TYPE, int WIDTH>
struct Median
{
	__device__ static inline TYPE get(const TYPE *raw)
	{
		TYPE sorted[WIDTH], tmp;
		static const int MEDIAN_POSITION = WIDTH / 2;
		#pragma unroll
		for (int k = 0; k < WIDTH; k++)
		{
			sorted[k] = raw[k];
		}
		#pragma unroll
		for (int j = 0; j < (MEDIAN_POSITION + 1); j++)
		{
			for (int i = j + 1; i < WIDTH; i++)
			{
				if (sorted[j] > sorted[i])
				{
					tmp = sorted[j];
					sorted[j] = sorted[i];
					sorted[i] = tmp;
				}
			}
		}
		TYPE median;
		if ((STIXELS_STEP_SIZE) % 2)
		{
			median = sorted[MEDIAN_POSITION];
		}
		else
		{
			median = (sorted[MEDIAN_POSITION - 1] + sorted[MEDIAN_POSITION]) / 2;
		}
		return median;
	}
};

template<typename Smoothing>
__device__ void horizontalSmoothingAndTranspose_(float *output,
	const float * __restrict__ inputRaw,
	int heightRaw, int widthRaw, int width)
{
	// raw(heightRaw, widthRaw) -> output(height, heightRaw)
	const int indexL = blockIdx.x * blockDim.x + threadIdx.x;
	const int length = heightRaw * width;
	if (indexL >= length)
	{
		return;
	}
	const int indexX = indexL % width;
	const int indexY = indexL / width;
	const int index = (indexX + 1) * heightRaw - indexY - 1; // transposed
	const int indexRaw = widthRaw * indexY + indexX * (STIXELS_STEP_SIZE) + (STIXELS_WIDTH_MARGIN);
	output[index] = Smoothing::get(inputRaw + indexRaw);
}

extern "C" __global__ void horizontalMeanAndTranspose(float *output,
	const float * __restrict__ inputRaw,
	int heightRaw, int widthRaw, int width)
{
	horizontalSmoothingAndTranspose_<Mean<float, (STIXELS_STEP_SIZE)> >(output, inputRaw, heightRaw, widthRaw, width);
}

extern "C" __global__ void horizontalMedianAndTranspose(float *output,
	const float * __restrict__ inputRaw,
	int heightRaw, int widthRaw, int width)
{
	horizontalSmoothingAndTranspose_<Median<float, (STIXELS_STEP_SIZE)> >(output, inputRaw, heightRaw, widthRaw, width);
}

extern "C" __global__ void computeObjectDisparityRange(float *output)
{
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= (STIXELS_MAX_DISPARITY))
	{
		return;
	}

	if (index == 0)
	{
		output[index] = 0.f;
		return;
	}

	const float mean = static_cast<float>(index);
	const float fB = (STIXELS_FOCAL_LENGTH) * (STIXELS_BASELINE);
	const float pMeanPlusZ =  fmaf(1 / mean, fB, (STIXELS_RANGE_OBJECTS_Z));
	output[index] = fmaf(1 / pMeanPlusZ, -fB, mean);
}

extern "C" __global__ void computeObjectModel(float2 *output)
{
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= (STIXELS_MAX_DISPARITY))
	{
		return;
	}
	const float disparity = static_cast<float>(index);
	const float invFB = 1.f/((STIXELS_FOCAL_LENGTH) * (STIXELS_BASELINE));
	const float sigmaObject = disparity * disparity * (STIXELS_RANGE_OBJECTS_Z) * invFB;
	const float s = sqrtf(fmaf((STIXELS_SIGMA_DISPARITY_OBJECT), (STIXELS_SIGMA_DISPARITY_OBJECT), sigmaObject * sigmaObject));
	const float aRange = 0.5f*(erf(((STIXELS_MAX_DISPARITY) - disparity)/(s * (STIXELS_MATH_SQRT_2))) - erf(-disparity/(s * (STIXELS_MATH_SQRT_2))));
	output[index] = make_float2(
		__logf(aRange * (s * (STIXELS_MATH_SQRT_2PI)) / (1.0f - (STIXELS_P_OUT))),
		1/(2 * s * s)
	);
}

extern "C" __global__ void computeObjectCostLUT(float *output,
	const float2 * __restrict__ input)
{
	const int indexX = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexY = blockIdx.y * blockDim.y + threadIdx.y;
	if ((indexX >= (STIXELS_MAX_DISPARITY)) || (indexY >= (STIXELS_MAX_DISPARITY)))
	{
		return;
	}
	const int index = indexY * (STIXELS_MAX_DISPARITY) + indexX;

	const float diff = static_cast<float>(indexX) - indexY;
	const float pGaussian =  fmaf(diff * diff, input[indexX].y, input[indexX].x);
	const float pData = fminf((STIXELS_P_UNIFORM), pGaussian);
	float dataCost = pData + (STIXELS_NO_P_N_EXISTS_OBJECT_LOG);
	output[index] = dataCost;
}
