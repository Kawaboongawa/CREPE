#include "fourier_descriptor.cuh"

__global__ void centroid_distance_kernel(
	ushort2* src,
	float2* dst,
	uint size,
	float2 centroid)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	while (index < size)
	{
		float dist_x = static_cast<float>(src[index].x) - centroid.x;
		float dist_y = static_cast<float>(src[index].y) - centroid.y;
		dst[index] = make_float2(sqrt(dist_x * dist_x + dist_y * dist_y), 0);
		index += blockDim.x * gridDim.x;
	}
}


void centroid_distance_caller(
	ushort2* src,
	float2* dst,
	uint size,
	float2 centroid)

{
	uint threads = get_max_threads();
	uint blocks = map_blocks_to_problem(size, threads);
	centroid_distance_kernel << <blocks, threads, 0, 0 >> > (src, dst, size, centroid);
}

__global__ void compute_descriptors_kernel(
	float2*	    src,
	float2*	    dst,
	uint        size)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	while (index < size)
	{
		float modulus = sqrt(src[index].x * src[index].x + src[index].x * src[index].x);
		dst[index].x = sqrt(modulus * modulus + src[0].x * src[0].x);
		index += blockDim.x * gridDim.x;
	}
}

void compute_descriptors_caller(
		float2*	    src,
		float2*	    dst,
		uint        size,
		const cufftHandle	plan1D)
	{
		uint threads = get_max_threads();
		uint blocks = map_blocks_to_problem(size, threads);
		cufftExecC2C(plan1D, src, dst, CUFFT_FORWARD);
		compute_descriptors_kernel << <blocks, threads, 0, 0 >> > (src, dst, size);

	}
