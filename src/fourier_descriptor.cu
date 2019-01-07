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

__global__ void compute_magnitude_kernel(
	float2*	    src,
	float2*	    dst,
	uint        size)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= size)
		return;


	float modulus = sqrt(src[index].x * src[index].x + src[index].y * src[index].y);
	dst[index].x = modulus;
	dst[index].y = 0.0f;

}

__global__ void compute_centroid_signature_kernel(
	float2*	    src,
	float2*	    dst,
	float       centroid,
	uint        size)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= size)
		return;
	dst[index].x = src[index].x / centroid;

}

void compute_centroid_signature_caller(
	float2*	    src,
	float2*	    dst,
	uint        size,
	const cufftHandle	plan1D)
{
	uint threads = get_max_threads();
	uint blocks = map_blocks_to_problem(size, threads);

	cufftExecC2C(plan1D, src, dst, CUFFT_FORWARD);

	compute_magnitude_kernel << <blocks, threads, 0, 0 >> > (dst, dst, size);

	float centroid = 0.f;
	cudaMemcpy(&centroid, dst, sizeof(float), cudaMemcpyDeviceToHost);
	compute_centroid_signature_kernel << <blocks, threads, 0, 0 >> >(dst + 1, dst + 1, centroid, size - 1);
}

__global__ void compare_descriptors_kernel(
	float2* src1,
	float2* src2,
    float*  dst,
	uint    size)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ volatile float smem[128];
	if (index >= size)
		return;

	float2 val1 = src1[index];
	float2 val2 = src2[index];
	float mod = abs(val1.x - val2.x);
	smem[threadIdx.x] = mod * mod;

	__syncthreads();

	if (threadIdx.x == 0)
	{
		float res = 0.f;
		for (int i = 0; i < MIN(size, 128); i++)
			res += smem[i];
		res = sqrt(res);
		src1[0].y = res;
		atomicAdd(dst, res);
	}

}

float compare_descriptors_caller(
	float2* src1,
	float2* src2,
	uint    size)
{
	


	const dim3 block(128);
	const dim3 grid((size + 127) / 128);

	float* dst;
	cudaMalloc(&dst, sizeof(float));
	cudaMemset(dst, 0, sizeof(float));
	compare_descriptors_kernel << <grid, block, 0, 0 >> > (src1, src2, dst, size);

	float res;
	cudaMemcpy(&res, dst, sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(dst);
	return res;
}
