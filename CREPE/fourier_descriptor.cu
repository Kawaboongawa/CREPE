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

	compute_magnitude_kernel << <blocks, threads, 0, 0 >> > (src, dst, size);

	float centroid = 0.f;
	cudaMemcpy(&centroid, dst, sizeof(float), cudaMemcpyDeviceToHost);
	compute_centroid_signature_kernel << <blocks, threads, 0, 0 >> >(dst + 1, dst + 1, centroid, size - 1);

}

__device__ float _result = 0;

__global__ void compare_descriptors_kernel(
	float2* src1,
	float2* src2,
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

#pragma unroll
	if (threadIdx.x == 0)
	{
		float res = 0;
		for (int i = 0; i < 128; i++)
			res += smem[i];
		atomicAdd(&_result, sqrt(res));
	}

}

float compare_descriptors_caller(
	float2* src1,
	float2* src2,
	uint    size)
{
	void* result_ptr;
	cudaGetSymbolAddress(&result_ptr, _result);

	const dim3 block(128);
	const dim3 grid(std::min(size, 65535u), ((size + 65534u) / 65535u), 1);

	compare_descriptors_kernel << <grid, block, 0, 0 >> > (src1 + 1, src2 + 1, size);

	float count;
	cudaMemcpyAsync(&count, result_ptr, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemsetAsync(result_ptr, 0, sizeof(float));

	//This is a useful example that shows how to debug CUDA without Nsight
	//very tedious yet better than nothing...

	/*
	float2* a = (float2*)malloc(300 * sizeof(float2));
	cudaMemcpy(a, src1, 100 * sizeof(float2), cudaMemcpyDeviceToHost);
	float2 x0 = a[0];
	float2 x1 = a[1];
	float2 x2 = a[2];
	float2 x3 = a[3];
	float2 x4 = a[4];
	float2 x5 = a[5];

	float2* b = (float2*)malloc(300 * sizeof(float2));
	cudaMemcpy(b, src2, 100 * sizeof(float2), cudaMemcpyDeviceToHost);
	float2 y0 = b[0];
	float2 y1 = b[1];
	float2 y2 = b[2];
	float2 y3 = b[3];
	float2 y4 = b[4];
	float2 y5 = b[5];

	free(a);
	free(b);
	*/
	return count;
}
