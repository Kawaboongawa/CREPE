#include "filters.cuh"


__global__ void set_cyan_kernel(
	uchar3* frame,
	const uint frame_resolution)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	while (index < frame_resolution)
	{
		frame[index].x = 0;
		frame[index].y = 100;
		frame[index].z = 130;
		index += blockDim.x * gridDim.x;
	}
}

__global__ void swap_rb_kernel(const cv::cuda::PtrStepSz<uchar3> src, cv::cuda::PtrStep<uchar3> dst)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x < src.cols && y < src.rows)
	{
		uchar3 v = src(y, x);
		dst(y, x) = make_uchar3(v.z, v.y, v.x);
	}
}

__global__ void remove_b_kernel(cv::cuda::PtrStepSz<uchar3> src)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x < src.cols && y < src.rows)
	{
		uchar3 v = src(y, x);
		//NB: In opencv channel order is BGR
		src(y, x) = make_uchar3(0, v.y, v.z);
	}
}

void swap_rb_caller(const cv::cuda::PtrStepSz<uchar3>& src, cv::cuda::PtrStep<uchar3> dst)
{
	unsigned int threads_2d = get_max_threads_2d();
	dim3 threads(threads_2d, threads_2d);
	dim3 blocks((src.cols + threads_2d - 1) / threads_2d, (src.rows + threads_2d - 1) / threads_2d); 
	swap_rb_kernel <<<blocks, threads, 0, 0>>> (src, dst);
}

void remove_b_caller(cv::cuda::PtrStepSz<uchar3> src)
{
	unsigned int threads_2d = get_max_threads_2d();
	dim3 threads(threads_2d, threads_2d);
	dim3 blocks((src.cols + threads_2d - 1) / threads_2d, (src.rows + threads_2d - 1) / threads_2d);
	remove_b_kernel <<<blocks, threads, 0, 0>>> (src);
}


void set_cyan_caller(
	void *	    frame,
	const uint  frame_resolution)
{
	uint threads = get_max_threads();
	uint blocks = map_blocks_to_problem(frame_resolution, threads);
	set_cyan_kernel << <blocks, threads, 0, 0 >> > (static_cast<uchar3 *>(frame), frame_resolution);
}