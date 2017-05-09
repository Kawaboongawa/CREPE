#include "edgethinning.cuh"

namespace filter
{

	__device__ __forceinline__ bool checkIdx(int y, int x, int rows, int cols)
	{
		return (y >= 0) && (y < rows) && (x >= 0) && (x < cols);
	}

	__global__ void morphological_edge_thinning_kernel(
		PtrStepSzb src,
		PtrStepb dst)
	{
		__shared__ volatile int smem[18][18];

		const int x = blockIdx.x * blockDim.x + threadIdx.x;
		const int y = blockIdx.y * blockDim.y + threadIdx.y;


		smem[threadIdx.y + 1][threadIdx.x + 1] = checkIdx(y, x, src.rows, src.cols) ? src(y, x) : 0;
		if (threadIdx.y == 0)
			smem[0][threadIdx.x + 1] = checkIdx(y - 1, x, src.rows, src.cols) ? src(y - 1, x) : 0;
		if (threadIdx.y == blockDim.y - 1)
			smem[blockDim.y + 1][threadIdx.x + 1] = checkIdx(y + 1, x, src.rows, src.cols) ? src(y + 1, x) : 0;
		if (threadIdx.x == 0)
			smem[threadIdx.y + 1][0] = checkIdx(y, x - 1, src.rows, src.cols) ? src(y, x - 1) : 0;
		if (threadIdx.x == blockDim.x - 1)
			smem[threadIdx.y + 1][blockDim.x + 1] = checkIdx(y, x + 1, src.rows, src.cols) ? src(y, x + 1) : 0;
		if (threadIdx.x == 0 && threadIdx.y == 0)
			smem[0][0] = checkIdx(y - 1, x - 1, src.rows, src.cols) ? src(y - 1, x - 1) : 0;
		if (threadIdx.x == blockDim.x - 1 && threadIdx.y == 0)
			smem[0][blockDim.x + 1] = checkIdx(y - 1, x + 1, src.rows, src.cols) ? src(y - 1, x + 1) : 0;
		if (threadIdx.x == 0 && threadIdx.y == blockDim.y - 1)
			smem[blockDim.y + 1][0] = checkIdx(y + 1, x - 1, src.rows, src.cols) ? src(y + 1, x - 1) : 0;
		if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1)
			smem[blockDim.y + 1][blockDim.x + 1] = checkIdx(y + 1, x + 1, src.rows, src.cols) ? src(y + 1, x + 1) : 0;

		__syncthreads();

		if (x >= src.cols || y >= src.rows)
			return;

		//B7
		ushort res = 0;

		if (smem[threadIdx.y + 2][threadIdx.x] == 255 && smem[threadIdx.y + 2][threadIdx.x + 1] == 255 &&
			smem[threadIdx.y + 2][threadIdx.x + 2] == 255 && smem[threadIdx.y + 1][threadIdx.x + 1] == 255 &&
			smem[threadIdx.y][threadIdx.x] == 0 && smem[threadIdx.y][threadIdx.x + 1] == 0 &&
			smem[threadIdx.y][threadIdx.x + 2] == 0)
			res = 255;

		__syncthreads();

		if (smem[threadIdx.y + 1][threadIdx.x + 1] == 255 && res == 0)
			smem[threadIdx.y + 1][threadIdx.x + 1] = 255;
		else
			smem[threadIdx.y + 1][threadIdx.x + 1] = 0;

		__syncthreads();

		//B8
		res = 0;

		if (smem[threadIdx.y + 1][threadIdx.x] == 255 && smem[threadIdx.y + 2][threadIdx.x + 1] == 255
			&& smem[threadIdx.y + 1][threadIdx.x + 1] == 255 && smem[threadIdx.y][threadIdx.x + 1] == 0
			&& smem[threadIdx.y][threadIdx.x + 2] == 0 && smem[threadIdx.y + 1][threadIdx.x + 2] == 0)
			res = 255;

		__syncthreads();

		if (smem[threadIdx.y + 1][threadIdx.x + 1] == 255 && res == 0)
			smem[threadIdx.y + 1][threadIdx.x + 1] = 255;
		else
			smem[threadIdx.y + 1][threadIdx.x + 1] = 0;

		__syncthreads();

		//B1
		res = 0;

		if (smem[threadIdx.y][threadIdx.x] == 255 && smem[threadIdx.y + 1][threadIdx.x] == 255 &&
			smem[threadIdx.y + 2][threadIdx.x] == 255 && smem[threadIdx.y + 1][threadIdx.x + 1] == 255 &&
			smem[threadIdx.y][threadIdx.x + 2] == 0 && smem[threadIdx.y + 1][threadIdx.x + 2] == 0 &&
			smem[threadIdx.y + 2][threadIdx.x + 2] == 0)
			res = 255;

		__syncthreads();

		if (smem[threadIdx.y + 1][threadIdx.x + 1] == 255 && res == 0)
			smem[threadIdx.y + 1][threadIdx.x + 1] = 255;
		else
			smem[threadIdx.y + 1][threadIdx.x + 1] = 0;

		__syncthreads();
		//B2
		res = 0;

		if (smem[threadIdx.y][threadIdx.x + 1] == 255 && smem[threadIdx.y + 1][threadIdx.x] == 255 && 
			smem[threadIdx.y + 1][threadIdx.x + 1] == 255 && smem[threadIdx.y + 1][threadIdx.x + 2] == 0 
			&& smem[threadIdx.y + 2][threadIdx.x + 1] == 0 && smem[threadIdx.y + 2][threadIdx.x + 2] == 0)
			res = 255;

		__syncthreads();

		if (smem[threadIdx.y + 1][threadIdx.x + 1] == 255 && res == 0)
			smem[threadIdx.y + 1][threadIdx.x + 1] = 255;
		else
			smem[threadIdx.y + 1][threadIdx.x + 1] = 0;

		__syncthreads();
		//B3
		res = 0;

		if (smem[threadIdx.y][threadIdx.x] == 255 && smem[threadIdx.y][threadIdx.x + 1] == 255 &&
			smem[threadIdx.y][threadIdx.x + 2] == 255 && smem[threadIdx.y + 1][threadIdx.x + 1] == 255 &&
			smem[threadIdx.y + 2][threadIdx.x] == 0 && smem[threadIdx.y + 2][threadIdx.x + 1] == 0 &&
			smem[threadIdx.y + 2][threadIdx.x + 2] == 0)
			res = 255;

		__syncthreads();

		if (smem[threadIdx.y + 1][threadIdx.x + 1] == 255 && res == 0)
			smem[threadIdx.y + 1][threadIdx.x + 1] = 255;
		else
			smem[threadIdx.y + 1][threadIdx.x + 1] = 0;

		__syncthreads();

		//B4
		res = 0;

		if (smem[threadIdx.y][threadIdx.x + 1] == 255 &&
			smem[threadIdx.y + 1][threadIdx.x + 2] == 255 && smem[threadIdx.y + 1][threadIdx.x + 1] == 255 &&
			smem[threadIdx.y + 1][threadIdx.x] == 0 && smem[threadIdx.y + 2][threadIdx.x] == 0 &&
			smem[threadIdx.y + 2][threadIdx.x + 1] == 0)
			res = 255;

		__syncthreads();

		if (smem[threadIdx.y + 1][threadIdx.x + 1] == 255 && res == 0)
			smem[threadIdx.y + 1][threadIdx.x + 1] = 255;
		else
			smem[threadIdx.y + 1][threadIdx.x + 1] = 0;

		__syncthreads();

		//B5
		res = 0;

		if (smem[threadIdx.y][threadIdx.x + 2] == 255 && smem[threadIdx.y + 1][threadIdx.x + 2] == 255 &&
			smem[threadIdx.y + 2][threadIdx.x + 2] == 255 && smem[threadIdx.y + 1][threadIdx.x + 1] == 255 &&
			smem[threadIdx.y][threadIdx.x] == 0 && smem[threadIdx.y + 1][threadIdx.x] == 0 &&
			smem[threadIdx.y + 2][threadIdx.x] == 0)
			res = 255;

		__syncthreads();

		if (smem[threadIdx.y + 1][threadIdx.x + 1] == 255 && res == 0)
			smem[threadIdx.y + 1][threadIdx.x + 1] = 255;
		else
			smem[threadIdx.y + 1][threadIdx.x + 1] = 0;

		__syncthreads();

		//B6
		res = 0;

		if (smem[threadIdx.y + 1][threadIdx.x + 2] == 255 &&
			smem[threadIdx.y + 2][threadIdx.x + 1] == 255 && smem[threadIdx.y + 1][threadIdx.x + 1] == 255 &&
			smem[threadIdx.y][threadIdx.x] == 0 && smem[threadIdx.y][threadIdx.x + 1] == 0 &&
			smem[threadIdx.y + 1][threadIdx.x] == 0)
			res = 255;

		__syncthreads();

		if (smem[threadIdx.y + 1][threadIdx.x + 1] == 255 && res == 0)
			smem[threadIdx.y + 1][threadIdx.x + 1] = 255;
		else
			smem[threadIdx.y + 1][threadIdx.x + 1] = 0;

		__syncthreads();

		dst(y, x) = smem[threadIdx.y + 1][threadIdx.x + 1];

	}

	void morphological_edge_thinning_caller(PtrStepSzb src, PtrStepb dst)
	{
		dim3 threads(16, 16);
		dim3 blocks((src.cols + threads.x - 1) / threads.x, (src.rows + threads.y - 1) / threads.y);
		morphological_edge_thinning_kernel << <blocks, threads, 0, 0 >> > (src, dst);
	}
}