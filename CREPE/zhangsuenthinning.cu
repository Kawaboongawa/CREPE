#include "edgethinning.cuh"

namespace filter
{

	__device__ __forceinline__ bool checkIdx(int y, int x, int rows, int cols)
	{
		return (y >= 0) && (y < rows) && (x >= 0) && (x < cols);
	}

	__device__ __forceinline__ ushort checkTransition(ushort a, ushort b)
	{
		return (a == 0 && b == 255) ? 1 : 0;
	}

	__constant__ int c_dx[8] = { 0, 1, 1, 1, 0, -1, -1, -1 };
	__constant__ int c_dy[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };

	__global__ void  zhang_suen_edge_thinning_kernel(
		PtrStepSzb src,
		PtrStepb dst,
		int step)
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

		bool res = true;
		ushort n = 0;
		ushort n_transi = 0;

		for (int j = -1; j < 2; j++)
		{
			for (int i = -1; i < 2; i++)
			{
				if (i == 0 && j == 0)
					continue;
				if (smem[threadIdx.y + 1 + j][threadIdx.x + 1 + i] == 255)
					n++;
			}
		}

		if (n < 2 || n > 6)
			res = false;

		for (int i = 0; i < 8; i++)
			n_transi += checkTransition(smem[threadIdx.y + 1 + c_dy[i]][threadIdx.x + 1 + c_dx[i]], smem[threadIdx.y + 1 + c_dy[(i + 1) % 8]][threadIdx.x + 1 + c_dx[(i + 1) % 8]]);

		if (n_transi != 1)
			res = false;
		if (step == 0)
		{
			if (smem[threadIdx.y][threadIdx.x + 1] * smem[threadIdx.y + 1][threadIdx.x + 2] * smem[threadIdx.y + 2][threadIdx.x + 1] != 0 ||
				smem[threadIdx.y + 1][threadIdx.x + 2] * smem[threadIdx.y + 2][threadIdx.x + 1] * smem[threadIdx.y + 1][threadIdx.x] != 0)
				res = false;
		}
		else
		{
			if (smem[threadIdx.y][threadIdx.x + 1] * smem[threadIdx.y + 1][threadIdx.x + 2] * smem[threadIdx.y + 1][threadIdx.x] != 0 ||
				smem[threadIdx.y][threadIdx.x + 1] * smem[threadIdx.y + 2][threadIdx.x + 1] * smem[threadIdx.y + 1][threadIdx.x] != 0)
				res = false;
		}


		__syncthreads();

		if (res == true && smem[threadIdx.y + 1][threadIdx.x + 1] == 255)
			smem[threadIdx.y + 1][threadIdx.x + 1] = 0;

		dst(y, x) = smem[threadIdx.y + 1][threadIdx.x + 1];

	}

	void  zhang_suen_edge_thinning_caller(PtrStepSzb src, PtrStepb dst)
	{
		dim3 threads(16, 16);
		dim3 blocks((src.cols + threads.x - 1) / threads.x, (src.rows + threads.y - 1) / threads.y);
		zhang_suen_edge_thinning_kernel << <blocks, threads, 0, 0 >> > (src, src, 0);
		zhang_suen_edge_thinning_kernel << <blocks, threads, 0, 0 >> > (src, dst, 1);
	}
}