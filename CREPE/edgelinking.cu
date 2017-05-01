#include "edgelinking.cuh"
/*
**We implement here one pixel gap filling as explained in http://ceng.anadolu.edu.tr/cv/PEL/.
*/

namespace filter
{

	__constant__ int c_dx[8] = { -1,  0,  1, -1, 1, -1, 0, 1 };
	__constant__ int c_dy[8] = { -1, -1, -1,  0, 0,  1, 1, 1 };
	__device__ int counter = 0;

	__device__ __forceinline__ bool checkIdx(int y, int x, int rows, int cols)
	{
		return (y >= 0) && (y < rows) && (x >= 0) && (x < cols);
	}

	__global__ void FillGaps_kernel(
		PtrStepSzb src,
		PtrStepb dst)
	{
		__shared__ volatile int smem[20][20];
		const int x = blockIdx.x * blockDim.x + threadIdx.x;
		const int y = blockIdx.y * blockDim.y + threadIdx.y;

		smem[threadIdx.y + 2][threadIdx.x + 2] = checkIdx(y, x, src.rows, src.cols) ? src(y, x) : 0;
		if (threadIdx.y == 0)
		{
			smem[1][threadIdx.x + 2] = checkIdx(y - 1, x, src.rows, src.cols) ? src(y - 1, x) : 0;
			smem[0][threadIdx.x + 2] = checkIdx(y - 2, x, src.rows, src.cols) ? src(y - 2, x) : 0;
		}
		if (threadIdx.y == blockDim.y - 1)
		{
			smem[blockDim.y + 1][threadIdx.x + 2] = checkIdx(y + 1, x, src.rows, src.cols) ? src(y + 1, x) : 0;
			smem[blockDim.y + 2][threadIdx.x + 2] = checkIdx(y + 2, x, src.rows, src.cols) ? src(y + 2, x) : 0;
		}
		if (threadIdx.x == 0)
		{
			smem[threadIdx.y + 2][1] = checkIdx(y, x - 1, src.rows, src.cols) ? src(y, x - 1) : 0;
			smem[threadIdx.y + 2][0] = checkIdx(y, x - 2, src.rows, src.cols) ? src(y, x - 2) : 0;
		}
		if (threadIdx.x == blockDim.x - 1)
		{
			smem[threadIdx.y + 2][blockDim.x + 1] = checkIdx(y, x + 1, src.rows, src.cols) ? src(y, x + 1) : 0;
			smem[threadIdx.y + 2][blockDim.x + 2] = checkIdx(y, x + 2, src.rows, src.cols) ? src(y, x + 2) : 0;
		}
		if (threadIdx.x == 0 && threadIdx.y == 0)
		{
			smem[0][0] = checkIdx(y - 2, x - 2, src.rows, src.cols) ? src(y - 2, x - 2) : 0;
			smem[1][1] = checkIdx(y - 1, x - 1, src.rows, src.cols) ? src(y - 1, x - 1) : 0;
			smem[0][1] = checkIdx(y - 2, x - 1, src.rows, src.cols) ? src(y - 2, x - 1) : 0;
			smem[1][0] = checkIdx(y - 1, x - 2, src.rows, src.cols) ? src(y - 1, x - 2) : 0;
		}
		if (threadIdx.x == blockDim.x - 1 && threadIdx.y == 0)
		{
			smem[1][blockDim.x + 1] = checkIdx(y - 1, x + 1, src.rows, src.cols) ? src(y - 1, x + 1) : 0;
			smem[0][blockDim.x + 2] = checkIdx(y - 2, x + 2, src.rows, src.cols) ? src(y - 2, x + 2) : 0;
			smem[0][blockDim.x + 1] = checkIdx(y - 2, x + 1, src.rows, src.cols) ? src(y - 2, x + 1) : 0;
			smem[1][blockDim.x + 2] = checkIdx(y - 1, x + 2, src.rows, src.cols) ? src(y - 1, x + 2) : 0;
		}
		if (threadIdx.x == 0 && threadIdx.y == blockDim.y - 1)
		{
			smem[blockDim.y + 1][1] = checkIdx(y + 1, x - 1, src.rows, src.cols) ? src(y + 1, x - 1) : 0;
			smem[blockDim.y + 2][0] = checkIdx(y + 2, x - 2, src.rows, src.cols) ? src(y + 2, x - 2) : 0;
			smem[blockDim.y + 1][0] = checkIdx(y + 1, x - 2, src.rows, src.cols) ? src(y + 1, x - 2) : 0;
			smem[blockDim.y + 2][1] = checkIdx(y + 2, x - 1, src.rows, src.cols) ? src(y + 2, x - 1) : 0;
		}
		if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1)
		{
			smem[blockDim.y + 1][blockDim.x + 1] = checkIdx(y + 1, x + 1, src.rows, src.cols) ? src(y + 1, x + 1) : 0;
			smem[blockDim.y + 2][blockDim.x + 2] = checkIdx(y + 2, x + 2, src.rows, src.cols) ? src(y + 2, x + 2) : 0;
			smem[blockDim.y + 2][blockDim.x + 1] = checkIdx(y + 2, x + 1, src.rows, src.cols) ? src(y + 2, x + 1) : 0;
			smem[blockDim.y + 1][blockDim.x + 2] = checkIdx(y + 1, x + 2, src.rows, src.cols) ? src(y + 1, x + 2) : 0;
		}

		__syncthreads();

		if (x >= src.cols || y >= src.rows)
			return;

		if (smem[threadIdx.y + 2][threadIdx.x + 2] != 255)
			return;

		dst(y, x) = 255;
		int n = 0;
#pragma unroll
		for (int j = -1; j <= 1; j++)
		{
			for (int i = -1; i <= 1; i++)
			{
				if (i == 0 && j == 0)
					continue;
				if (smem[threadIdx.y + j + 2][threadIdx.x + i + 2] == 255)
					n++;
			}
		}
		__syncthreads();

		if (n >= 2)
			return;

		/*
		** Please notice that we always add +2 to match the mapping of smem where everything
		** as an offset of [+2,+2].
		*/

#pragma unroll
		for (int j = -1; j <= 1; j++)
		{
			for (int i = -1; i <= 1; i++)
			{
				if (i == 0 && j == 0)
					continue;
				if (smem[threadIdx.y + j + 2][threadIdx.x + i + 2] == 255)
				{
					if (smem[threadIdx.y + j * (-2) + 2][threadIdx.x + i * (-2) + 2] == 255)
					{
						dst(y + j * (-1), x + i * (-1)) = 255;
					} 
					//left/right cases
					else if (j == 0)
					{
						if (smem[threadIdx.y][threadIdx.x + 2 + i * (-1)] == 255
							|| smem[threadIdx.y][threadIdx.x + 2 + i * (-2)] == 255
							|| smem[threadIdx.y + 1][threadIdx.x + 2 + i * (-2)] == 255)
							dst(y - 1, x + i * (-1)) = 255;
						else if (smem[threadIdx.y + 4][threadIdx.x + 2 + i * (-1)] == 255
							|| smem[threadIdx.y + 4][threadIdx.x + 2 + i * (-2)] == 255
							|| smem[threadIdx.y + 3][threadIdx.x + 2 + i * (-2)] == 255)
							dst(y + 1, x + i * (-1)) = 255;

					}
					//up/down cases
					else if (i == 0)
					{
						if (smem[threadIdx.y + 2 + j * (-2)][threadIdx.x + 3] == 255
							|| smem[threadIdx.y + 2 + j * (-2)][threadIdx.x + 3] == 255
							|| smem[threadIdx.y + 2 + j * (-1)][threadIdx.x + 4] == 255)
							dst(y + j * (-1), x + 1) = 255;
						else if (smem[threadIdx.y + 2 + j * (-2)][threadIdx.x] == 255
							|| smem[threadIdx.y + 2 + j * (-2)][threadIdx.x + 1] == 255
							|| smem[threadIdx.y + 2 + j * (-1)][threadIdx.x] == 255)
							dst(y + j * (-1), x - 1) = 255;
					}
					//diagonals cases
					else
					{
						if (smem[threadIdx.y + 2 + j * (-2)][threadIdx.x + 2 + i * (-1)] == 255
							|| smem[threadIdx.y + 2 + j * (-1)][threadIdx.x + 2 + i * (-2)] == 255)
							dst(y + j * (-1), x + i * (-1)) = 255;
						else if (smem[threadIdx.y + 2 + j * (-2)][threadIdx.x + 2] == 255)
							dst(y + j * (-1), x) = 255;
						else if (smem[threadIdx.y + 2][threadIdx.x + 2 + i * (-2)] == 255)
							dst(y, x + i * (-1)) = 255;
					}
				}
			}
		}

	}


	void fillGaps_caller(
		PtrStepSzb src,
		PtrStepb dst)
	{
		dim3 threads(16, 16);
		dim3 blocks((src.cols + threads.x - 1) / threads.x, (src.rows + threads.y - 1) / threads.y);
		FillGaps_kernel << <blocks, threads, 0, 0 >> > (src, dst);
	}

}
