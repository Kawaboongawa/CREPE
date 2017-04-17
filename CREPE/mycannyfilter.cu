
#include "mycannyfilter.cuh"

namespace canny
{

	__device__ int counter = 0;

	__global__ void getMagnitude_kernel(
		const PtrStepSz<uchar> src,
		PtrStepi dx,
		PtrStepi dy,
		PtrStepf mag,
		PtrStepi atan)
	{
		const int x = blockIdx.x * blockDim.x + threadIdx.x;
		const int y = blockIdx.y * blockDim.y + threadIdx.y;

		if (y >= src.rows - 1 || x >= src.cols - 1)
			return;

		int dxVal = (src(y - 1, x + 1) + 2 * src(y, x + 1) + src(y + 1, x + 1)) - (src(y - 1, x - 1) + 2 * src(y, x - 1) + src(y + 1, x - 1));
		int dyVal = (src(y + 1, x - 1) + 2 * src(y + 1, x) + src(y + 1, x + 1)) - (src(y - 1, x - 1) + 2 * src(y - 1, x) + src(y - 1, x + 1));

		dx(y, x) = dxVal;
		dy(y, x) = dyVal;

		mag(y, x) = sqrt(static_cast<float>(dxVal * dxVal + dyVal * dyVal));

		/*This part can be improved (probably)*/

		atan(y, x) = static_cast<int>(atan2f(dyVal, dxVal) * PI_180 + 0.5);


	}

	void getMagnitude_caller(
		const PtrStepSz<uchar>& src,
		PtrStepi dx,
		PtrStepi dy,
		PtrStepf mag,
		PtrStepi atan)
	{
		dim3 threads(16, 16);
		dim3 blocks((src.cols + threads.x - 1) / threads.x, (src.rows + threads.y - 1) / threads.y);
		getMagnitude_kernel << <blocks, threads, 0, 0 >> > (src, dx, dy, mag, atan);
	}

	
	__global__ void nonMaximaSupress_kernel(
		PtrStepSzf mag,
		PtrStepi atan,
		PtrStepi map,
		const int low_thresh,
		const int high_thresh)
	{

		
		const int x = blockIdx.x * blockDim.x + threadIdx.x;
		const int y = blockIdx.y * blockDim.y + threadIdx.y;

		if (y == 0 || y >= mag.rows - 1 || x == 0 || x >= mag.cols - 1)
			return;

		// 0 - the pixel can not belong to an edge
		// 1 - the pixel is a weak edge
		// 2 - the pixel is a strong edge
		int edge_type = 0;
		const float m = mag(y, x);
		if (m > low_thresh)
		{
			int dir = atan(y, x);
			if (dir < 0)
				dir *= -1;

			if (dir >= 158 || dir < 23)
			{
				if (m >= mag(y, x - 1) && m >= mag(y, x + 1))
					edge_type = 1 + static_cast<int>(m > high_thresh);
			}
			else if (dir < 68)
			{
				if (m >= mag(y - 1, x + 1) && m >= mag(y + 1, x - 1))
					edge_type = 1 + static_cast<int>(m > high_thresh);
			}
			else if (dir < 113)
			{
				if (m >= mag(y - 1, x) && m >= mag(y + 1, x))
					edge_type = 1 + static_cast<int>(m > high_thresh);
			}
			else
			{
				if (m >= mag(y - 1, x - 1) && m >= mag(y + 1, x + 1))
					edge_type = 1 + static_cast<int>(m > high_thresh);
			}
		}
		map(y, x) = edge_type;
	}

	void nonMaximaSupress_caller(
		PtrStepSzf mag,
		PtrStepi atan,
		PtrStepi map,
		const int low_thresh,
		const int high_thresh)
	{
		dim3 threads(16, 16);
		dim3 blocks((mag.cols + threads.x - 1) / threads.x, (mag.rows + threads.y - 1) / threads.y);
		nonMaximaSupress_kernel << <blocks, threads, 0, 0 >> > (mag, atan, map, low_thresh, high_thresh);
	}



	__device__ __forceinline__ bool checkIdx(int y, int x, int rows, int cols)
	{
		return (y >= 0) && (y < rows) && (x >= 0) && (x < cols);
	}

	__global__ void edgesHysteresisLocal_kernel(
		PtrStepSzi map,
		short2* shared_edges)
	{
		__shared__ volatile int smem[18][18];

		const int x = blockIdx.x * blockDim.x + threadIdx.x;
		const int y = blockIdx.y * blockDim.y + threadIdx.y;

		smem[threadIdx.y + 1][threadIdx.x + 1] = checkIdx(y, x, map.rows, map.cols) ? map(y, x) : 0;
		if (threadIdx.y == 0)
			smem[0][threadIdx.x + 1] = checkIdx(y - 1, x, map.rows, map.cols) ? map(y - 1, x) : 0;
		if (threadIdx.y == blockDim.y - 1)
			smem[blockDim.y + 1][threadIdx.x + 1] = checkIdx(y + 1, x, map.rows, map.cols) ? map(y + 1, x) : 0;
		if (threadIdx.x == 0)
			smem[threadIdx.y + 1][0] = checkIdx(y, x - 1, map.rows, map.cols) ? map(y, x - 1) : 0;
		if (threadIdx.x == blockDim.x - 1)
			smem[threadIdx.y + 1][blockDim.x + 1] = checkIdx(y, x + 1, map.rows, map.cols) ? map(y, x + 1) : 0;
		if (threadIdx.x == 0 && threadIdx.y == 0)
			smem[0][0] = checkIdx(y - 1, x - 1, map.rows, map.cols) ? map(y - 1, x - 1) : 0;
		if (threadIdx.x == blockDim.x - 1 && threadIdx.y == 0)
			smem[0][blockDim.x + 1] = checkIdx(y - 1, x + 1, map.rows, map.cols) ? map(y - 1, x + 1) : 0;
		if (threadIdx.x == 0 && threadIdx.y == blockDim.y - 1)
			smem[blockDim.y + 1][0] = checkIdx(y + 1, x - 1, map.rows, map.cols) ? map(y + 1, x - 1) : 0;
		if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1)
			smem[blockDim.y + 1][blockDim.x + 1] = checkIdx(y + 1, x + 1, map.rows, map.cols) ? map(y + 1, x + 1) : 0;

		__syncthreads();

		if (x >= map.cols || y >= map.rows)
			return;

		int n;

#pragma unroll
		for (int k = 0; k < 16; ++k)
		{
			n = 0;

			if (smem[threadIdx.y + 1][threadIdx.x + 1] == 1)
			{
				n += smem[threadIdx.y][threadIdx.x] == 2;
				n += smem[threadIdx.y][threadIdx.x + 1] == 2;
				n += smem[threadIdx.y][threadIdx.x + 2] == 2;

				n += smem[threadIdx.y + 1][threadIdx.x] == 2;
				n += smem[threadIdx.y + 1][threadIdx.x + 2] == 2;

				n += smem[threadIdx.y + 2][threadIdx.x] == 2;
				n += smem[threadIdx.y + 2][threadIdx.x + 1] == 2;
				n += smem[threadIdx.y + 2][threadIdx.x + 2] == 2;
			}

			__syncthreads();

			if (n > 0)
				smem[threadIdx.y + 1][threadIdx.x + 1] = 2;

			__syncthreads();
		}

		const int e = smem[threadIdx.y + 1][threadIdx.x + 1];

		map(y, x) = e;

		n = 0;

		if (e == 2)
		{
			n += smem[threadIdx.y][threadIdx.x] == 1;
			n += smem[threadIdx.y][threadIdx.x + 1] == 1;
			n += smem[threadIdx.y][threadIdx.x + 2] == 1;

			n += smem[threadIdx.y + 1][threadIdx.x] == 1;
			n += smem[threadIdx.y + 1][threadIdx.x + 2] == 1;

			n += smem[threadIdx.y + 2][threadIdx.x] == 1;
			n += smem[threadIdx.y + 2][threadIdx.x + 1] == 1;
			n += smem[threadIdx.y + 2][threadIdx.x + 2] == 1;
		}

		if (n > 0)
		{
			const int ind = atomicAdd(&counter, 1);
			shared_edges[ind] = make_short2(x, y);
		}
	}

	void edgesHysteresisLocal_caller(
		PtrStepSzi map,
		short2* shared_edges)
	{
		dim3 threads(16, 16);
		dim3 blocks((map.cols + threads.x - 1) / threads.x, (map.rows + threads.y - 1) / threads.y);
		edgesHysteresisLocal_kernel << <blocks, threads, 0, 0 >> > (map, shared_edges);
	}

	__constant__ int c_dx[8] = { -1,  0,  1, -1, 1, -1, 0, 1 };
	__constant__ int c_dy[8] = { -1, -1, -1,  0, 0,  1, 1, 1 };

	__global__ void edgesHysteresisGlobal_kernel(PtrStepSzi map, short2* st1, short2* st2, const int count)
	{
		const int stack_size = 512;

		__shared__ int s_counter;
		__shared__ int s_ind;
		__shared__ short2 s_st[stack_size];

		if (threadIdx.x == 0)
			s_counter = 0;

		__syncthreads();

		int ind = blockIdx.y * gridDim.x + blockIdx.x;

		if (ind >= count)
			return;

		short2 pos = st1[ind];

		if (threadIdx.x < 8)
		{
			pos.x += c_dx[threadIdx.x];
			pos.y += c_dy[threadIdx.x];

			if (pos.x > 0 && pos.x < map.cols - 1 && pos.y > 0 && pos.y < map.rows - 1 && map(pos.y, pos.x) == 1)
			{
				map(pos.y, pos.x) = 2;

				ind = atomicAdd(&s_counter, 1);

				s_st[ind] = pos;
			}
		}

		__syncthreads();

		while (s_counter > 0 && s_counter <= stack_size - blockDim.x)
		{
			const int subTaskIdx = threadIdx.x >> 3;
			const int portion = s_counter <= (blockDim.x >> 3) ? s_counter : (blockDim.x >> 3);

			if (subTaskIdx < portion)
				pos = s_st[s_counter - 1 - subTaskIdx];

			__syncthreads();

			if (threadIdx.x == 0)
				s_counter -= portion;

			__syncthreads();

			if (subTaskIdx < portion)
			{
				pos.x += c_dx[threadIdx.x & 7];
				pos.y += c_dy[threadIdx.x & 7];

				if (pos.x > 0 && pos.x < map.cols - 1 && pos.y > 0 && pos.y < map.rows - 1 && map(pos.y, pos.x) == 1)
				{
					map(pos.y, pos.x) = 2;

					ind = atomicAdd(&s_counter, 1);

					s_st[ind] = pos;
				}
			}

			__syncthreads();
		}

		if (s_counter > 0)
		{
			if (threadIdx.x == 0)
			{
				s_ind = atomicAdd(&counter, s_counter);

				if (s_ind + s_counter > map.cols * map.rows)
					s_counter = 0;
			}

			__syncthreads();

			ind = s_ind;

			for (int i = threadIdx.x; i < s_counter; i += blockDim.x)
				st2[ind + i] = s_st[i];
		}
	}

	void edgesHysteresisGlobal_caller(PtrStepSzi map, short2* st1, short2* st2)
	{
		void* counter_ptr;
		cudaGetSymbolAddress(&counter_ptr, canny::counter);

		int count;
		cudaMemcpyAsync(&count, counter_ptr, sizeof(int), cudaMemcpyDeviceToHost);

		while (count > 0)
		{
			cudaMemsetAsync(counter_ptr, 0, sizeof(int));

			const dim3 block(128);
			const dim3 grid(std::min(count, 65535), ((count + 65534) / 65535), 1);

			edgesHysteresisGlobal_kernel << <grid, block, 0, 0 >> > (map, st1, st2, count);

			cudaDeviceSynchronize();

			cudaMemcpyAsync(&count, counter_ptr, sizeof(int), cudaMemcpyDeviceToHost);

			count = std::min(count, map.cols * map.rows);

			short2* tmp = st1;
			st1 = st2;
			st2 = tmp;
		}
	}

	__global__ void drawEdges_kernel(
		PtrStepSzi map,
		PtrStepSzb dst)
	{
		const int x = blockIdx.x * blockDim.x + threadIdx.x;
		const int y = blockIdx.y * blockDim.y + threadIdx.y;

		if (y >= map.rows - 1 || x >= map.cols - 1)
			return;
		if (map(y, x) == 2)
			dst(y, x) = 255;
		else
			dst(y, x) = 0;
	}

	void drawEdges_caller(
		PtrStepSzi map,
		PtrStepSzb dst)
	{
		dim3 threads(16, 16);
		dim3 blocks((map.cols + threads.x - 1) / threads.x, (map.rows + threads.y - 1) / threads.y);
		drawEdges_kernel << <blocks, threads, 0, 0 >> > (map, dst);
	}

}

