#include "filters.cuh"


__global__ void set_red(
	rgb* frame,
	const uint frame_resolution)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	while (index < frame_resolution)
	{
		frame[index].r = 0;
		frame[index].g = 100;
		frame[index].b = 130;
		index += blockDim.x * gridDim.x;
	}
}


void compute_raylight(
	void *	    frame,
	const uint  frame_resolution)
{
	uint threads = get_max_threads();
	uint blocks = map_blocks_to_problem(frame_resolution, threads);
 	set_red <<<blocks, threads, 0, 0>>> (static_cast<rgb *>(frame), frame_resolution);
}