#include "tools.hh"

unsigned int get_max_threads()
{
	static int max_threads_per_block_1d;
	static bool initialized = false;

	if (!initialized)
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, 0);
		max_threads_per_block_1d = prop.maxThreadsPerBlock;
		initialized = true;
	}

	return max_threads_per_block_1d;
}

unsigned int get_max_blocks()
{
	static int max_blocks;
	static bool initialized = false;

	if (!initialized)
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, 0);
		max_blocks = prop.maxGridSize[0];
		initialized = true;
	}

	return max_blocks;
}

unsigned map_blocks_to_problem(const size_t problem_size,
	const unsigned nb_threads)
{
	unsigned nb_blocks = static_cast<unsigned>(
		std::ceil(static_cast<float>(problem_size) / static_cast<float>(nb_threads)));

	if (nb_blocks > get_max_blocks())
		nb_blocks = get_max_blocks();

	return nb_blocks;
}

