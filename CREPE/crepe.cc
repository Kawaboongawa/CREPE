#include "crepe.hh"
#include <iostream>

namespace crepe
{
	Crepe::Crepe(const std::pair<int, int>& screen_size, void* gpu_frame)
		: screen_size_(screen_size)
		, gpu_frame_(gpu_frame)
	{}

	Crepe::~Crepe()
	{
		cudaFree(gpu_frame_);
	}


	void Crepe::run()
	{
		int fps = 20;
		const std::chrono::high_resolution_clock::duration frame_frequency = std::chrono::microseconds(1000000 / fps);
		auto next_game_tick = std::chrono::high_resolution_clock::now();
		bool stop_requested = false;
		while (!stop_requested)
		{
			while (std::chrono::high_resolution_clock::now() > next_game_tick && !stop_requested)
			{
				compute_raylight(gpu_frame_, screen_size_.first * screen_size_.second);
				next_game_tick += frame_frequency;
			}
		}
	}
}