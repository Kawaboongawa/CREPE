#pragma once

#include "cuda_tools.cuh"


void compute_raylight(
	void *	           frame,
	const uint frame_resolution);
