#pragma once

#include "cuda_tools.cuh"

#include "tools.hh"

void compute_raylight(
	void *	           frame,
	const uint frame_resolution);
