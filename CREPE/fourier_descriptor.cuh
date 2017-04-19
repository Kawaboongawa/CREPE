#pragma once
#include <opencv2/cudaarithm.hpp>
# include <cufft.h>
#include "tools.hh"

void centroid_distance_caller(
	ushort2* src, 
	float2* dst, 
	uint size,
	float2 centroid);

void compute_descriptors_caller(
	float2* src,
    float2*	dst,
	uint    size,
	const cufftHandle	plan1D);