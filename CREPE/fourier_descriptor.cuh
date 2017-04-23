#pragma once
#include <opencv2/cudaarithm.hpp>
# include <cufft.h>
#include <algorithm>
#include "tools.hh"

void centroid_distance_caller(
	ushort2* src, 
	float2* dst, 
	uint size,
	float2 centroid);

void compute_centroid_signature_caller(
	float2* src,
    float2*	dst,
	uint    size,
	const cufftHandle	plan1D);

float compare_descriptors_caller(
	float2* src1,
	float2* src2,
	uint    size);