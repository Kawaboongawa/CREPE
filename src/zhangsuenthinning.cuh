#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda/common.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include "tools.hh"

using namespace cv::cuda;

/*
**We implement here zhang-suen thinning  as explained in https://rosettacode.org/wiki/Zhang-Suen_thinning_algorithm.
*/


namespace filter
{
	void zhang_suen_edge_thinning_caller(PtrStepSzb src, PtrStepb dst);
}