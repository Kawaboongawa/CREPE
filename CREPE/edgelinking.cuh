#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda/common.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include "tools.hh"

using namespace cv::cuda;

namespace filter
{

	void fillGaps_caller(
		PtrStepSzb src,
		PtrStepb dst);
}