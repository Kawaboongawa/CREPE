#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda/common.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include "tools.hh"

using namespace cv::cuda;

/*
**We implement here morphological thinning  as explained in http://homepages.inf.ed.ac.uk/rbf/HIPR2/thin.htm.
*/


namespace filter
{
	void morphological_edge_thinning_caller(PtrStepSzb src, PtrStepb dst);
}