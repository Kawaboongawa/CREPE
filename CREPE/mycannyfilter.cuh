#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda/common.hpp>
#include <math.h>
#include <algorithm>
#include <device_functions.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.hpp>
#include <cuda.h>

#include "tools.hh"

// 180/PI value
#define PI_180 57.2957795130823208767981548141051703324054f

using namespace cv::cuda;

namespace filter
{
	void getMagnitude_caller(
		const PtrStepSz<uchar>& src, 
		PtrStepi dx, 
		PtrStepi dy, 
		PtrStepf mag, 
		PtrStepi atan);

	void nonMaximaSupress_caller(
		PtrStepSzf mag,
		PtrStepi atan,
	    PtrStepi map,
	    int low_thresh,
		const int high_thresh);
	

	void edgesHysteresisLocal_caller(
		PtrStepSzi map,
		short2* shared_edges);

	void edgesHysteresisGlobal_caller(
		PtrStepSzi map,
		short2* st1,
		short2* st2);

	void drawEdges_caller(
		PtrStepSzi map,
		PtrStepSzb dst);
}