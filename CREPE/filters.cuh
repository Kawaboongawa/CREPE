#pragma once


#include <opencv2/cudaarithm.hpp>
#include "tools.hh"

//void swap_rb_kernel(const cv::cuda::GpuMat a, )

void cyan_screen(
	void *	           frame,
	const uint frame_resolution);

void swap_rb_caller(
	const cv::cuda::PtrStepSz<uchar3>& src,
	cv::cuda::PtrStep<uchar3> dst);
