#pragma once


#include <opencv2/cudaarithm.hpp>
#include "tools.hh"

using namespace cv::cuda;
//void swap_rb_kernel(const cv::cuda::GpuMat a, )

void set_cyan_caller(
	void *	           frame,
	const uint frame_resolution);

void swap_rb_caller(
	const cv::cuda::PtrStepSz<uchar3>& src,
	cv::cuda::PtrStep<uchar3> dst);

void remove_b_caller(cv::cuda::PtrStepSz<uchar3> src);

void complementary_caller(PtrStepSzb src, PtrStepb dst);

void get_AND_matrix(PtrStepSzb src1, PtrStepb src2, PtrStepb dst);
