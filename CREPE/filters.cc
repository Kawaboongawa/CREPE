#include "filters.hh"

namespace crepe
{
	//TODO : IF needed add parameters to constructor to customize filters
	Filter::Filter(int type)
		: type_(type)
		, gauss_filter_(cv::cuda::createGaussianFilter(type, -1, cv::Size(3, 3), 1.5, 1.5))
		, sobel_filter_(cv::cuda::createSobelFilter(type, -1, 1, 1))
		, canny_edge_detector_(cv::cuda::createCannyEdgeDetector(100, 300))
	{}

	Filter::~Filter()
	{}

	void Filter::gauss(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst)
	{
		gauss_filter_->apply(src, dst);
	}

	void Filter::sobel(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst)
	{
		sobel_filter_->apply(src, dst);
	}

	void Filter::canny(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst)
	{
		canny_edge_detector_->detect(src, dst);
	}


	void Filter::rgb2grey(cv::cuda::GpuMat&src, cv::cuda::GpuMat& dst)
	{
		cv::cuda::cvtColor(src, dst, CV_RGB2GRAY, 1);
	}

	void Filter::rgb2binary(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst)
	{
	 	cv::cuda::cvtColor(src, dst, CV_RGB2GRAY, 1);
		cv::cuda::threshold(dst, dst, 128, 255, CV_THRESH_BINARY);
	}

}