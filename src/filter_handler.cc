#include "filter_handler.hh"

namespace filter
{
	FilterHandler::FilterHandler()
		: rgb_filter_(Filter(CV_8UC3))
		, c1_filter_(Filter(CV_8U))
	{}

	FilterHandler::~FilterHandler()
	{}

	GpuMat FilterHandler::compute_edges(GpuMat& src)
	{
		cv::cuda::GpuMat dst;
		rgb_filter_.gauss(src, src);
		rgb_filter_.rgb2grey(src, src);
		rgb_filter_.canny(src, dst);
		c1_filter_.dilate(dst, dst);
		//We are suppose to iterate until no change is made between one and another instead of
		//iterating a fixed amount. More perfs can be obtained looking at this point.
		c1_filter_.edge_thinning(dst, dst, 6);
		return dst;
	}

}