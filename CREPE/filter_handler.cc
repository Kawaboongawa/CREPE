#include "filter_handler.hh"

namespace filter
{
	FilterHandler::FilterHandler(bool customs_filters)
		: rgb_filter_(Filter(CV_8UC3))
		, c1_filter_(Filter(CV_8U))
		, canny_filter_(MyCannyFilter(100, 300))
	{}

	FilterHandler::~FilterHandler()
	{}


	GpuMat FilterHandler::compute_edges(GpuMat& src)
	{
		cv::cuda::GpuMat dst;
		rgb_filter_.rgb2grey(src, src);
		rgb_filter_.canny(src, dst);

		return dst;
	}

}