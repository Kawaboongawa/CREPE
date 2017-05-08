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

	void FilterHandler::fill_edges(GpuMat& src, GpuMat& dst)
	{
		fillGaps_caller(src, dst);
	}

	GpuMat FilterHandler::compute_edges(GpuMat& src)
	{
		cv::cuda::GpuMat dst;

		//Is Gauss useful ?
		rgb_filter_.gauss(src, src);
		////////////////////
		rgb_filter_.rgb2grey(src, src);
		rgb_filter_.canny(src, dst);
		cv::Mat lol;
		dst.download(lol);
		cv::imshow("canny", lol);
		c1_filter_.dilate(dst, dst);
		dst.download(lol);
		cv::imshow("dilated", lol);
		c1_filter_.edge_thinning(dst, dst, 10);
		//fill_edges(dst, dst);
		return dst;
	}

}