#pragma once

#include <opencv2/cudaarithm.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include "filters.hh"
#include "mycannyfilter.hh"
#include "edgelinking.cuh"

namespace filter
{
	class FilterHandler
	{
	public:
		
		FilterHandler(bool customs_filters = false);
		~FilterHandler();

		// take a CV_8UC3 matrix and return a CV_8UC1 one with edges contours
		GpuMat compute_edges(GpuMat& src);

		void fill_edges(GpuMat& src, GpuMat& dst);

	private:

		Filter rgb_filter_;

		Filter c1_filter_;

		MyCannyFilter canny_filter_;

		bool customs_filters_;

	};
}