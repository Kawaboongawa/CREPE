#pragma once

#include <opencv2/cudaarithm.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include "filters.hh"

namespace filter
{
	class FilterHandler
	{
	public:
		
		FilterHandler();
		~FilterHandler();

		// take a CV_8UC3 matrix and return a CV_8UC1 one with edges contours
		GpuMat compute_edges(GpuMat& src);

	private:

		Filter rgb_filter_;

		Filter c1_filter_;
	};
}