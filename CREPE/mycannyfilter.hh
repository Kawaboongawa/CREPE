#pragma once


#include <opencv2/cudaarithm.hpp>
#include<memory>
#include "mycannyfilter.cuh"

using namespace cv::cuda;

namespace canny
{
	class MyCannyFilter
	{
	public:
		MyCannyFilter(int min_thresh, int max_thresh);
		~MyCannyFilter();

		void init(cv::Size size);
		void apply(GpuMat src, GpuMat dst);

	private:
	
		const int min_thresh_;
		const int max_thresh_;

		//gradient of every pixel on x axis
		std::shared_ptr<GpuMat> dx_;
		//gradient of every pixel on y axis
		std::shared_ptr<GpuMat> dy_;
		//gradient magnitude 
		std::shared_ptr<GpuMat> mag_;
		/** 
		** gradient direction. NB : Opencv manage to compute directly
		** the non-maxima supression without using a container. If VRAM 
		** (or perfs) is needed  in the future further work on this 
		** function might be needed.
		**/
		std::shared_ptr<GpuMat> atan_;
		//type of edge for every pixels (0, 1 or 2)
		std::shared_ptr<GpuMat> map_;

		std::shared_ptr<GpuMat> shared_edges_;

		std::shared_ptr<GpuMat> tmp_shared_edges_;

	};
}