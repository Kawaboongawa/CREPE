#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/opencv.hpp>

#include "filters.cuh"

namespace crepe
{
	class Filter
	{
	public:
		Filter(int type);
		~Filter();
		void gauss(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst);
		void sobel(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst);
		void canny(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst);
		void rgb2binary(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst);
		void rgb2grey(cv::cuda::GpuMat&src, cv::cuda::GpuMat& dst);

	private:

		//type of every filter
		const int type_;

		//Gauss Filter for type_ matrix format
		cv::Ptr<cv::cuda::Filter> gauss_filter_;

		//Sobel Filter for type_ matrix format
		cv::Ptr<cv::cuda::Filter> sobel_filter_;

		//Sobel Filter for CV_8UC3 matrix format
		cv::Ptr<cv::cuda::CannyEdgeDetector> canny_edge_detector_;
	};
}