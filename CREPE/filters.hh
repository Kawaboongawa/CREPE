#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/opencv.hpp>

#include "filters.cuh"
#include "zhangsuenthinning.cuh"

using namespace cv::cuda;

namespace filter
{
	class Filter
	{
	public:
		Filter(int type);
		~Filter();
		void gauss(GpuMat& src, GpuMat& dst);
		void sobel(GpuMat& src, GpuMat& dst);
		void canny(GpuMat& src, GpuMat& dst);
		void dilate(GpuMat& src, GpuMat& dst);
		void edge_thinning(GpuMat& src, GpuMat& dst, int n);
		void rgb2binary(GpuMat& src, GpuMat& dst);
		void rgb2grey(GpuMat& src, GpuMat& dst);
		cv::Mat get_first_structuring_elt();
		cv::Mat get_second_structuring_elt();

	private:

		//type of every filter
		const int type_;

		//Gauss Filter for type_ matrix format
		cv::Ptr<cv::cuda::Filter> gauss_filter_;

		//Sobel Filter for type_ matrix format
		cv::Ptr<cv::cuda::Filter> sobel_filter_;

		//Sobel Filter for CV_8UC3 matrix format
		cv::Ptr<CannyEdgeDetector> canny_edge_detector_;

		//Dilate Filter
		cv::Ptr<cv::cuda::Filter> dilate_filter_;

		//Edge_thinning1 Filter
		cv::Ptr<cv::cuda::Filter> erode_filter1_;

		//Edge_thinning2 Filter
		cv::Ptr<cv::cuda::Filter> erode_filter2_;
	};
}