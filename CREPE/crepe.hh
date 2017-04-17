/*! \file
*
* Core class of the Crepe Project  */
#pragma once

#include <string>
#include <sstream>
#include <chrono>
#include <memory>
#include <iostream>
#include <windows.h>

#include <QMainWindow>
#include <QThread>

#include <opencv2/cudaarithm.hpp>
#include <opencv2/opencv.hpp>


#include "filters.hh"
#include "mycannyfilter.hh"

using namespace cv::cuda;
namespace crepe
{

	class Crepe : public QThread
	{

	public:
		Crepe(const std::pair<int, int>& screen_size, cv::VideoCapture capture);

		~Crepe();

		void run() override;

		GpuMat process(cv::cuda::GpuMat src);

		// take a CV_8UC3 matrix and return a CV_8UC1 one with edges contours
		GpuMat compute_edges(GpuMat& src);

		void init_database();



	private:

		cv::VideoCapture capture_;

		std::pair<int, int> screen_size_;

		int fps_;

		Filter rgb_filter_;

	    Filter c1_filter_;

		canny::MyCannyFilter canny_filter_;

	};
}