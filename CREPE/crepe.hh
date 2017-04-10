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

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>

# include <cuda_gl_interop.h>
# include <cuda_runtime.h>

#include <QMainWindow>
#include <QThread>

#include "filters.cuh"


namespace crepe
{

	class Crepe : public QThread
	{

	public:
		Crepe(const std::pair<int, int>& screen_size, void* gpu_frame, cv::VideoCapture capture);

		~Crepe();

		void run() override;

		void process();

	private:

		cv::VideoCapture capture_;

		std::pair<int, int> screen_size_;

		void* gpu_frame_;

		int fps_;

	};
}