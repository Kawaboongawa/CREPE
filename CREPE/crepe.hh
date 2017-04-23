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
#include <stdio.h>
#include <direct.h>

#include <QMainWindow>
#include <QThread>

#include <opencv2/cudaarithm.hpp>
#include <opencv2/opencv.hpp>

#include "fourier_descriptor.hh"
#include "filter_handler.hh"
#include "database_handler.hh"

using namespace cv::cuda;
namespace crepe
{

	class Crepe : public QThread
	{

	public:
		Crepe(
			const std::pair<int, int>& screen_size, 
			cv::VideoCapture capture);

		~Crepe();

		void run() override;

		void process(cv::Mat src);

		void draw_contours(
			cv::Mat src, 
			std::vector<std::vector<cv::Point> > contours, 
			std::vector<std::string> names);

	private:

		cv::VideoCapture capture_;

		std::pair<int, int> screen_size_;

		int fps_;

		filter::FilterHandler filter_;

		Database database_;

	};
}