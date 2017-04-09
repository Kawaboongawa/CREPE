// main.cpp : définit le point d'entrée pour l'application console.
//

#include <stdio.h>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>

#include "crepe.hh"
#include "gui_gl_window.hh"



void process() {
	cv::cuda::setDevice(0);
	cv::VideoCapture capture("C:\\Users\\Cyril\\Desktop\\water_bottle\\video.avi");
	cv::Mat templ_h = cv::imread("C:\\Users\\Cyril\\Desktop\\water_bottle\\photo.jpg");
	cv::cuda::GpuMat templ_d(templ_h);
	cv::cuda::GpuMat image_d, result;
	int fpms = 1000 / capture.get(CV_CAP_PROP_FPS);
	cv::namedWindow("yolo", CV_WINDOW_NORMAL);
	cv::resizeWindow("yolo", 600, 1024);
	for (;;)
	{
		cv::Mat frame;
		capture >> frame;
		if (frame.empty())
		{
			capture.set(CV_CAP_PROP_POS_FRAMES, 0);
			continue;
		}
		image_d.upload(frame);
        //cv::cuda::TemplateMatching* matcher = cv::cuda::createTemplateMatching(CV_8U, CV_TM_CCORR);
		//matcher->match(image_d, templ_d, result);
		//cv::Ptr<cv::cuda::Filter> gauss = cv::cuda::createGaussianFilter(image_d.type(), -1, cv::Size(11, 11), 1.5, 1.5);
        //cv::Ptr<cv::cuda::Filter> laplace = cv::cuda::createLaplacianFilter(image_d.type(), -1);
		//laplace->apply(image_d, result);
		double max_value;
		cv::Mat result_host;
		image_d.download(result_host);
		cv::imshow("yolo", result_host);
		cv::waitKey(fpms);
		/*if (max_value > 10.f)
			drawRectangleAndShowImage(location, ...);*/
	}
}


	int main(int argc, char* argv[])
	{
		try
		{
			/*cv::Mat src_host = cv::imread("C:\\Users\\Cyril\\Desktop\\water_bottle\\photo.jpg", CV_LOAD_IMAGE_GRAYSCALE);
			cv::cuda::GpuMat dst, src;
			src.upload(src_host);

			cv::cuda::threshold(src, dst, 128.0, 255.0, CV_THRESH_BINARY);

			cv::Mat result_host;
			dst.download(result_host);

			cv::imshow("Result", result_host);
			cv::waitKey();*/
			process();
		}
		catch (const cv::Exception& ex)
		{
			std::cout << "Error: " << ex.what() << std::endl;
		}
		return 0;
	}

/*int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	void* gpu_frame = nullptr;
	unsigned int width = 800;
	unsigned int height = 600;
	auto u = cudaMalloc(&gpu_frame, width * height * sizeof(unsigned char) * 3);
	crepe::Crepe crepe(std::make_pair(800, 600), gpu_frame);
	gui::GuiGLWindow w(width, height, gpu_frame);
	crepe.start();
	w.show();
	return a.exec();

}*/

