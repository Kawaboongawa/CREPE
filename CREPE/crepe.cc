#include "crepe.hh"
#include <vector>

namespace crepe
{
	Crepe::Crepe(const std::pair<int, int>& screen_size, cv::VideoCapture capture)
		: screen_size_(screen_size)
		, capture_(capture)
		, rgb_filter_(Filter(CV_8UC3))
		, c1_filter_(Filter(CV_8U))
	    , canny_filter_(canny::MyCannyFilter(100, 300))
	{
		//camera
	    //fps_ = 20;

		//video
		fps_ = capture_.get(CV_CAP_PROP_FPS);
	}

	Crepe::~Crepe()
	{}


	void yolo(cv::Mat& dst, cv::Mat& src)
	{
		distanceTransform(src, dst, CV_DIST_L2, 3);
		cv::normalize(dst, dst, 0, 1., cv::NORM_MINMAX);
		cv::threshold(dst, dst, .4, 1., CV_THRESH_BINARY);
		// Dilate a bit the dist image
		cv::Mat kernel1 = cv::Mat::ones(3, 3, CV_8UC1);
		dilate(dst, dst, kernel1);
		erode(dst, dst, kernel1);
		// Create the CV_8U version of the distance image
		// It is needed for findContours()
	/*	cv::Mat dist_8u;
		dst.convertTo(dist_8u, CV_8U);
		// Find total markers
		std::vector<std::vector<cv::Point> > contours;
		findContours(dist_8u, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
		// Create the marker image for the watershed algorithm
		cv::Mat markers = cv::Mat::zeros(dst.size(), CV_32SC1);
		// Draw the foreground markers
		for (size_t i = 0; i < contours.size(); i++)
			cv::drawContours(markers, contours, static_cast<int>(i), cv::Scalar::all(static_cast<int>(i) + 1), -1);
		// Draw the background marker
		cv::circle(markers, cv::Point(5, 5), 3, CV_RGB(255, 255, 255), -1);*/

	}

	void Crepe::run()
	{
		cv::cuda::setDevice(0);
		int delay = 1000 / fps_;
		cv::namedWindow("CREPE", CV_WINDOW_NORMAL);
		cv::resizeWindow("CREPE", screen_size_.first, screen_size_.second);
		cv::Mat frame;
		capture_ >> frame;
		canny_filter_.init(frame.size());
		for (;;)
		{
			cv::cuda::GpuMat frame_device;
			frame_device.upload(frame);
			cv::cuda::GpuMat result_device = process(frame_device);
			cv::Mat result_host;
			result_device.download(result_host);
			//yolo(result_host, result_host);
			cv::imshow("CREPE", result_host);
			cv::waitKey(delay);
			capture_ >> frame;
			if (frame.empty())
			{
				capture_.set(CV_CAP_PROP_POS_FRAMES, 0);
				capture_ >> frame;
			}
		}
	}

	cv::cuda::GpuMat Crepe::process(cv::cuda::GpuMat src) {

		//cv::cuda::GpuMat dst;
		//rgb_filter_.sobel(src, dst);
		cv::cuda::GpuMat res;
		//res.create(src.size(), CV_8UC);
		rgb_filter_.rgb2grey(src, res);
		rgb_filter_.rgb2grey(src, src);
		cv::Ptr<cv::cuda::CannyEdgeDetector> canny_edge_detector = cv::cuda::createCannyEdgeDetector(100, 300);
		canny_filter_.apply(res, res);
		canny_edge_detector->detect(src, src);
		//c1_filter_.sobel(res, res);
		cv::Mat yolo;
		src.download(yolo);
		cv::imshow("le vrai canny", yolo);
		return res;
	}

}