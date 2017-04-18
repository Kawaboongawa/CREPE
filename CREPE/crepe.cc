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
		init_database();
	}

	Crepe::~Crepe()
	{}

	void Crepe::run()
	{
		cv::cuda::setDevice(0);
		int delay = 1000 / fps_;
		cv::namedWindow("CREPE", CV_WINDOW_NORMAL);
		//cv::resizeWindow("CREPE", screen_size_.first, screen_size_.second);
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

	cv::cuda::GpuMat Crepe::process(GpuMat src) {

		GpuMat res = compute_edges(src);
		cv::Mat tmp;
		std::vector<std::vector<cv::Point> > contours;
		res.download(tmp);
		cv::Mat dist_8u;
		tmp.convertTo(dist_8u, CV_8U);
		//findContours(dist_8u, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
		return res;
	}

	GpuMat Crepe::compute_edges(GpuMat& src)
	{
		cv::cuda::GpuMat dst;
		rgb_filter_.rgb2grey(src, dst);
		rgb_filter_.canny(dst, dst);
		return dst;
	}

	void Crepe::init_database()
	{
		/*cv::Mat src; 
		cv::Mat src_gray;
		cv::Mat canny_output;
		cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
			1, 1, 1,
			1, -8, 1,
			1, 1, 1);

		src = cv::imread("C:\\Users\\Cyril\\Desktop\\chess_video\\knight\\IMAG1365.jpg");
		//cv::filter2D(src, src, CV_8UC3, kernel);
		cv::imshow("original photo", src);
		GpuMat srcdev;
		srcdev.upload(src);
		GpuMat res = compute_edges(srcdev);
		res.download(src_gray);
		//cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);
		//cv::imshow("gray photo", src_gray);
		//Canny(src_gray, canny_output, 100,  300, 3);
		cv::imshow("canny photo", src_gray);*/

	}

}