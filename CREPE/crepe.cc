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

		rng = cv::RNG(12345);

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
		std::vector<std::vector<cv::Point>> contours;
		res.download(tmp);
		
		cv::Mat dist_8u;
		std::vector<cv::Vec4i> hierarchy;

		tmp.convertTo(dist_8u, CV_8U);
		myFindContours(dist_8u, contours);
		//findContours(dist_8u, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

		/// Draw contours
		for (int i = 0; i< contours.size(); i++)
		{
			cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			drawContours(dist_8u, contours, i, color, 2, 8, hierarchy, 0, cv::Point());
		}

		return GpuMat(dist_8u);
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
		cv::Mat src; 
		cv::Mat src_gray;
		cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
			1, 1, 1,
			1, -8, 1,
			1, 1, 1);
		std::vector<std::vector<cv::Point>> contours;
		std::vector<cv::Vec4i> hierarchy;
		
		//src = cv::imread("..\\..\\database\\knight\\IMAG1366.png");
		
		char currentPath[FILENAME_MAX];
		_getcwd(currentPath, sizeof (currentPath));
		src = cv::imread("..\\..\\database\\knight\\IMAG1366.png");
		
		//cv::filter2D(src, src, CV_8UC3, kernel);

		cv::imshow("original photo", src);
		GpuMat srcdev;
		srcdev.upload(src);
		GpuMat res = compute_edges(srcdev);
		res.download(src_gray);
		
		//cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);
		//cv::imshow("gray photo", src_gray);
		//Canny(src_gray, canny_output, 100,  300, 3);

		cv::findContours(src_gray, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

		// Draw contours
		cv::RNG rng(12345);
		cv::Mat drawing = cv::Mat::zeros(src_gray.size(), CV_8UC3);
		std::vector<cv::Rect> boundRect(contours.size());
		std::vector<std::vector<cv::Point> > contours_poly(contours.size());

		for (int i = 0; i < contours.size(); i++)
		{
			cv::approxPolyDP(cv::Mat(contours[i]), contours_poly[i], 3, true);
			boundRect[i] = cv::boundingRect(cv::Mat(contours_poly[i]));
			if (contours[i].size() < 100)
				continue;

			boundRect[i] = cv::boundingRect(cv::Mat(contours_poly[i]));
			cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, cv::Point());
			rectangle(src, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
		}

	
		int index = 0;
		for (int i = 0; i < contours.size(); i++)
		{
			if (contours[i].size() > contours[index].size())
				index = i;
		}

		int size = contours[index].size();
		ushort2* edges = (ushort2*)malloc(size * sizeof(ushort2));
		for (int i = 0; i < size; i++)
		{
			edges[i].x = contours[index][i].x;
			edges[i].y = contours[index][i].y;
		}
		
		FourierDescriptor fd(edges, size);
		fd.compute_descriptors();
		cv::namedWindow("Contours", cv::WINDOW_AUTOSIZE);
		cv::imshow("Contours", drawing);
		//cv::waitKey(0);
	}

}