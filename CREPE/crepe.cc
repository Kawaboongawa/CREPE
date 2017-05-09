#include "crepe.hh"
#include <vector>

namespace crepe
{
	Crepe::Crepe(const std::pair<int, int>& screen_size)
		: screen_size_(screen_size)
		, capture_(cv::VideoCapture(0))
		, filter_(filter::FilterHandler())
		, database_(Database(filter_))
		, picture_path_("")
	{
		SetupWindow setup_window;
		setup_window.show();
		QEventLoop loop;
		connect(&setup_window, SIGNAL(close()), &loop, SLOT(quit()));
		loop.exec();
		SetupWindow::input_kind kind = setup_window.get_kind();
		if (kind == SetupWindow::input_kind::CAMERA)
			fps_ = 20;
		else if (kind == SetupWindow::input_kind::VIDEO)
		{
			capture_ = cv::VideoCapture(setup_window.get_path());
			fps_ = capture_.get(CV_CAP_PROP_FPS);
		}
		else
		{
			capture_.release();
			picture_path_ = setup_window.get_path();
		}

	}

	Crepe::~Crepe()
	{}

	void Crepe::run()
	{
		cv::namedWindow("CREPE", CV_WINDOW_NORMAL);
		cv::resizeWindow("CREPE", screen_size_.first, screen_size_.second);
		if (picture_path_ != "")
			process_picture(cv::imread(picture_path_));
		if (!capture_.isOpened())
			return;
		cv::cuda::setDevice(0);
		int delay = 1000 / fps_;
		cv::Mat frame;
		capture_ >> frame;
		for (;;)
		{
			capture_ >> frame;
			if (frame.empty())
			{
				capture_.set(CV_CAP_PROP_POS_FRAMES, 0);
				continue;
			}
			process(frame); 
			cv::waitKey(delay);
		}
	}


	void Crepe::process_picture(cv::Mat src)
	{
		process(src);
		cv::waitKey(0);
	}

	std::vector<cv::Point> compute_equal_length_points(const std::vector<cv::Point>& arr, int n)
	{
		int size = arr.size();
		std::vector<cv::Point> dst(n);
		float point_dist = static_cast<float>(size) / static_cast<float>(n);
		int index = 0;
		float sum = 0;
		for (int i = 0; i < n; i++)
		{
			dst[i] = arr[static_cast<int>(sum)];
			sum += point_dist;
		}
		return dst;
	}

	std::vector<std::vector<cv::Point>> normalize_shapes(const std::vector<std::vector<cv::Point>>& contours)
	{
		std::vector<std::vector<cv::Point>> dst;
		for (int i = 0; i < contours.size(); i++)
		{
			if (contours[i].size() > 300)
				dst.push_back(compute_equal_length_points(contours[i], 256));
		}
		return dst;
	}

	void Crepe::process(cv::Mat src)
	{
		std::vector<std::vector<cv::Point>> sh_contours;
		GpuMat src_device;
		src_device.upload(src);
		GpuMat device_canny = filter_.compute_edges(src_device);
		cv::Mat canny;
		device_canny.download(canny);
		cv::imshow("Canny", canny);
		cv::findContours(canny, sh_contours, cv::RETR_TREE, cv::CHAIN_APPROX_NONE, cv::Point(0, 0));
		std::vector<std::vector<cv::Point>> contours = normalize_shapes(sh_contours);
		int size = contours.size();
		std::vector<std::string> names(size);
		//function that gets contours
		//FIXME: THIS IS UGLY and need to be changed
		//////////////////
		for (int index = 0; index < size; index++)
		{
			int edge_size = contours[index].size();
			ushort2* edges = (ushort2*)malloc(edge_size * sizeof(ushort2));
			for (int i = 0; i < edge_size; i++)
			{
				edges[i].x = contours[index][i].x;
				edges[i].y = contours[index][i].y;
			}
			//////////////////
			FourierDescriptor fd = FourierDescriptor(edges, contours[index].size());
			free(edges);
			fd.compute_descriptors();
			names[index] = database_.match_object(fd, 128);
		}
		draw_contours(src, contours, names);
	}

	void Crepe::draw_contours(
		cv::Mat src,
		std::vector<std::vector<cv::Point> > contours,
		std::vector<std::string> names)
	{
		// Draw contours
		std::vector<cv::Vec4i> hierarchy;
		int fontFace = cv::FONT_HERSHEY_PLAIN;
		double fontScale = 2;
		int thickness = 3;
		cv::RNG rng(12345);
		cv::Mat drawing = cv::Mat::zeros(src.size(), CV_8UC3);
		std::vector<cv::Rect> boundRect(contours.size());
		std::vector<std::vector<cv::Point> > contours_poly(contours.size());
		for (int i = 0; i < contours.size(); i++)
		{
			cv::approxPolyDP(cv::Mat(contours[i]), contours_poly[i], 3, true);
			boundRect[i] = cv::boundingRect(cv::Mat(contours_poly[i]));
			cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, cv::Point());
			rectangle(src, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
			cv::Point center = (boundRect[i].br() + boundRect[i].tl()) / 2;
			center.x -= boundRect[i].width / 2;
			cv::putText(src, names[i], center, fontFace, fontScale, cv::Scalar::all(255),
				thickness, 8);
		}

		//This is only useful for debug and shows detected edges (colorized)
		imshow("bound", drawing);
		imshow("CREPE", src);
	}
}