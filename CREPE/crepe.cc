#include "crepe.hh"

namespace crepe
{
	Crepe::Crepe(const std::pair<int, int>& screen_size, void* gpu_frame, cv::VideoCapture capture)
		: screen_size_(screen_size)
		, gpu_frame_(gpu_frame)
		, capture_(capture)
	{
		fps_ = capture_.get(CV_CAP_PROP_FPS);
	}

	Crepe::~Crepe()
	{
		cudaFree(gpu_frame_);
	}


	void Crepe::run()
	{
		cv::cuda::setDevice(0);
		int delay = 1000 / fps_;
		cv::namedWindow("CREPE", CV_WINDOW_NORMAL);
		cv::resizeWindow("CREPE", screen_size_.first, screen_size_.second);
		for (;;)
		{
			process();
			cv::waitKey(delay);
		}
	}

	void Crepe::process() {

		//cv::Mat templ_h = cv::imread("C:\\Users\\Cyril\\Desktop\\water_bottle\\photo.jpg");
		//cv::cuda::GpuMat templ_d(templ_h);
		cv::cuda::GpuMat image_d;
		cv::cuda::GpuMat result;
		cv::Mat frame;
		capture_ >> frame;
		if (frame.empty())
		{
			capture_.set(CV_CAP_PROP_POS_FRAMES, 0);
			return;
		}
		image_d.upload(frame);
		result.create(image_d.size(), image_d.type());

		//example of sobel & gauss algorithm.
		//cv::cuda::TemplateMatching* matcher = cv::cuda::createTemplateMatching(CV_8U, CV_TM_CCORR);
		//matcher->match(image_d, templ_d, result);
		//cv::Ptr<cv::cuda::Filter> gauss = cv::cuda::createGaussianFilter(image_d.type(), -1, cv::Size(11, 11), 1.5, 1.5);
		//cv::Ptr<cv::cuda::Filter> sobel = cv::cuda::createSobelFilter(image_d.type(), -1, 2, 2);
		//sobel->apply(image_d, result);


 		swap_rb_caller(image_d, result);
		cv::Mat result_host;
		result.download(result_host);
		cv::imshow("CREPE", result_host);
	}

}