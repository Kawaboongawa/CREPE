#include "filters.hh"

namespace filter
{
	//TODO : IF needed add parameters to constructor to customize filters
	Filter::Filter(int type)
		: type_(type)
		, gauss_filter_(createGaussianFilter(type, -1, cv::Size(3, 3), 1.5, 1.5))
		, sobel_filter_(createSobelFilter(type, -1, 1, 1))
		, canny_edge_detector_(createCannyEdgeDetector(100, 300))
		, dilate_filter_(createMorphologyFilter(cv::MORPH_DILATE, CV_8UC1, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 11))))
		, erode_filter1_(createMorphologyFilter(cv::MORPH_ERODE, CV_8UC1, get_first_structuring_elt()))
		, erode_filter2_(createMorphologyFilter(cv::MORPH_ERODE, CV_8UC1, get_second_structuring_elt()))
	{}

	Filter::~Filter()
	{}

	cv::Mat Filter::get_first_structuring_elt()
	{
		cv::Mat res = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
		for (int i = 0; i < 3; i++)
			res.at<uchar>(0, i) = 0;
		res.at<uchar>(1, 0) = 0;
		res.at<uchar>(1, 2) = 0;
		return res;
	}
	cv::Mat Filter::get_second_structuring_elt()
	{
		cv::Mat res = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
		res.at<uchar>(0, 1) = 0;
		res.at<uchar>(1, 2) = 0;
		return res;
	}


	void Filter::gauss(GpuMat& src, GpuMat& dst)
	{
		cv::MorphTypes a;
		gauss_filter_->apply(src, dst);
	}

	void Filter::sobel(GpuMat& src, GpuMat& dst)
	{
		sobel_filter_->apply(src, dst);
	}

	void Filter::canny(GpuMat& src, GpuMat& dst)
	{
		canny_edge_detector_->detect(src, dst);
	}

	void Filter::dilate(GpuMat& src, GpuMat& dst)
	{
		dilate_filter_->apply(src, dst);
	}

	void Filter::edge_thinning(GpuMat& src, GpuMat& dst, int n)
	{
		GpuMat comp;
		comp.create(src.size(), src.type());
		for (int i = 0; i < n; i++)
		{
			erode_filter1_->apply(src, src);
			complementary_caller(src, comp);
			cv::Mat lol;
			erode_filter2_->apply(comp, comp);
			get_AND_matrix(src, comp, src);
			src.download(lol);
			cv::imshow("yolo", lol);
			cv::waitKey(0);

		}
		src.copyTo(dst);
	}

	void Filter::rgb2grey(GpuMat& src, GpuMat& dst)
	{
		cvtColor(src, dst, CV_RGB2GRAY, 1);
	}

	void Filter::rgb2binary(GpuMat& src, GpuMat& dst)
	{
		cvtColor(src, dst, CV_RGB2GRAY, 1);
		threshold(dst, dst, 128, 255, CV_THRESH_BINARY);
	}

}