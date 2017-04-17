#include "mycannyfilter.hh"

namespace canny
{
	MyCannyFilter::MyCannyFilter(int min_thresh, int max_thresh)
		: min_thresh_(min_thresh)
		, max_thresh_(max_thresh)
		, dx_(nullptr)
		, dy_(nullptr)
		, mag_(nullptr)
		, atan_(nullptr)
		, map_(nullptr)
		, shared_edges_(nullptr)
	{
	}

	MyCannyFilter::~MyCannyFilter()
	{}

	void MyCannyFilter::init(cv::Size size)
	{
		dx_ = std::make_shared<GpuMat>(size, CV_32S);
		dy_ = std::make_shared<GpuMat>(size, CV_32S);
		mag_ = std::make_shared<GpuMat>(size, CV_32F);
		atan_ = std::make_shared<GpuMat>(size, CV_32S);
		map_ = std::make_shared<GpuMat>(size, CV_32S);
		shared_edges_ = std::make_shared<GpuMat>(size, CV_16SC2);
		tmp_shared_edges_ = std::make_shared<GpuMat>(size, CV_16SC2);
	}

	void MyCannyFilter::apply(GpuMat src, GpuMat dst)
	{
		// we assume dst is already malloc'd
		getMagnitude_caller(src, *dx_, *dy_, *mag_, *atan_);
		nonMaximaSupress_caller(*mag_, *atan_, *map_, min_thresh_, max_thresh_);
		edgesHysteresisLocal_caller(*map_, (*shared_edges_).ptr<short2>());
		edgesHysteresisGlobal_caller(*map_, (*shared_edges_).ptr<short2>(), (*tmp_shared_edges_).ptr<short2>());
		drawEdges_caller(*map_, dst);
		
	}
}