#pragma once

#include <opencv2/cudaarithm.hpp>
#include <opencv2/opencv.hpp>
#include <cufft.h>
#include "fourier_descriptor.cuh"

using namespace cv::cuda;

namespace crepe
{
	class FourierDescriptor
	{
	public:

		FourierDescriptor(ushort2* shape, int size);
		~FourierDescriptor();

		void compute_descriptors();


	private:
		void compute_centroid();

	private:

		ushort2* shape_;

		int size_;

		float2 centroid_;

		float2* gpu_descriptors_;

		cufftHandle plan1d_;
	};
}