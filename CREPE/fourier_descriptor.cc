#include "fourier_descriptor.hh"

namespace crepe
{
	FourierDescriptor::FourierDescriptor(ushort2* shape, int size)
		: shape_(shape)
		, size_(size)
		, centroid_(make_float2(0, 0))
		, plan1d_(0)
	{
		cufftPlan1d(&plan1d_, size, CUFFT_C2C, 1);
		compute_centroid();
		cudaMalloc(&gpu_descriptors_, size * sizeof(float2));
		cudaMalloc(&shape_, size * sizeof(ushort2));
		cudaMemcpy(shape_, shape, size * sizeof(ushort2), cudaMemcpyHostToDevice);
		free(shape);
	}

	FourierDescriptor::~FourierDescriptor()
	{
		cufftDestroy(plan1d_);

		cudaFree(gpu_descriptors_);

		cudaFree(shape_);
	}

	void FourierDescriptor::compute_descriptors()
	{
		centroid_distance_caller(shape_, gpu_descriptors_, size_, centroid_);
		compute_descriptors_caller(gpu_descriptors_, gpu_descriptors_, size_, plan1d_);
		/*float2* a = (float2*)malloc(300 * sizeof(float2));
		cudaMemcpy(a, gpu_descriptors_, 100 * sizeof(float2), cudaMemcpyDeviceToHost);
		float2 x0 = a[0];
		float2 x1 = a[1];
		float2 x2 = a[2];
		float2 x3 = a[3];
		float2 x4 = a[4];
		float2 x5 = a[5];*/
	}

	void FourierDescriptor::compute_centroid()
	{
		for (int i = 0; i < size_; i++)
		{
			ushort2 curr = shape_[i];
			centroid_.x += static_cast<float>(curr.x);
			centroid_.y += static_cast<float>(curr.y);
		}
		centroid_.x /= static_cast<float>(size_);
		centroid_.y /= static_cast<float>(size_);
	}
}