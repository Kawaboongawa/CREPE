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
	}

	FourierDescriptor::~FourierDescriptor()
	{
		cufftDestroy(plan1d_);

		cudaFree(gpu_descriptors_);

		if (shape_ != nullptr)
			cudaFree(shape_);
	}

	void FourierDescriptor::compute_descriptors()
	{
		centroid_distance_caller(shape_, gpu_descriptors_, size_, centroid_);
		cudaFree(shape_);
		shape_ = nullptr;
		compute_centroid_signature_caller(gpu_descriptors_, gpu_descriptors_, size_, plan1d_);

		//example
		std::vector<float2> a(size_);
		cudaMemcpy(&a[0], gpu_descriptors_, size_* sizeof(float2), cudaMemcpyDeviceToHost);
	}

	float FourierDescriptor::compare_descriptors(const FourierDescriptor& desc, uint size)
	{

		float value = compare_descriptors_caller(gpu_descriptors_ + 1, desc.gpu_descriptors_ + 1, size);
		
		/*std::vector<float2> a(size);
		cudaMemcpy(&a[0], gpu_descriptors_, size * sizeof(float2), cudaMemcpyDeviceToHost);
		std::vector<float2> b(size);
		cudaMemcpy(&b[0], desc.gpu_descriptors_, size * sizeof(float2), cudaMemcpyDeviceToHost);
		*/
		return value;
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