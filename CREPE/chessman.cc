#include "chessman.hh"

namespace crepe
{
	Chessman::Chessman(chesspiece chesspiece)
		: piece_(chesspiece)
		, descriptors_(std::vector<std::shared_ptr<FourierDescriptor>>())
	{}

	Chessman::~Chessman()
	{}

	void Chessman::add_descriptor(std::shared_ptr<FourierDescriptor> descriptor)
	{
		descriptors_.push_back(descriptor);
	}
}