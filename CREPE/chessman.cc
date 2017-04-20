#include "chessman.hh"

namespace crepe
{
	Chessman::Chessman(chesspiece chesspiece)
		: piece_(chesspiece)
		, descriptors_(std::vector<FourierDescriptor>())
	{}

	Chessman::~Chessman()
	{}

	void Chessman::add_descriptor(FourierDescriptor descriptor)
	{
		descriptors_.push_back(descriptor);
	}
}