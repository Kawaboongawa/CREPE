#pragma once

#include <string>
#include <memory>

#include "fourier_descriptor.hh"

enum chesspiece
{
	KING = 0,
	QUEEN = 1,
	BISHOP = 2,
	KNIGHT = 3,
	ROOK = 4,
	PAWN = 5
};

namespace crepe
{

	class Chessman
	{
	public:
		Chessman(chesspiece chesspiece);

		~Chessman();

		void add_descriptor(std::shared_ptr<FourierDescriptor> descriptor);

		std::vector<std::shared_ptr<FourierDescriptor>> get_descriptors() 
		{ 
			return descriptors_; 
		}

	private:

		chesspiece piece_;

		std::vector<std::shared_ptr<FourierDescriptor>> descriptors_;

	};
}