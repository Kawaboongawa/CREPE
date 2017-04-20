#pragma once

#include <string>

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

		void add_descriptor(FourierDescriptor descriptor);

	private:

		chesspiece piece_;

		std::vector<FourierDescriptor> descriptors_;

	};
}