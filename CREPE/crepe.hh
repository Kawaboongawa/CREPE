/*! \file
*
* Core class of the Crepe Project  */
#pragma once

#include <string>
#include <sstream>
#include <chrono>
#include <memory>
#include <windows.h>

# include <cuda_gl_interop.h>
# include <cuda_runtime.h>

#include <QMainWindow>
#include <QThread>

#include "filters.cuh"


namespace crepe
{

	class Crepe : public QThread
	{

	public:
		Crepe(const std::pair<int, int>& screen_size, void* gpu_frame);

		~Crepe();

		void run() override;

	private:

		std::pair<int, int> screen_size_;

		void* gpu_frame_;
	};
}