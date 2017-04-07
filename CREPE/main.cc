// main.cpp : définit le point d'entrée pour l'application console.
//

#include <stdio.h>

#include "crepe.hh"
#include "gui_gl_window.hh"


int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	void* gpu_frame = nullptr;
	unsigned int width = 800;
	unsigned int height = 600;
	auto u = cudaMalloc(&gpu_frame, width * height * sizeof(unsigned char) * 3);
	crepe::Crepe crepe(std::make_pair(800, 600), gpu_frame);
	gui::GuiGLWindow w(width, height, gpu_frame);
	crepe.start();
	w.show();
	return a.exec();

}

