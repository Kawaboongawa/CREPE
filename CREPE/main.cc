// main.cpp : définit le point d'entrée pour l'application console.
//

#include <stdio.h>
#include <iostream>
#include <QApplication>

#include "crepe.hh"

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	cv::VideoCapture capture("C:\\Users\\Cyril\\Desktop\\water_bottle\\video.avi");
	if (!capture.isOpened())
		return 1;
	unsigned int width = 800;
	unsigned int height = 600;
	void* gpu_frame = nullptr;
	crepe::Crepe crepe(std::make_pair(600, 1024), gpu_frame, capture);
	crepe.start();
	return a.exec();

}

