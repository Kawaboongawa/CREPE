// main.cpp : définit le point d'entrée pour l'application console.
//

#include <stdio.h>
#include <iostream>
#include <QApplication>

#include "crepe.hh"


#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	//camera 
	//cv::VideoCapture capture(0);

	//video 
	cv::VideoCapture capture("C:\\Users\\Cyril\\Desktop\\chess_video\\chess\\black_queen&king.avi");
	if (!capture.isOpened())
		return 1;
	unsigned int width = 1280;
	unsigned int height = 720;
	crepe::Crepe crepe(std::make_pair(width, height), capture);
	crepe.start();
	return a.exec();

}


