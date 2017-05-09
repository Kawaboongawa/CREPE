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
	//cv::VideoCapture capture("C:\\Users\\Cyril\\Desktop\\chess_video\\chess\\full.mp4");
	unsigned int width = 640;
	unsigned int height = 480;
	crepe::Crepe crepe(std::make_pair(width, height));
	crepe.start();
	return a.exec();

}


