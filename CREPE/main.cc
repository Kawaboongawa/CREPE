// main.cpp : définit le point d'entrée pour l'application console.
//

#include <stdio.h>
#include <iostream>
#include <QApplication>

#include "crepe.hh"


#include <opencv2/opencv.hpp>
#include <iostream>

/*using namespace cv;
int main(int argc, char** argv)
{
	VideoCapture cap;
	// open the default camera, use something different from 0 otherwise;
	// Check VideoCapture documentation.
	if (!cap.open(0))
		return 0;
	for (;;)
	{
		Mat frame;
		cap >> frame;
		if (frame.empty()) break; // end of video stream
		imshow("this is you, smile! :)", frame);
		if (waitKey(10) == 27) break; // stop capturing by pressing ESC 
	}
	// the camera will be closed automatically upon exit
	// cap.close();
	return 0;
}*/

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


