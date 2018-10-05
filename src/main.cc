// main.cpp�: d�finit le point d'entr�e pour l'application console.
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
	unsigned int width = 640;
	unsigned int height = 480;
	crepe::Crepe crepe(std::make_pair(width, height));
	crepe.start();
	return a.exec();

}


