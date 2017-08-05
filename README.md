# Chess Recognition Enhanced and Parallelized Engine (CREPE)

CREPE is a real-time chess pieces recognition project written in C++. The objective is to match shapes of a given database to those of a picture, a video or a recording camera.
The project is using CUDA for GPU computations. The goal is to get every shapes for each frame and match them to the database's shapes using Fourier descriptors.

<p align="center">                                                                                                                                                      
<img src =sample/demo.gif/>                                                    
</p>

A screenshot of the presentation video 
![Screenshot](sample/crepe1.png)
A second one with 30° angle 
![Screenshot](sample/crepe2.png)

## Hardware requirement

To compile this project you will need Windows 10  (other version might work but this hasn't been tested), and a Nvidia GPU. The requierements
for the GPU are not very high (It worked well with a 660M). A camera will be needed if you want to try your setup, yet not absolutely necessary as 
CREPE can aswell read videos. Video formats and cameras supported are the one supported by OpenCV library. 

## What to do build CREPE

The project was developped on Windows, visual studio 2015. To build the project, you'll need at least CUDA 8.0, OPENCV library compiled with its CUDA functions
(A very good tutorial for windows can be found following this [link](https://inside.mines.edu/~whoff/courses/EENG510/lectures/other/CompilingOpenCV.pdf)), 
and Qt 5.7.1.

Once this has been done you can open CREPE.sln in the root directory and then build the project in visual studio. You can run the CREPE.exe in 
/path/to/directory/CREPE/x64/Release. The QT interface is very straightforward and should not give any problem to use. 

You can test if it is working with the few demo videos in the directory sample . 

## How to modify the database

You can modify the database pretty easily just by adding/deleting pictures in the database directory. Beware that you must not modify the name of directories
for now (bishop, king, knight, ...) in this version of CREPE. The program behaviour and recognition will be modified as you modify the database.

## How it works

#### Image preprocessing

The preprocessing wasn't the main objective of this project and thus we used already implemented OpenCV functions mostly.
* We first apply a gaussian blur to reduce noise.
* We then get the grayscale version of the current frame.
* Finally, A  canny edge detector is applied as it is a very efficient way to detect edges.

#### Edge linking

As a camera approach doen't give perfects results, A edge might have several holes. To prevent this, we did two things. First, we added 
a dilatition to the preprocessing to fill-up the holes. However, our edges were now far bigger than needed. To reduce it, we used the
Zhang-Suen thinning algorithm which is pretty easy to implement on GPU. As a result, the edges were full and given without problems to
the opencv findcontours algorithm.

#### Fourier descriptors

The program first compute the Fourier descriptors for the whole database. To get proper Fourier descriptors this is roughly speaking the followed steps :
* We first normalized every shape to 256 pixels using a very simple algorithm.
* Then, we used the centroïd distance to express every point according to this.
* The FFT can now be applied.
* The points obtained are descriptors and can be compared to other shapes using a simple euclidian distance.

To anyone wanting to dig a bit deeper in fourier descriptors method, I'd recommend this wonderful 
[article](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.67.2229&rep=rep1&type=pdf)  which is very understandable and sums up very well the whole
process.

#### Object matching

Once the database is done, we only need to compute everything we did previously back again with the current frame and to compare the descriptors. The best 
score gives the object that program think the current shape is.
