#pragma once
# include <algorithm>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>



/*! \brief Getter on max threads in one dimension
**
** Fetch the maximum number of threads available in one dimension
** for a kernel/CUDA computation. It asks directly the
** NVIDIA graphic card. This function, when called several times,
** will only ask once the hardware.
*/
unsigned int get_max_threads();


/*! \brief Getter on max threads in two dimensions
**
** Fetch the maximum number of threads available in two dimensions
** for a kernel/CUDA computation. It asks directly the
** NVIDIA graphic card. This function, when called several times,
** will only ask once the hardware.
*/
unsigned int get_max_threads_2d();

/*! \brief Getter on max blocks
**
** Fetch the maximum number of blocks available in one dimension
** for a kernel/CUDA computation. It asks directly the
** NVIDIA graphic card. This function, when called several times,
** will only ask once the hardware.
*/

unsigned int get_max_blocks();

/*! \function Given a problem of *size* elements, compute the lowest number of
* blocks needed to fill a compute grid.
*
* \param nb_threads Number of threads per block. */
unsigned map_blocks_to_problem(const size_t problem_size,
	const unsigned nb_threads);

enum Direction {
	NORTH = 0,
	NORTH_EAST = 1,
	EAST = 2,
	SOUTH_EAST = 3,
	SOUTH = 4,
	SOUTH_WEST = 5,
	WEST = 6,
	NORTH_WEST = 7,
	NODIR = 8
};

void myFindContours(cv::Mat img, cv::OutputArrayOfArrays contours);
bool isOuterBorderStart(cv::Mat img, size_t i, size_t j);
bool isHoleBorderStart(cv::Mat img, size_t i, size_t j);
bool crossesEastBorder(cv::Mat img, bool checked[8], cv::Point p);
cv::Point active(Direction d, cv::Mat img, cv::Point point);
std::vector<cv::Point> directedContour(cv::Mat img, cv::Point ij, cv::Point i2j2, float ndb);
Direction fromTo(cv::Point from, cv::Point to);
Direction clockwise(Direction d);
Direction counterClockwise(Direction d);