#include "tools.hh"

unsigned int get_max_threads()
{
	static int max_threads_per_block_1d;
	static bool initialized = false;

	if (!initialized)
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, 0);
		max_threads_per_block_1d = prop.maxThreadsPerBlock;
		initialized = true;
	}

	return max_threads_per_block_1d;
}


unsigned int get_max_threads_2d()
{
	static int max_threads_per_block_2d;
	static bool initialized = false;

	if (!initialized)
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, 0);
		max_threads_per_block_2d = static_cast<unsigned int>(sqrt(prop.maxThreadsPerBlock));
		initialized = true;
	}

	return max_threads_per_block_2d;
}

unsigned int get_max_blocks()
{
	static int max_blocks;
	static bool initialized = false;

	if (!initialized)
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, 0);
		max_blocks = prop.maxGridSize[0];
		initialized = true;
	}

	return max_blocks;
}

unsigned map_blocks_to_problem(const size_t problem_size,
	const unsigned nb_threads)
{
	unsigned nb_blocks = static_cast<unsigned>(
		std::ceil(static_cast<float>(problem_size) / static_cast<float>(nb_threads)));

	if (nb_blocks > get_max_blocks())
		nb_blocks = get_max_blocks();

	return nb_blocks;
}

void myFindContours(cv::InputOutputArray img, cv::OutputArrayOfArrays contours) {
	float lnbd[] = { 1 };
	float nbd[] = { 1 };
	std::vector<std::vector<cv::Point>> d = contours.getMatVector;

	for (size_t i = 0; i < img.getMat().rows; i++)
	{
		lnbd[0] = 1;
		for (size_t j = 0; j < img.getMat().cols; j++)
		{
			
			uchar fji = img.getMat().at<uchar>(j, i);
			bool isOuter = isOuterBorderStart(img.getMat(), i, j);
			bool isHole = isHoleBorderStart(img.getMat(), i, j);

			if (isOuter || isHole) {
				std::pair<std::vector<cv::Point>, int> border = std::make_pair(std::vector<cv::Point>(), 0);
				std::vector<cv::Point> borderPrime;
				cv::Point from(j, i);
				if (isOuter) {
					nbd[0] += 1;
					from.x -= 1;
					border.second = 0; // OUTER

				}
				else {
					nbd[0] += 1;
					if (fji >= 1) {
						lnbd[0] = fji;
					}

					from.x += 1;
					border.second = 0; // HOLE

				}

				cv::Point ij(j, i);

			}

		}
	}
}

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

Direction clockwise(Direction d) {
	return static_cast<Direction>((static_cast<int>(d) + 1) % 8);
}

Direction fromTo(cv::Point from, cv::Point to) {
	if (from.x == to.x && from.y == to.y) {
		return NODIR;
	}
	if (from.y == to.y) {
		if (from.x < to.x)
			return EAST;
		else
			return WEST;
	}
	else if (from.y < to.y) {
		if (from.x == to.x)
			return SOUTH;
		else if (from.x < to.x)
			return SOUTH_EAST;
		else
			return SOUTH_WEST;
	}
	else {
		if (from.x == to.x)
			return NORTH;
		if (from.x < to.x)
			return NORTH_EAST;
		else
			return NORTH_WEST;
	}
}

std::vector<cv::Point> directedContour(cv::Mat img, cv::Point ij, cv::Point i2j2, float ndb) {
	Direction dir = fromTo(ij, i2j2);
	Direction trace = clockwise(dir);
	std::vector<cv::Point> border;

	cv::Point i1j1(-1, -1);
	while (trace != dir) {
		cv::Point activePixel = active(trace, img, ij);
		if (activePixel.x > -1 && activePixel.y > -1) {
			i1j1 = activePixel;
			break;
		}
		trace = clockwise(trace);
	}
	if (i1j1.x == -1 && i1j1.y == -1)
		return;

	i2j2 = i1j1;
	cv::Point i3j3 = ij;
	bool checked[8] = { false, false, false, false, false, false, false , false};
	while (true) {
		dir = fromTo(i3j3, i2j2);
		trace = clockwise(dir);
		cv::Point i4j4(-1, -1);
		for (size_t i = 0; i < 8; i++)
		{
			checked[i] = false;
		}
		while (true) {
			i4j4 = active(trace, img, i3j3);
			if (i4j4.x > -1 && i4j4.y > -1)
				break;
			checked[static_cast<int>(trace)] = true;
			trace = clockwise(trace);
		}
		// BEGIN TODO: OPERATION PERFORM i3j3, checked
		border.push_back(i3j3);
		if (crossesEastBorder(img, checked, i3j3)) {
			img.at<uchar>(i3j3) = -ndb;
		}
		else if (img.at<uchar>(i3j3) == 1) {
			img.at<uchar>(i3j3) = ndb;
		}
		// END TODO

		if (i4j4.x == ij.x && i4j4.y == ij.y && i3j3.x == i1j1.x && i3j3.y == i1j1.y)
			break;
		i2j2 = i3j3;
		i3j3 = i4j4;
	}


}

cv::Point active(Direction d, cv::Mat img, cv::Point point) {
	int dirx[] = { 0, 1, 1, 1, 0, -1, -1, -1 };
	int diry[] = { -1, -1, 0, 1, 1, 1, 0, -1 };

	int ord = static_cast<int>(d);
	int yy = point.y + diry[ord];
	int xx = point.x + dirx[ord];
	if (xx < 0 || xx >= img.cols || yy < 0 || yy >= img.rows)
		return cv::Point(-1, -1);
	float pix = img.at<float>(yy, xx);
	return pix != 0 ? cv::Point(xx, yy) : cv::Point(-1, -1);
}

bool isOuterBorderStart(cv::Mat img, size_t i, size_t j) {
	return (img.at<uchar>(i, j) == 1 && (j == 0 || img.at<uchar>(i, j - 1) == 0));
}

bool isHoleBorderStart(cv::Mat img, size_t i, size_t j) {
	return (img.at<uchar>(i, j) >= 1 && (j == img.cols - 1 || img.at<uchar>(i, j + 1) == 0));
}

bool crossesEastBorder(cv::Mat img, bool checked[8], cv::Point p) {
	bool b = checked[static_cast<int>(fromTo(p, cv::Point(p.x + 1, p.y)))];
	return img.at<uchar>(p) != 0 && (p.x == img.cols - 1 || b);
}