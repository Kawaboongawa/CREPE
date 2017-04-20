#include "database_handler.hh"

namespace crepe
{
	Database::Database(filter::FilterHandler filter)
		: chessmans_(std::vector<Chessman>())
		, piece_name_(std::vector<std::string>
	{"king", "queen", "knight", "bishop", "rook", "pawn"})
		, filter_(filter)
	{
		chessmans_.push_back(Chessman(chesspiece::KING));
		chessmans_.push_back(Chessman(chesspiece::QUEEN));
		chessmans_.push_back(Chessman(chesspiece::KNIGHT));
		chessmans_.push_back(Chessman(chesspiece::BISHOP));
		chessmans_.push_back(Chessman(chesspiece::ROOK));
		chessmans_.push_back(Chessman(chesspiece::PAWN));
		init_database();
	}

	Database::~Database()
	{}


	int findMaxIndex(const std::vector<std::vector<cv::Point>>& arr)
	{
		int index = 0;
		for (int i = 0; i < arr.size(); i++)
		{
			if (arr[i].size() > arr[index].size())
				index = i;
		}
		return index;
	}

	FourierDescriptor& Database::get_decriptor(const std::string& path)
	{

		std::vector<std::vector<cv::Point>> contours;
		std::vector<cv::Vec4i> hierarchy;
		cv::Mat src = cv::imread(path);
		cv::Mat canny;
		cv::imshow("original photo", src);
		GpuMat srcdev;
		srcdev.upload(src);
		GpuMat res = filter_.compute_edges(srcdev);
		res.download(canny);
		cv::findContours(canny, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
		int index = findMaxIndex(contours);
		int size = contours[index].size();
		//FIXME: THIS IS UGLY and need to be changed
		//////////////////
		ushort2* edges = (ushort2*)malloc(size * sizeof(ushort2));
		for (int i = 0; i < size; i++)
		{
			edges[i].x = contours[index][i].x;
			edges[i].y = contours[index][i].y;
		}
		//////////////////
		FourierDescriptor fd(edges, size);
		free(edges);
		fd.compute_descriptors();
		return fd;
	}


	inline std::wstring stringTowstring(const std::string &s)
	{
		return std::wstring(s.begin(), s.end());
	}

	inline std::string wstringTostring(const std::wstring &ws)
	{
		return std::string(ws.begin(), ws.end());
	}


	void Database::init_database()
	{
		WIN32_FIND_DATA data;
		HANDLE hFind;


		for (int i = 0; i < chessmans_.size(); i++)
		{
			std::string path = "..\\..\\database\\" + piece_name_[i] + "\\";
			std::wstring wpath = stringTowstring(path + "*");
			hFind = FindFirstFile(wpath.c_str(), &data);
			if (hFind != INVALID_HANDLE_VALUE)
			{
				do
				{
					std::string filename = wstringTostring(data.cFileName);
					if (filename == "." || filename == "..")
						continue;
					std::string fullpath = path + filename;
					FourierDescriptor fd = get_decriptor(fullpath);
					chessmans_[i].add_descriptor(fd);
				} while (FindNextFile(hFind, &data));
			}
		}
	}
}