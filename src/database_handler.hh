#pragma once

#include <windows.h>
#include <tchar.h>
#include <stdio.h>
#include <dirent/dirent.h>
#include <algorithm>

#include "chessman.hh"
#include "filter_handler.hh"


namespace crepe
{
	class Database
	{
	public : 
		Database(filter::FilterHandler filter);
		~Database();

		void init_database();

		std::string match_object(const FourierDescriptor& fd, uint ndesc);

	private :

		std::shared_ptr<FourierDescriptor> get_descriptor(const std::string& path);

	private:

		std::vector<Chessman> chessmans_;

		std::vector<std::string> piece_name_;

		filter::FilterHandler filter_;
		
	};
}