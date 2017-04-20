#pragma once

#include <windows.h>
#include <tchar.h>
#include <stdio.h>

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

	private :

		FourierDescriptor& get_decriptor(const std::string& path);

		

	private:

		std::vector<Chessman> chessmans_;

		std::vector<std::string> piece_name_;

		filter::FilterHandler filter_;
	};
}