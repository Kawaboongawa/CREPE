#pragma once

#include <QDialog>
#include <QMessageBox>
#include <QLineEdit>
#include <QFileDialog>
#include <memory>

#include "ui_setup_window.h"


class SetupWindow : public QDialog
{
	Q_OBJECT


public:
		
	enum input_kind
	{
		CAMERA,
		VIDEO,
		PHOTO
	};

signals:
	void close();


public slots :
	void launch();
	void browse_path();


public:
	SetupWindow(QWidget *parent = 0);
	~SetupWindow();
	input_kind get_kind() { return kind_; }
	std::string get_path() { return path_; }


private:
	input_kind kind_;
	std::string path_;
};
