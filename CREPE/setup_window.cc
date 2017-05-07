#include "setup_window.hh"


SetupWindow::SetupWindow(QWidget *parent) :
	QDialog(parent),
	kind_(CAMERA),
	path_("_")
{
	Ui_Dialog window;
	window.setupUi(this);
	
}

SetupWindow::~SetupWindow()
{
}

void SetupWindow::launch()
{
	QComboBox* input = findChild<QComboBox*>("inputKindComboBox");
	kind_ = static_cast<input_kind>(input->currentIndex());
	QLineEdit* input_path = findChild<QLineEdit*>("pathLineEdit");
	std::string str = input_path->text().toUtf8();
	path_ = str;
	emit close();
}

void SetupWindow::browse_path()
{
	QString filename = QFileDialog::getOpenFileName(this,
		tr("Input path"), ".");

	QLineEdit* roi_output_line_edit = findChild<QLineEdit*>("pathLineEdit");
	roi_output_line_edit->clear();
	roi_output_line_edit->insert(filename);
}


