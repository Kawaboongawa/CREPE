#include "gui_gl_window.hh"

#define DEFAULT_GLWIDGET_SIZE 600

namespace gui
{
	GuiGLWindow::GuiGLWindow(
		const unsigned int width,
		const unsigned int height,
		void * frame,
		QWidget* parent)
		: QMainWindow(parent)
		, gl_widget_(nullptr)
	{
		ui.setupUi(this);
		this->setWindowIcon(QIcon("icon1.ico"));

		this->resize(QSize(width, height));
		this->show();

		gl_widget_.reset(new GLWidget(frame, width, height, this));
		gl_widget_->show();
	}

	GuiGLWindow::~GuiGLWindow()
	{
		gl_widget_.reset(nullptr);
	}
}