#pragma once
#include <QMainWindow>
#include <QResizeEvent>
#include <QShortcut>

#include <memory>

#include "gui_gl_widget.hh"
#include "ui_gl_window.h"

/* Forward declarations. */
namespace holovibes
{
	class Holovibes;
	class Queue;
}

namespace gui
{
	/*! \brief QMainWindow overload used to display the real time OpenGL frame. */
	class GuiGLWindow : public QMainWindow
	{
		Q_OBJECT

	public:
		/*! \brief GuiGLWindow constructor
		**
		** \param pos initial position of the window
		** \param width width of the window in pixels
		** \param height height of the window in pixels
		** \param h holovibes object
		** \param q Queue from where to grab frames to display
		** \param parent Qt parent
		*/
		GuiGLWindow(
			const unsigned int width,
			const unsigned int height,
			void* frame,
			QWidget* parent = nullptr);

		/* \brief GuiGLWindow destructor */
		~GuiGLWindow();

		/*! \brief Call when windows is resize */
	//	void resizeEvent(QResizeEvent* e) override;

		/*! \brief Returns a reference to a GLWidget object. */
		GLWidget& get_gl_widget() const
		{
			return *gl_widget_;
		}

	private:
		Ui::GLWindow ui;

		/*! GL widget, it updates itself */
		std::unique_ptr<GLWidget> gl_widget_;


	};
}