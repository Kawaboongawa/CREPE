#pragma once

# include <array>
# include <iostream>

# include <QtOpenGL>

# include <cuda_gl_interop.h>
# include <cuda_runtime.h>

/* Forward declaration. */
namespace holovibes
{
	class Queue;
}

namespace gui
{

	/*! \brief OpenGL widget used to display frames contained in Queue(s).
	*
	* Users can select zone and move in display surf
	* Selected zone with mouse will emit qt signals.
	*/
	class GLWidget : public QGLWidget, protected QOpenGLFunctions
	{
		Q_OBJECT

			/*! Frame rate of the display in Hertz (Frame.s-1) */
			const unsigned int DISPLAY_FRAMERATE = 30;

	public:
		/* \brief GLWidget constructor
		**
		** Build the widget and start a display QTimer.
		**
		** \param void* frame containing the displayed screen in RGB format
		** \param width widget's width
		** \param height widget's height
		** \param parent Qt parent (should be a GUIGlWindow)
		*/
		GLWidget(
			void* frame,
			const unsigned int width,
			const unsigned int height,
			QWidget* parent = 0);

		virtual ~GLWidget();

		/*! \brief This property holds the recommended minimum size for the widget. */
		QSize minimumSizeHint() const override;

		/*! \brief This property holds the recommended size for the widget. */
		QSize sizeHint() const override;


		public slots:
		void resizeFromWindow(const int width, const int height);


	protected:
		/* \brief Initialize all OpenGL components needed */
		void initializeGL() override;

		/*! \brief Called whenever the OpenGL widget is resized */
		void resizeGL(int width, int height) override;

		/*! \brief Paint the scene and the selection zone(s) according to selection_mode_.
		**
		** The image is painted directly from the GPU, avoiding several
		** back and forths memory transfers.
		** This method uses the NVI idiom with set_texture_format by wrapping it
		** with common boilerplate code.
		*/
		void paintGL() override;

	protected:
		QWidget* parent_;
		
		void*     frame_;

		/*! \brief QTimer used to refresh the OpenGL widget */
		QTimer timer_;

		/*! \{ \name OpenGl graphique buffer */
		GLuint  buffer_;
		struct cudaGraphicsResource*  cuda_buffer_;
		cudaStream_t cuda_stream_; //!< Drawing operates on a individual stream.
								   /*! \} */

								   /*! \{ \name Selection */
								   /*! \brief User is currently select zone ? */
		/*! \{ \name Window size hints */
		const unsigned int width_;
		const unsigned int height_;
		/*! \} */

	private:

		/*! \brief Check glError and print then
		*
		* Use only in debug mode, glGetError is slow and should be avoided
		*/
		void gl_error_checking();
	};
}