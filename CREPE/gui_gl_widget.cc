#include "gui_gl_widget.hh"


namespace gui
{
	GLWidget::GLWidget(
		void* frame,
		const unsigned int width,
		const unsigned int height,
		QWidget *parent)
		: QGLWidget(QGLFormat(QGL::SampleBuffers), parent)
		, QOpenGLFunctions()
		, timer_(this)
		, width_(width)
		, height_(height)
		, frame_(frame)
		, buffer_(0)
		, cuda_buffer_(nullptr)
		, parent_(parent)
	{
		this->setObjectName("GLWidget");
		this->resize(QSize(width, height));
		connect(&timer_, SIGNAL(timeout()), this, SLOT(update()));
		timer_.start(1000 / DISPLAY_FRAMERATE);

		// Create a new computation stream on the graphics card.
		if (cudaStreamCreate(&cuda_stream_) != cudaSuccess)
			cuda_stream_ = 0; // Use default stream as a fallback

		setMouseTracking(true);
	}

	GLWidget::~GLWidget()
	{
		/* Unregister buffer for access by CUDA. */
		cudaGraphicsUnregisterResource(cuda_buffer_);
		/* Free the associated computation stream. */
		cudaStreamDestroy(cuda_stream_);
		/* Destroy buffer name. */
		glDeleteBuffers(1, &buffer_);
		glDisable(GL_TEXTURE_2D);
	}


	QSize GLWidget::minimumSizeHint() const
	{
		return QSize(width_, height_);
	}

	QSize GLWidget::sizeHint() const
	{
		return QSize(width_, height_);
	}

	void GLWidget::initializeGL()
	{
		initializeOpenGLFunctions();
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glEnable(GL_TEXTURE_2D);

		/* Generate buffer name. */
		glGenBuffers(1, &buffer_);

		/* Bind a named buffer object to the target GL_TEXTURE_BUFFER. */
		glBindBuffer(GL_TEXTURE_BUFFER, buffer_);

		//frame_desc_.frame_size();
		unsigned int size = width_ * height_ * 3;


		/* Creates and initialize a buffer object's data store. */
		glBufferData(GL_TEXTURE_BUFFER, size, nullptr, GL_DYNAMIC_DRAW);
		/* Unbind any buffer of GL_TEXTURE_BUFFER target. */
		glBindBuffer(GL_TEXTURE_BUFFER, 0);
		/* Register buffer name to CUDA. */
		cudaGraphicsGLRegisterBuffer(
			&cuda_buffer_,
			buffer_,
			cudaGraphicsMapFlags::cudaGraphicsMapFlagsNone);

		glViewport(0, 0, width_, height_);
	}

	void GLWidget::resizeGL(int width, int height)
	{
		glViewport(0, 0, width, height);
	}

	void GLWidget::paintGL()
	{
		glEnable(GL_TEXTURE_2D);
		glClear(GL_COLOR_BUFFER_BIT);

		/* Map the buffer for access by CUDA. */
		cudaGraphicsMapResources(1, &cuda_buffer_, cuda_stream_);
		size_t	buffer_size;
		void*	buffer_ptr;
		cudaGraphicsResourceGetMappedPointer(&buffer_ptr, &buffer_size, cuda_buffer_);
		/* CUDA memcpy of the frame to opengl buffer. */
		cudaMemcpy(buffer_ptr, frame_, buffer_size, cudaMemcpyKind::cudaMemcpyDeviceToDevice);

		/* Unmap the buffer for access by CUDA. */
		cudaGraphicsUnmapResources(1, &cuda_buffer_, cuda_stream_);

		/* Bind the buffer object to the target GL_PIXEL_UNPACK_BUFFER.
		* This affects glTexImage2D command. */
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer_);

		//if (frame_desc_.endianness == camera::BIG_ENDIAN)
		//	glPixelStorei(GL_UNPACK_SWAP_BYTES, GL_TRUE);
		//else
		//glPixelStorei(GL_UNPACK_SWAP_BYTES, GL_FALSE);
		
		// in case debbugging is needed
		
		/*void * res = malloc(50);
		cudaMemcpy(res, buffer_ptr, 50, cudaMemcpyDeviceToHost);
		unsigned char * resc = (unsigned char *)res;
		unsigned char r = resc[12];
		unsigned char g = resc[13];
		unsigned char b = resc[14];
		free(res);*/


		auto depth = GL_UNSIGNED_BYTE;
		auto kind = GL_RGB;
		//auto kind = GL_RED;
		//auto depth = GL_UNSIGNED_BYTE;

		glTexImage2D(GL_TEXTURE_2D, 0, kind, width_, height_, 0, kind, depth, nullptr);
		glGenerateMipmap(GL_TEXTURE_2D);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		glBegin(GL_QUADS);
		glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
		glTexCoord2d(0.0, 0.0); glVertex2d(-1.0, +1.0);
		glTexCoord2d(1.0, 0.0); glVertex2d(+1.0, +1.0);
		glTexCoord2d(1.0, 1.0); glVertex2d(+1.0, -1.0);
		glTexCoord2d(0.0, 1.0); glVertex2d(-1.0, -1.0);
		glEnd();

		glDisable(GL_TEXTURE_2D);
		gl_error_checking();
	}

	void GLWidget::mousePressEvent(QMouseEvent* e)
	{
	}

	void GLWidget::mouseMoveEvent(QMouseEvent* e)
	{
		
	}

	void GLWidget::mouseReleaseEvent(QMouseEvent* e)
	{
	}

	void GLWidget::gl_error_checking()
	{
		// Sometimes this will occur when opengl is having some
		// trouble, and this will cause glGetString to return NULL.
		// That's why we need to check it, in order to avoid crashes.
		GLenum error = glGetError();
		auto err_string = glGetString(error);
		if (error != GL_NO_ERROR && err_string)
			std::cerr << "[GL] " << err_string << std::endl;
	}

	void GLWidget::resizeFromWindow(const int width, const int height)
	{
		resizeGL(width, height);
		resize(QSize(width, height));
	}
}