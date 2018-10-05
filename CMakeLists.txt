cmake_minimum_required(VERSION 3.8)

PROJECT(CREPE LANGUAGES CXX CUDA)

ADD_DEFINITIONS(-DUNICODE)
ADD_DEFINITIONS(-D_UNICODE)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules)
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT CREPE)

### Subdirectories ####
include_directories(${CMAKE_SOURCE_DIR}/3rdparty)

### CUDA ###
SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_50,code=sm_50")
SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_52,code=sm_52")
SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_60,code=sm_60")
SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_61,code=sm_61")
SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_70,code=sm_70")
SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_72,code=sm_72")
SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_75,code=sm_75")

### OPENCV ###
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

### CMAKE ###
# Instruct CMake to run moc automatically when needed
set(CMAKE_AUTOMOC ON)
# Create code from a list of Qt designer ui files
set(CMAKE_AUTOUIC ON)
find_package(Qt5 COMPONENTS Core Widgets Gui REQUIRED)

# Generate code from ui files
qt5_wrap_ui(UI_HEADERS setup_window.ui)

include_directories(${OpenCV_INCLUDE_DIRS})

set (dependencies ${CUDA_LIBRARIES} ${OpenCV_LIBS} Qt5::Core Qt5::Widgets Qt5::Gui)


file(GLOB crepe_src src/*.cc)
file(GLOB crepe_hdr src/*.hh src/*.cuh)
file(GLOB cuda_src src/*.cu)
		
add_executable(CREPE ${crepe_src} ${crepe_hdr} ${cuda_src} ${UI_HEADERS})

target_link_libraries(CREPE ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${OpenCV_LIBS} Qt5::Core Qt5::Widgets Qt5::Gui)





