cmake_minimum_required(VERSION 3.10)
project(yolofast_license_plate)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)



# 指定 Tesseract 和 Leptonica 頭文件路徑

include_directories(${OpenCV_INCLUDE_DIRS})

# 添加目標
add_executable("yolofast_license_plate" main.cpp)

target_link_libraries("yolofast_license_plate" PRIVATE
    ${OpenCV_LIBS}
    pthread
)