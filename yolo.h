#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

struct Net_config
{
	float confThreshold; // Confidence threshold
	float nmsThreshold;  // Non-maximum suppression threshold
	int inpWidth;  // Width of network's input image
	int inpHeight; // Height of network's input image
	string classesFile;
	string modelConfiguration;
	string modelWeights;
	string netname;
};

class YOLO
{
	public:
		YOLO(Net_config config);
		void detect(Mat& frame);
	private:
		float confThreshold;
		float nmsThreshold;
		int inpWidth;
		int inpHeight;
		char netname[20];
		vector<string> classes;
		Net net;
		void postprocess(Mat& frame, const vector<Mat>& outs);
		void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);
};

Net_config yolo_nets[1] = {
	{0.5, 0.4, 320, 320,"license_plate.names", "yolo-fastest/yolo-fastest-1.1.cfg", "yolo-fastest/yolo-fastest-1_final.weights", "yolo-fastest"}
};