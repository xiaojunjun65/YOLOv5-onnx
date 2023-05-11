#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
using namespace cv;
using namespace dnn;
using namespace std;

void imshow(std::string name, const cv::Mat& cv_src)
{
	cv::namedWindow(name, 0);
	int max_rows = 320;
	int max_cols = 320;
	if (cv_src.rows >= cv_src.cols && cv_src.rows > max_rows)
	{
		cv::resizeWindow(name, cv::Size(cv_src.cols * max_rows / cv_src.rows, max_rows));
	}
	else if (cv_src.cols >= cv_src.rows && cv_src.cols > max_cols)
	{
		cv::resizeWindow(name, cv::Size(max_cols, cv_src.rows * max_cols / cv_src.cols));
	}
	cv::imshow(name, cv_src);
}

inline float sigmoid(float x)
{
	return 1.f / (1.f + exp(-x));
}

void sliceAndConcat(cv::Mat& img, cv::Mat* input)
{
	const float* srcData = img.ptr<float>();
	float* dstData = input->ptr<float>();
	using Vec12f = cv::Vec<float, 12>;
	for (int i = 0; i < input->size[2]; i++)
	{
		for (int j = 0; j < input->size[3]; j++)
		{
			for (int k = 0; k < 3; ++k)
			{
				dstData[k * input->size[2] * input->size[3] + i * input->size[3] + j] =
					srcData[k * img.size[2] * img.size[3] + 2 * i * img.size[3] + 2 * j];
			}
			for (int k = 0; k < 3; ++k)
			{
				dstData[(3 + k) * input->size[2] * input->size[3] + i * input->size[3] + j] =
					srcData[k * img.size[2] * img.size[3] + (2 * i + 1) * img.size[3] + 2 * j];
			}
			for (int k = 0; k < 3; ++k)
			{
				dstData[(6 + k) * input->size[2] * input->size[3] + i * input->size[3] + j] =
					srcData[k * img.size[2] * img.size[3] + 2 * i * img.size[3] + 2 * j + 1];
			}
			for (int k = 0; k < 3; ++k)
			{
				dstData[(9 + k) * input->size[2] * input->size[3] + i * input->size[3] + j] =
					srcData[k * img.size[2] * img.size[3] + (2 * i + 1) * img.size[3] + 2 * j + 1];
			}
		}
	}
}

std::vector<cv::String> getOutputNames(const cv::dnn::Net& net)
{
	static std::vector<cv::String> names;
	if (names.empty())
	{
		std::vector<int> outLayers = net.getUnconnectedOutLayers();
		std::vector<cv::String> layersNames = net.getLayerNames();
		names.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); i++)
		{
			names[i] = layersNames[outLayers[i] - 1];
		}
	}
	return names;
}

void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame,
	const std::vector<std::string>& classes)
{
	cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 255, 0), 3);
	std::string label = cv::format("%.2f", conf);
	if (!classes.empty()) {
		CV_Assert(classId < (int)classes.size());
		label = classes[classId] + ": " + label;
	}
	int baseLine;
	cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = std::max(top, labelSize.height);
	cv::rectangle(frame, cv::Point(left, top - round(1.5 * labelSize.height)), cv::Point(left + round(1.5 * labelSize.width), top + baseLine), cv::Scalar(0, 255, 0), cv::FILLED);
	cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(), 2);
}

void postprocess(cv::Mat& cv_src, std::vector<cv::Mat>& outs, const std::vector<std::string>& classes, int net_size)
{
	float confThreshold = 0.1f;
	float nmsThreshold = 0.1f;
	std::vector<int> classIds;
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;
	int strides[] = { 8, 16, 32 };
	std::vector<std::vector<int> > anchors =
	{
		{ 10,13, 16,30, 33,23 },
		{ 30,61, 62,45, 59,119 },
		{ 116,90, 156,198, 373,326 }
	};
	for (size_t k = 0; k < outs.size(); k++)
	{
		float* data = outs[k].ptr<float>();
		int stride = strides[k];
		int num_classes = outs[k].size[4] - 5;
		for (int i = 0; i < outs[k].size[2]; i++)
		{
			for (int j = 0; j < outs[k].size[3]; j++)
			{
				for (int a = 0; a < outs[k].size[1]; ++a)
				{
					float* record = data + a * outs[k].size[2] * outs[k].size[3] * outs[k].size[4] +
						i * outs[k].size[3] * outs[k].size[4] + j * outs[k].size[4];
					float* cls_ptr = record + 5;
					for (int cls = 0; cls < num_classes; cls++)
					{
						float score = sigmoid(cls_ptr[cls]) * sigmoid(record[4]);
						if (score > confThreshold)
						{
							float cx = (sigmoid(record[0]) * 2.f - 0.5f + (float)j) * (float)stride;
							float cy = (sigmoid(record[1]) * 2.f - 0.5f + (float)i) * (float)stride;
							float w = pow(sigmoid(record[2]) * 2.f, 2) * anchors[k][2 * a];
							float h = pow(sigmoid(record[3]) * 2.f, 2) * anchors[k][2 * a + 1];
							float x1 = std::max(0, std::min(cv_src.cols, int((cx - w / 2.f) * (float)cv_src.cols / (float)net_size)));
							float y1 = std::max(0, std::min(cv_src.rows, int((cy - h / 2.f) * (float)cv_src.rows / (float)net_size)));
							float x2 = std::max(0, std::min(cv_src.cols, int((cx + w / 2.f) * (float)cv_src.cols / (float)net_size)));
							float y2 = std::max(0, std::min(cv_src.rows, int((cy + h / 2.f) * (float)cv_src.rows / (float)net_size)));
							classIds.push_back(cls);
							confidences.push_back(score);
							boxes.push_back(cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)));
						}
					}
				}
			}
		}
	}
	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); i++)
	{
		int idx = indices[i];
		cv::Rect box = boxes[idx];
		drawPred(classIds[idx], confidences[idx], box.x, box.y,
			box.x + box.width, box.y + box.height, cv_src, classes);
	}
}

int main(int argc, char* argv[])
{
	string path = "test.png";
	cout << path;
	std::vector<std::string> filenames;
	cv::glob(path, filenames, false);

	for (auto name : filenames)
	{
		Mat cv_src = cv::imread(name);
		if (cv_src.empty())
		{
			continue;
		}
	
		std::vector<std::string> class_names{ "1","2","3","4" ,"5","6" };

		int net_size = 640;
		auto t0 = cv::getTickCount();
		Net net = readNet("best_wuli.onnx");
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
		Mat blob;
		blobFromImage(cv_src, blob, 1 / 255.0, cv::Size(640, 640), Scalar(0, 0, 0), true, false);
		net.setInput(blob);
		vector<Mat> netOutputImg;
		net.forward(netOutputImg, net.getUnconnectedOutLayersNames());
		postprocess(cv_src, netOutputImg, class_names, net_size);
		auto t1 = cv::getTickCount();
		std::cout << "elapsed time: " << (t1 - t0) * 1000.0 / cv::getTickFrequency() << "ms" << std::endl;

		imshow("img", cv_src);
		cv::waitKey();
	}

	return 0;
}
