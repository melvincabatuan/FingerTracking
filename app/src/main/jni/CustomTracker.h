#pragma once

#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#ifndef CUSTOM_TRACKER_H
#define CUSTOM_TRACKER_H

class CustomTracker
{
	public:
    	CustomTracker(); 
    	~CustomTracker();
    	CustomTracker(std::string  tracker_algorithm);
    	bool initialize(cv::Mat &im_gray, long topLeftx, long topLefty, long width, long height);   
		bool isInitialized();
		bool processFrame(cv::Mat &im_gray, cv::Mat &im_rgba);
		
	private:
		cv::Ptr<cv::Tracker> tracker;
		bool isInit;
};

#endif // CUSTOM_TRACKER_H
