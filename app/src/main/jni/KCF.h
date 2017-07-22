#pragma once

#include <opencv2/core.hpp>
#include "kcf_src/kcftracker.hpp"
#include "common.h"

#ifndef KCF_TRACKER_H
#define KCF_TRACKER_H

class KCF
{
	public:
    	KCF(); 
    	~KCF();
    	bool initialize(cv::Mat &im_gray, long topLeftx, long topLefty, long width, long height);   
		bool isInitialized();
		bool processFrame(cv::Mat &im_gray, cv::Mat &im_rgba);
		
	private:
	    //bool HOG = true;
		//bool FIXEDWINDOW = true;
		//bool MULTISCALE = true;
		//bool LAB = true;
		KCFTracker tracker;
		int detected_ROI[4] = {0, 0, 0, 0};
		bool isInit;
		Queue Trace;
};

#endif // CUSTOM_TRACKER_H
