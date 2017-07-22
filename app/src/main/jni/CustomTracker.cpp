#include "CustomTracker.h"


CustomTracker::CustomTracker(){
	tracker = cv::Tracker::create( "KCF" ); // default
};
CustomTracker::~CustomTracker(){};

CustomTracker::CustomTracker(std::string  tracker_algorithm){
	tracker = cv::Tracker::create( tracker_algorithm );
}

bool CustomTracker::initialize(cv::Mat &src, long topLeftx, long topLefty, long width, long height){
	cv::Mat bgr;
	cv::cvtColor(src, bgr, CV_BGRA2BGR);
    cv::Rect2d roi = cv::Rect(topLeftx, topLefty, width, height);
	return tracker->init(bgr, roi);
}

bool CustomTracker::isInitialized(){
	return isInit;
}

bool CustomTracker::processFrame(cv::Mat &src, cv::Mat &im_rgba){
	cv::Mat bgr;
	cv::cvtColor(src, bgr, CV_BGRA2BGR);
	cv::Rect2d boundingBox;
	tracker->update( bgr, boundingBox );
	cv::rectangle( im_rgba, boundingBox, cv::Scalar( 255, 0, 0 ), 2, 1 );
}
