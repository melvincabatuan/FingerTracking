#ifndef CMT_H

#define CMT_H

#include "common.h"
#include "Consensus.h"
#include "Fusion.h"
#include "Matcher.h"
#include "Tracker.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

using cv::FeatureDetector;
using cv::DescriptorExtractor;
using cv::polylines;
using cv::Point2f;
using cv::Ptr;
using cv::Rect;
using cv::RotatedRect;
using cv::Scalar;
using cv::Size2f;



namespace cmt
{

class CMT
{
public:
    CMT() : str_detector("FAST"), /* str_descriptor("ORB") */ str_descriptor("BRISK")  {};
    // void initialize(const Mat im_gray, const Rect rect);
    bool initialize(const Mat im_gray, long topLeftx, long topLefty, long width, long height);
    bool processFrame(const Mat im_gray, Mat &im_rgba);    
    bool isInitialized();    

    Fusion fusion;
    Matcher matcher;
    Tracker tracker;
    Consensus consensus;

    string str_detector;
    string str_descriptor;

    vector<Point2f> points_active; //public for visualization purposes
    RotatedRect bb_rot;
    
    Point2f voted_center;          //public for display

private:
    Ptr<FeatureDetector> detector;
    Ptr<DescriptorExtractor> descriptor;
    
    Size2f size_initial;
    
    vector<int> classes_active;

    float theta;
    
    bool isInit;  // ~ mkc
    
    bool isProcessed;  // ~ mkc

    Mat im_prev;
};

} /* namespace CMT */

#endif /* end of include guard: CMT_H */
