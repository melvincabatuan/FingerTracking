//
//  Camshift.h
//  Tracking
//
//  Created by FloodSurge on 8/5/15.
//  Copyright (c) 2015 FloodSurge. All rights reserved.
//

#ifndef __Tracking__Camshift__
#define __Tracking__Camshift__

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>

#include <vector>

using std::vector;
using namespace cv;

class Camshift {
    
    
public:
    Camshift(void);
    ~Camshift(void);
    bool initialize(const Mat image, long topLeftx, long topLefty, long width, long height);
    bool processFrame(cv::Mat im_src, Mat &im_rgba);
    bool isInitialized(void); 
    RotatedRect objectBox;
    
private:
    Mat frame, hsv, hue, mask, hist, histimg, backproj;
    cv::Rect selection;
    int hsize = 16;
    bool isInit;  // ~ mkc    
    bool isProcessed;  // ~ mkc    
};

#endif /* defined(__Tracking__Camshift__) */
