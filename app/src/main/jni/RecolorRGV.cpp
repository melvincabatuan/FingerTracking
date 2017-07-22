//
// Created by cobalt on 1/5/16.
//

#include <opencv2/core.hpp>
#include "RecolorRGV.h"

using namespace mhealth;

void RecolorRGVFilter::apply(cv::Mat &src, cv::Mat &dst)
{
    cv::split(src, mChannels);

    cv::Mat r = mChannels[0];
    cv::Mat g = mChannels[1];
    cv::Mat b = mChannels[2];

    // dst.b = min(dst.r, dst.g, dst.b)
    cv::min(b, r, b);
    cv::min(b, g, b);

    cv::merge(mChannels, 4, dst);
}