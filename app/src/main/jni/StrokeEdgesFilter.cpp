//
// Created by cobalt on 1/8/16.
//

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "StrokeEdgesFilter.h"

using namespace mhealth;

StrokeEdgesFilter::StrokeEdgesFilter() {

    size_t KERNEL_SIZE = 5;
   // int8_t kernel[][KERNEL_SIZE] = {{0, 0, 1, 0, 0},{0, 1, 2,   1, 0},{1, 2, -16, 2, 1}, {0, 1, 2,   1, 0},{0, 0, 1,   0, 0}};
   cv::Mat kernel = (cv::Mat_<char>(KERNEL_SIZE, KERNEL_SIZE) << 0, 0, 1, 0, 0, 0, 1, 2, 1, 0 , 1, 2, -16, 2, 1, 0, 1, 2, 1, 0, 0, 0, 1,   0, 0);

    /* Create data for mKernel */
    mKernel.create(KERNEL_SIZE, KERNEL_SIZE, CV_8S);

    /* Assign Kernel */
    for (size_t i = 0; i < KERNEL_SIZE; ++i) {
        for (size_t j = 0; j < KERNEL_SIZE; ++j) {
            mKernel.at<char>(i, j) = kernel.at<char>(i, j);
        }
    }
}


void StrokeEdgesFilter::apply(cv::Mat &src, cv::Mat &dst) {
    cv::filter2D(src, mEdges, -1, mKernel);
    cv::bitwise_not(mEdges, mEdges);
    cv::multiply(src, mEdges, dst, 1.0 / 255.0);
}
