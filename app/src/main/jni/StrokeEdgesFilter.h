//
// Created by cobalt on 1/8/16.
//

#ifndef MHEALTH_STROKEEDGESFILTER_H
#define MHEALTH_STROKEEDGESFILTER_H


#include <opencv2/core/mat.hpp>

namespace mhealth {

    class StrokeEdgesFilter {
    public:
        StrokeEdgesFilter();

        void apply(cv::Mat &src, cv::Mat &dst);

    private:
        cv::Mat mKernel;
        cv::Mat mEdges;
    };

} // namespace mhealth


#endif //MHEALTH_STROKEEDGESFILTER_H
