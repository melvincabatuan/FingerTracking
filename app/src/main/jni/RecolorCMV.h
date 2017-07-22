//
// Created by cobalt on 1/5/16.
//

#ifndef MHEALTH_RECOLORCMVFILTER_H
#define MHEALTH_RECOLORCMVFILTER_H

#include <opencv2/core/mat.hpp>

namespace mhealth {

    class RecolorCMVFilter
    {
    public:
        void apply(cv::Mat &src, cv::Mat &dst);

    private:
        cv::Mat mChannels[4];
    };

} // namespace mhealth

#endif //MHEALTH_RECOLORCMVFILTER_H
