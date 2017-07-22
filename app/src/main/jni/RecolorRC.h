//
// Created by cobalt on 1/4/16.
//

#ifndef MHEALTH_RECOLORRCFILTER_H
#define MHEALTH_RECOLORRCFILTER_H

#include <opencv2/core/mat.hpp>

namespace mhealth {

    class RecolorRCFilter {
    public:
        void apply(cv::Mat &src, cv::Mat &dst);

    private:
        cv::Mat mChannels[4];
    };

} // namespace mhealth


#endif //MHEALTH_RECOLORRCFILTER_H
