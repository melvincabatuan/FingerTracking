#ifndef CONSENSUSMATCHINGTRACKER_H
#define CONSENSUSMATCHINGTRACKER_H

#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <cmath>

namespace mhealth {

    class ConsensusMatchingTracker {

    private:

        int descriptorLength;
        int thrOutlier;

        float thrConf;
        float thrRatio;

        bool estimateScale;
        bool estimateRotation;
        bool initialized;
        bool hasResult;

        cv::Ptr<cv::FeatureDetector> detector;
        cv::Ptr<cv::DescriptorExtractor> descriptorExtractor;
        cv::Ptr<cv::DescriptorMatcher> descriptorMatcher;

        cv::Mat selectedFeatures;

        cv::Mat featuresDatabase;

        cv::Mat im_prev;

        cv::Point2f topLeft;
        cv::Point2f topRight;
        cv::Point2f bottomRight;
        cv::Point2f bottomLeft;

        cv::Point2f centerToTopLeft;
        cv::Point2f centerToTopRight;
        cv::Point2f centerToBottomRight;
        cv::Point2f centerToBottomLeft;

        std::vector<int> selectedClasses;
        std::vector<int> classesDatabase;

        std::vector<std::vector<float> > squareForm;
        std::vector<std::vector<float> > angles;

        std::vector<cv::Point2f> springs;

        std::vector<std::pair<cv::KeyPoint, int> > activeKeypoints;
        std::vector<std::pair<cv::KeyPoint, int> > trackedKeypoints;


        std::vector<std::pair<cv::KeyPoint, int> > outliers;
        
        std::vector<cv::Point2f> votes;

        size_t initialKeypointSize;

        cv::Rect_<float> boundingbox;
        
        cv::Size2f roi_size_initial;


    public:

        ConsensusMatchingTracker();

        bool isInitialized();

        void initialize(cv::Mat im_gray0, long topLeftx, long topLefty, long width, long height);

        void estimate(const std::vector<std::pair<cv::KeyPoint, int> > &keypointsIN,
                      cv::Point2f &center, float &scaleEstimate, float &medRot,
                      std::vector<std::pair<cv::KeyPoint, int> > &keypoints);

        void processFrame(cv::Mat &im_gray, cv::Mat &im_rgba);

        void track(cv::Mat im_prev, cv::Mat im_gray,
                   const std::vector<std::pair<cv::KeyPoint, int> > &keypointsIN,
                   std::vector<std::pair<cv::KeyPoint, int> > &keypointsTracked,
                   std::vector<unsigned char> &status, int THR_FB = 20); // THR_FB - delta parameter

        cv::Point2f rotate(cv::Point2f p, float rad);
    };

/*
    struct Cluster {
        int first, second; //cluster id
        float dist;
        int num;
    };
*/

}  /// mhealth namespace


#endif // CONSENSUSMATCHINGTRACKER_H
