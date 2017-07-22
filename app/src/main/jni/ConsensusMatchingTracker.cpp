#include "ConsensusMatchingTracker.h"
#include "common.h"


using namespace mhealth;


ConsensusMatchingTracker::ConsensusMatchingTracker() {

    thrOutlier = 20;
    thrConf = 0.75;
    //thrConf = 0.25; // IMPACT ? unnoticable
    thrRatio = 0.8;
    descriptorLength = 512;
    estimateScale = true;
    estimateRotation = true;
    initialized = false;
    initialKeypointSize = 0;
    detector = cv::ORB::create(); // descriptor and extractor at the same time
    descriptorExtractor = cv::ORB::create();
    //detector = cv::FastFeatureDetector::create(); // used in original CMT
    //descriptorExtractor = cv::BRISK::create(); 
    //detector = cv::AKAZE::create();    
    //descriptorExtractor = cv::AKAZE::create();   
    descriptorMatcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
}


void ConsensusMatchingTracker::initialize(cv::Mat im_gray0, long topLeftx, long topLefty, long width, long height) {

    /* Initialize the selected region-of-interest */
    cv::Point2f topleft(topLeftx, topLefty);
    cv::Rect roi = cv::Rect(topleft.x, topleft.y, width, height);
    
    /* Initial roi size */
    roi_size_initial = roi.size();
    
    /* Store initial image */
    im_prev = im_gray0.clone();

    /* Get initial keypoints in whole image */
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(im_gray0, keypoints);

    /* Remember keypoints that are in the roi as selected keypoints */
    std::vector<cv::KeyPoint> selected_keypoints;
    std::vector<cv::KeyPoint> background_keypoints;

    /* Segregate keypoints into inside roi points (in) and outside roi points (out) */
    for (size_t i = 0; i < keypoints.size(); i++) {
        if (roi.contains(keypoints[i].pt)) {
            selected_keypoints.push_back(keypoints[i]);
        } else {
            background_keypoints.push_back(keypoints[i]);
        }
    }

    /* Extract the features */
    descriptorExtractor->compute(im_gray0, selected_keypoints, selectedFeatures);

    /* Store initial keypoints */
    initialKeypointSize = selected_keypoints.size();

	/* Cancel initialization if no keypoints are detected */
    if (initialKeypointSize == 0) {
        initialized = false;
    }

    /* Extract the descriptors for the background as well */
    cv::Mat background_features;
    descriptorExtractor->compute(im_gray0, background_keypoints, background_features);

    /* Assign each keypoint an index starting from 1..initialKeypointSize, background is 0 */
    selectedClasses.reserve(selected_keypoints.size()+1);
    for (size_t i = 1; i <= selected_keypoints.size(); i++){
        selectedClasses.push_back(i);
    }

	/* Assign zero index for background keypoints */
    std::vector<int> backgroundClasses;
    backgroundClasses.reserve(background_keypoints.size());
    for (size_t i = 0; i < background_keypoints.size(); i++){
        backgroundClasses.push_back(0);
    }

    /* Stack background features and selected features into database */
    featuresDatabase = cv::Mat(background_features.rows + selectedFeatures.rows,
                               std::max(background_features.cols, selectedFeatures.cols),
                               background_features.type());

    if (background_features.cols > 0){
        background_features.copyTo(featuresDatabase(
                cv::Rect(0, 0, background_features.cols, background_features.rows)));
    }

    if (selectedFeatures.cols > 0){
        selectedFeatures.copyTo(featuresDatabase(
                cv::Rect(0, background_features.rows, selectedFeatures.cols,
                         selectedFeatures.rows)));
    }

    /* Store the labels as well */
    classesDatabase = std::vector<int>();

    for (size_t i = 0; i < backgroundClasses.size(); i++){
        classesDatabase.push_back(backgroundClasses[i]);
    }

    for (size_t i = 0; i < selectedClasses.size(); i++){
        classesDatabase.push_back(selectedClasses[i]);
    }

    // Get all distances between selected keypoints in squareform and get all angles between selected keypoints
    squareForm = std::vector<std::vector<float> >();
    angles = std::vector<std::vector<float> >();
    for (size_t i = 0; i < selected_keypoints.size(); i++) {
        std::vector<float> lineSquare;
        std::vector<float> lineAngle;
        for (size_t j = 0; j < selected_keypoints.size(); j++) {
            float dx = selected_keypoints[j].pt.x - selected_keypoints[i].pt.x;
            float dy = selected_keypoints[j].pt.y - selected_keypoints[i].pt.y;
            lineSquare.push_back(sqrt(dx * dx + dy * dy));
            lineAngle.push_back(atan2(dy, dx));
        }
        squareForm.push_back(lineSquare);
        angles.push_back(lineAngle);
    }

    /* Find the center of selected keypoints */
    cv::Point2f center(0, 0);
    for (size_t i = 0; i < selected_keypoints.size(); i++){
        center += selected_keypoints[i].pt;
    }
    center *= (1.0 / selected_keypoints.size());

    /* Rectangle coordinates relative to the center */
    cv::Point2f bottomright = cv::Point2f(topleft.x + width, topleft.y + height);
    centerToTopLeft = topleft - center;
    centerToTopRight = cv::Point2f(bottomright.x, topleft.y) - center;
    centerToBottomRight = bottomright - center;
    centerToBottomLeft = cv::Point2f(topleft.x, bottomright.y) - center;

    /* Calculate springs of each keypoint or the normalized points */
    springs = std::vector<cv::Point2f>();
    for (size_t i = 0; i < selected_keypoints.size(); i++){
        springs.push_back(selected_keypoints[i].pt - center);
    }

    /* Make keypoints 'active' keypoints */
    activeKeypoints = std::vector<std::pair<cv::KeyPoint, int> >();
    for (size_t i = 0; i < selected_keypoints.size(); i++){
        activeKeypoints.push_back(std::make_pair(selected_keypoints[i], selectedClasses[i]));
    }

    // Already initialized!
    initialized = true;

} //END_INITIALIZE




/// Track keypoint from previous frame (im_prev) to current (im_gray)
void ConsensusMatchingTracker::track(cv::Mat im_prev, cv::Mat im_gray,
                const std::vector<std::pair<cv::KeyPoint, int> > &keypointsIN,
                std::vector<std::pair<cv::KeyPoint, int> > &keypointsTracked,
                std::vector<unsigned char> &status,
                int THR_FB)// THR_FB - Forward-Backward Error Threshold
{
    //Status of tracked keypoint - True means successfully tracked
    status = std::vector<unsigned char>();
    //for(int i = 0; i < keypointsIN.size(); i++)
    //  status.push_back(false);
    //If at least one keypoint is active
    if (keypointsIN.size() > 0) {
        std::vector<cv::Point2f> pts;
        std::vector<cv::Point2f> pts_back;
        std::vector<cv::Point2f> nextPts;
        std::vector<unsigned char> status_back;
        std::vector<float> err;
        std::vector<float> err_back;
        std::vector<float> fb_err;

        for (size_t i = 0; i < keypointsIN.size(); i++)
            pts.push_back(cv::Point2f(keypointsIN[i].first.pt.x, keypointsIN[i].first.pt.y));


        //Calculate forward optical flow for prev_location
        cv::calcOpticalFlowPyrLK(im_prev, im_gray, pts, nextPts, status, err);

        //Calculate backward optical flow for prev_location
        cv::calcOpticalFlowPyrLK(im_gray, im_prev, nextPts, pts_back, status_back, err_back);

        //Calculate forward-backward error (fb_err)
        for (size_t i = 0; i < pts.size(); i++) {
            cv::Point2f v = pts_back[i] - pts[i];
            fb_err.push_back(sqrt(v.dot(v)));
        }

        //Set status depending on fb_err and lk error
        for (size_t i = 0; i < status.size(); i++)
            status[i] = (fb_err[i] <= THR_FB) & status[i];


        //Remember tracked keypoints depending on fb_err and lk error

        keypointsTracked = std::vector<std::pair<cv::KeyPoint, int> >();

        for (size_t i = 0; i < pts.size(); i++) {
            std::pair<cv::KeyPoint, int> p = keypointsIN[i];
            if (status[i]) {
                p.first.pt = nextPts[i];
                keypointsTracked.push_back(p);
            }
        }
    }
    else keypointsTracked = std::vector<std::pair<cv::KeyPoint, int> >();
}


//Rotates a point by angle rad
cv::Point2f ConsensusMatchingTracker::rotate(cv::Point2f p, float rad) {
    if (rad == 0)
        return p;
    float s = sin(rad);
    float c = cos(rad);
    return cv::Point2f(c * p.x - s * p.y, s * p.x + c * p.y);
}


bool ConsensusMatchingTracker::isInitialized() {
    return initialized;
}

/*
typedef std::pair<int, int> PairInt;

typedef std::pair<float, int> PairFloat;

template<typename T>
bool comparatorPair(const std::pair<T, int> &l, const std::pair<T, int> &r) {
    return l.first < r.first;
}

template<typename T>
bool comparatorPairDesc(const std::pair<T, int> &l, const std::pair<T, int> &r) {
    return l.first > r.first;
}

template<typename T>
T sign(T t) {
    if (t == 0)
        return T(0);
    else
        return (t < 0) ? T(-1) : T(1);
}

template<typename T>
T median(std::vector<T> list) {
    T val;
    std::nth_element(&list[0], &list[0] + list.size() / 2, &list[0] + list.size());
    val = list[list.size() / 2];
    if (list.size() % 2 == 0) {
        std::nth_element(&list[0], &list[0] + list.size() / 2 - 1, &list[0] + list.size());
        val = (val + list[list.size() / 2 - 1]) / 2;
    }
    return val;
}

float findMinSymetric(const std::vector<std::vector<float> > &dist, const std::vector<bool> &used,
                      int limit, int &i, int &j) {
    float min = dist[0][0];
    i = 0;
    j = 0;
    for (int x = 0; x < limit; x++) {
        if (!used[x]) {
            for (int y = x + 1; y < limit; y++)
                if (!used[y] && dist[x][y] <= min) {
                    min = dist[x][y];
                    i = x;
                    j = y;
                }
        }
    }
    return min;
}

std::vector<Cluster> linkage(const std::vector<cv::Point2f> &list) {
    float inf = 10000000.0;
    std::vector<bool> used;
    for (size_t i = 0; i < 2 * list.size(); i++)
        used.push_back(false);
    std::vector<std::vector<float> > dist;
    for (size_t i = 0; i < list.size(); i++) {
        std::vector<float> line;
        for (size_t j = 0; j < list.size(); j++) {
            if (i == j)
                line.push_back(inf);
            else {
                cv::Point2f p = list[i] - list[j];
                line.push_back(sqrt(p.dot(p)));
            }
        }
        for (size_t j = 0; j < list.size(); j++)
            line.push_back(inf);
        dist.push_back(line);
    }
    for (size_t i = 0; i < list.size(); i++) {
        std::vector<float> line;
        for (size_t j = 0; j < 2 * list.size(); j++)
            line.push_back(inf);
        dist.push_back(line);
    }
    std::vector<Cluster> clusters;
    while (clusters.size() < list.size() - 1) {
        int x, y;
        float min = findMinSymetric(dist, used, list.size() + clusters.size(), x, y);
        Cluster cluster;
        cluster.first = x;
        cluster.second = y;
        cluster.dist = min;
        cluster.num = (x < (int) list.size() ? 1 : clusters[x - list.size()].num) +
                      (y < (int) list.size() ? 1 : clusters[y - list.size()].num);
        used[x] = true;
        used[y] = true;
        int limit = list.size() + clusters.size();
        for (int i = 0; i < limit; i++) {
            if (!used[i])
                dist[i][limit] = dist[limit][i] = std::min(dist[i][x], dist[i][y]);
        }
        clusters.push_back(cluster);
    }
    return clusters;
}

///Hierarchical distance-based clustering
void fcluster_rec(std::vector<int> &data, const std::vector<Cluster> &clusters, float threshold,
                  const Cluster &currentCluster, int &binId) {
    int startBin = binId;
    if (currentCluster.first >= (int) data.size())
        fcluster_rec(data, clusters, threshold, clusters[currentCluster.first - data.size()],
                     binId);
    else data[currentCluster.first] = binId;

    if (startBin == binId && currentCluster.dist >= threshold)
        binId++;
    startBin = binId;

    if (currentCluster.second >= (int) data.size())
        fcluster_rec(data, clusters, threshold, clusters[currentCluster.second - data.size()],
                     binId);
    else data[currentCluster.second] = binId;

    if (startBin == binId && currentCluster.dist >= threshold)
        binId++;
}

std::vector<int> binCount(const std::vector<int> &T) {
    std::vector<int> result;
    for (size_t i = 0; i < T.size(); i++) {
        while (T[i] >= (int) result.size())
            result.push_back(0);
        result[T[i]]++;
    }
    return result;
}

int argmax(const std::vector<int> &list) {
    int max = list[0];
    int id = 0;
    for (size_t i = 1; i < list.size(); i++)
        if (list[i] > max) {
            max = list[i];
            id = i;
        }
    return id;
}

///Hierarchical distance-based clustering
std::vector<int> fcluster(const std::vector<Cluster> &clusters, float threshold) {
    std::vector<int> data;
    for (size_t i = 0; i < clusters.size() + 1; i++)
        data.push_back(0);
    int binId = 0;
    fcluster_rec(data, clusters, threshold, clusters[clusters.size() - 1], binId);
    return data;
}

bool comparatorPairSecond(const std::pair<int, int> &l, const std::pair<int, int> &r) {
    return l.second < r.second;
}

std::vector<int> argSortInt(const std::vector<int> &list) {
    std::vector<int> result(list.size());
    std::vector<std::pair<int, int> > pairList(list.size());
    for (int i = 0; i < list.size(); i++)
        pairList[i] = std::make_pair(i, list[i]);
    std::sort(&pairList[0], &pairList[0] + pairList.size(), comparatorPairSecond);
    for (int i = 0; i < list.size(); i++)
        result[i] = pairList[i].first;
    return result;
}
*/

void ConsensusMatchingTracker::estimate(const std::vector<std::pair<cv::KeyPoint, int> > &keypointsIN,
                   cv::Point2f &center, float &scaleEstimate, float &medRot,
                   std::vector<std::pair<cv::KeyPoint, int> > &keypoints) {
    center = cv::Point2f(NAN, NAN);
    scaleEstimate = NAN;
    medRot = NAN;

    //At least 2 keypoints are needed for scale
    if (keypointsIN.size() > 1) {
        //sort
        std::vector<PairInt> list;
        for (size_t i = 0; i < keypointsIN.size(); i++)
            list.push_back(std::make_pair(keypointsIN[i].second, i));
        std::sort(&list[0], &list[0] + list.size(), comparatorPair<int>);
        for (size_t i = 0; i < list.size(); i++)
            keypoints.push_back(keypointsIN[list[i].second]);

        std::vector<int> ind1;
        std::vector<int> ind2;
        for (size_t i = 0; i < list.size(); i++)
            for (size_t j = 0; j < list.size(); j++) {
                if (i != j && keypoints[i].second != keypoints[j].second) {
                    ind1.push_back(i);
                    ind2.push_back(j);
                }
            }
        if (ind1.size() > 0) {
            std::vector<int> class_ind1;
            std::vector<int> class_ind2;
            std::vector<cv::KeyPoint> pts_ind1;
            std::vector<cv::KeyPoint> pts_ind2;
            for (size_t i = 0; i < ind1.size(); i++) {
                class_ind1.push_back(keypoints[ind1[i]].second - 1);
                class_ind2.push_back(keypoints[ind2[i]].second - 1);
                pts_ind1.push_back(keypoints[ind1[i]].first);
                pts_ind2.push_back(keypoints[ind2[i]].first);
            }

            std::vector<float> scaleChange;
            std::vector<float> angleDiffs;
            for (size_t i = 0; i < pts_ind1.size(); i++) {
                cv::Point2f p = pts_ind2[i].pt - pts_ind1[i].pt;
                //This distance might be 0 for some combinations,
                //as it can happen that there is more than one keypoint at a single location
                float dist = sqrt(p.dot(p));
                float origDist = squareForm[class_ind1[i]][class_ind2[i]];
                scaleChange.push_back(dist / origDist);
                //Compute angle
                float angle = atan2(p.y, p.x);
                float origAngle = angles[class_ind1[i]][class_ind2[i]];
                float angleDiff = angle - origAngle;
                //Fix long way angles
                if (fabs(angleDiff) > CV_PI)
                    angleDiff -= sign(angleDiff) * 2 * CV_PI;
                angleDiffs.push_back(angleDiff);
            }
            scaleEstimate = median(scaleChange);
            if (!estimateScale)
                scaleEstimate = 1;
            medRot = median(angleDiffs);
            if (!estimateRotation)
                medRot = 0;
            votes = std::vector<cv::Point2f>();
            for (size_t i = 0; i < keypoints.size(); i++)
                votes.push_back(keypoints[i].first.pt -
                                scaleEstimate * rotate(springs[keypoints[i].second - 1], medRot));
            //Compute linkage between pairwise distances
            std::vector<Cluster> linkageData = linkage(votes);

            //Perform hierarchical distance-based clustering
            std::vector<int> T = fcluster(linkageData, thrOutlier);

            //Count votes for each cluster
            std::vector<int> cnt = binCount(T);

            //Get largest class
            int Cmax = argmax(cnt);

            //Remember outliers
            outliers = std::vector<std::pair<cv::KeyPoint, int> >();
            std::vector<std::pair<cv::KeyPoint, int> > newKeypoints;
            std::vector<cv::Point2f> newVotes;
            for (size_t i = 0; i < keypoints.size(); i++) {
                if (T[i] != Cmax)
                    outliers.push_back(keypoints[i]);
                else {
                    newKeypoints.push_back(keypoints[i]);
                    newVotes.push_back(votes[i]);
                }
            }
            keypoints = newKeypoints;

            center = cv::Point2f(0, 0);
            for (size_t i = 0; i < newVotes.size(); i++)
                center += newVotes[i];
            center *= (1.0 / newVotes.size());
        }
    }
}

/*
//todo : n*log(n) by sorting the second array and dichotomic search instead of n^2
std::vector<bool> in1d(const std::vector<int> &a, const std::vector<int> &b) {
    std::vector<bool> result;
    for (size_t i = 0; i < a.size(); i++) {
        bool found = false;
        for (size_t j = 0; j < b.size(); j++)
            if (a[i] == b[j]) {
                found = true;
                break;
            }
        result.push_back(found);
    }
    return result;
}
*/

void ConsensusMatchingTracker::processFrame(cv::Mat &im_gray, cv::Mat &im_rgba) {
    trackedKeypoints = std::vector<std::pair<cv::KeyPoint, int> >();
    std::vector<unsigned char> status;
    track(im_prev, im_gray, activeKeypoints, trackedKeypoints, status);

    cv::Point2f center;
    float scaleEstimate;
    float rotationEstimate;
    std::vector<std::pair<cv::KeyPoint, int> > trackedKeypoints2;
    estimate(trackedKeypoints, center, scaleEstimate, rotationEstimate, trackedKeypoints2);
    trackedKeypoints = trackedKeypoints2;

    //Detect keypoints, compute descriptors
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat features;
    detector->detect(im_gray, keypoints);
    descriptorExtractor->compute(im_gray, keypoints, features);

    //Create list of active keypoints
    activeKeypoints = std::vector<std::pair<cv::KeyPoint, int> >();

    //Get the best two matches for each feature
    std::vector<std::vector<cv::DMatch> > matchesAll, selectedMatchesAll;
    descriptorMatcher->knnMatch(features, featuresDatabase, matchesAll, 2);

    //Get all matches for selected features
    if (!std::isnan(center.x) && !std::isnan(center.y))
        descriptorMatcher->knnMatch(features, selectedFeatures, selectedMatchesAll,
                                    selectedFeatures.rows);

    std::vector<cv::Point2f> transformedSprings(springs.size());
    for (int i = 0; i < springs.size(); i++)
        transformedSprings[i] = scaleEstimate * rotate(springs[i], -rotationEstimate);

    //For each keypoint and its descriptor
    for (size_t i = 0; i < keypoints.size(); i++) {
        cv::KeyPoint keypoint = keypoints[i];

        //First: Match over whole image
        //Compute distances to all descriptors
        std::vector<cv::DMatch> matches = matchesAll[i];

        //Convert distances to confidences, do not weight
        std::vector<float> combined;
        for (size_t j = 0; j < matches.size(); j++)
            combined.push_back(1 - matches[j].distance / descriptorLength);

        std::vector<int> &classes = classesDatabase;

        //Get best and second best index
        int bestInd = matches[0].trainIdx;
        int secondBestInd = matches[1].trainIdx;

        //Compute distance ratio according to Lowe
        float ratio = (1 - combined[0]) / (1 - combined[1]);

        //Extract class of best match
        int keypoint_class = classes[bestInd];

        //If distance ratio is ok and absolute distance is ok and keypoint class is not background
        if (ratio < thrRatio && combined[0] > thrConf && keypoint_class != 0)
            activeKeypoints.push_back(std::make_pair(keypoint, keypoint_class));

        //In a second step, try to match difficult keypoints
        //If structural constraints are applicable
        if (!(std::isnan(center.x) | std::isnan(center.y))) {
            //Compute distances to initial descriptors
            std::vector<cv::DMatch> matches = selectedMatchesAll[i];
            std::vector<float> distances(matches.size()), distancesTmp(matches.size());
            std::vector<int> trainIndex(matches.size());
            for (int j = 0; j < matches.size(); j++) {
                distancesTmp[j] = matches[j].distance;
                trainIndex[j] = matches[j].trainIdx;
            }
            //Re-order the distances based on indexing
            std::vector<int> idxs = argSortInt(trainIndex);
            for (int j = 0; j < idxs.size(); j++)
                distances[j] = distancesTmp[idxs[j]];

            //Convert distances to confidences
            std::vector<float> confidences(matches.size());
            for (size_t j = 0; j < matches.size(); j++)
                confidences[j] = 1 - distances[j] / descriptorLength;

            //Compute the keypoint location relative to the object center
            cv::Point2f relative_location = keypoint.pt - center;

            //Compute the distances to all springs
            std::vector<float> displacements(springs.size());
            for (size_t j = 0; j < springs.size(); j++) {
                cv::Point2f p = (transformedSprings[j] - relative_location);
                displacements[j] = sqrt(p.dot(p));
            }

            //For each spring, calculate weight
            std::vector<float> combined(confidences.size());
            for (size_t j = 0; j < confidences.size(); j++)
                combined[j] = (displacements[j] < thrOutlier) * confidences[j];

            std::vector<int> &classes = selectedClasses;

            //Sort in descending order
            std::vector<PairFloat> sorted_conf(combined.size());
            for (size_t j = 0; j < combined.size(); j++)
                sorted_conf[j] = std::make_pair(combined[j], j);
            std::sort(&sorted_conf[0], &sorted_conf[0] + sorted_conf.size(),
                      comparatorPairDesc<float>);

            //Get best and second best index
            int bestInd = sorted_conf[0].second;
            int secondBestInd = sorted_conf[1].second;

            //Compute distance ratio according to Lowe
            float ratio = (1 - combined[bestInd]) / (1 - combined[secondBestInd]);

            //Extract class of best match
            int keypoint_class = classes[bestInd];

            //If distance ratio is ok and absolute distance is ok and keypoint class is not background
            if (ratio < thrRatio && combined[bestInd] > thrConf && keypoint_class != 0) {
                for (int i = activeKeypoints.size() - 1; i >= 0; i--)
                    if (activeKeypoints[i].second == keypoint_class)
                        activeKeypoints.erase(activeKeypoints.begin() + i);
                activeKeypoints.push_back(std::make_pair(keypoint, keypoint_class));
            }
        }
    }

    //If some keypoints have been tracked
    if (trackedKeypoints.size() > 0) {
        //Extract the keypoint classes
        std::vector<int> tracked_classes(trackedKeypoints.size());
        for (size_t i = 0; i < trackedKeypoints.size(); i++)
            tracked_classes[i] = trackedKeypoints[i].second;
        //If there already are some active keypoints
        if (activeKeypoints.size() > 0) {
            //Add all tracked keypoints that have not been matched
            std::vector<int> associated_classes(activeKeypoints.size());
            for (size_t i = 0; i < activeKeypoints.size(); i++)
                associated_classes[i] = activeKeypoints[i].second;
            std::vector<bool> notmissing = in1d(tracked_classes, associated_classes);
            for (size_t i = 0; i < trackedKeypoints.size(); i++)
                if (!notmissing[i])
                    activeKeypoints.push_back(trackedKeypoints[i]);
        }
        else activeKeypoints = trackedKeypoints;
    }

    //Update object state estimate
    std::vector<std::pair<cv::KeyPoint, int> > activeKeypointsBefore = activeKeypoints;
    im_prev = im_gray;
    topLeft = cv::Point2f(NAN, NAN);
    topRight = cv::Point2f(NAN, NAN);
    bottomLeft = cv::Point2f(NAN, NAN);
    bottomRight = cv::Point2f(NAN, NAN);

    boundingbox = cv::Rect_<float>(NAN, NAN, NAN, NAN);
    hasResult = false;
    if (!(std::isnan(center.x) | std::isnan(center.y)) &&
        activeKeypoints.size() > initialKeypointSize / 10) {
        hasResult = true;

        topLeft = center + scaleEstimate * rotate(centerToTopLeft, rotationEstimate);
        topRight = center + scaleEstimate * rotate(centerToTopRight, rotationEstimate);
        bottomLeft = center + scaleEstimate * rotate(centerToBottomLeft, rotationEstimate);
        bottomRight = center + scaleEstimate * rotate(centerToBottomRight, rotationEstimate);

        /// Compute the x and y scale factor
        //float px = (float) im_rgba.cols / (float) im_gray.cols;
        //float py = (float) im_rgba.rows / (float) im_gray.rows;

        float px = 1.0;
        float py = 1.0;

        // Scaled corners
        cv::Point _topLeft = cv::Point(topLeft.x * px, topLeft.y * py);
        cv::Point _topRight = cv::Point(topRight.x * px, topRight.y * py);
        cv::Point _bottomLeft = cv::Point(bottomLeft.x * px, bottomLeft.y * py);
        cv::Point _bottomRight = cv::Point(bottomRight.x * px, bottomRight.y * py);


        float minx = std::min(std::min(_topLeft.x, _topRight.x),
                              std::min(_bottomRight.x, _bottomLeft.x));
        float miny = std::min(std::min(_topLeft.y, _topRight.y),
                              std::min(_bottomRight.y, _bottomLeft.y));
        float maxx = std::max(std::max(_topLeft.x, _topRight.x),
                              std::max(_bottomRight.x, _bottomLeft.x));
        float maxy = std::max(std::max(_topLeft.y, _topRight.y),
                              std::max(_bottomRight.y, _bottomLeft.y));

        boundingbox = cv::Rect_<float>(minx, miny, maxx - minx, maxy - miny);

        /// Draw upright bounding box
        /*
        cv::rectangle(
                im_rgba,
                boundingbox.tl(),
                boundingbox.br(),
                cv::Scalar(0x00, 0x00, 0xff)  // blue  
        );
        */

        // Draw the keypoints in green color
        for (size_t i = 0; i < trackedKeypoints.size(); i++) {
            const cv::KeyPoint &kp = trackedKeypoints[i].first;
            cv::circle(im_rgba, cv::Point(kp.pt.x * px, kp.pt.y * py), 8, cv::Scalar(0, 255, 0));
        }

        /// Rotated box
        cv::line(im_rgba, _topLeft, _topRight, cv::Scalar(255, 0, 0));
        cv::line(im_rgba, _topRight, _bottomRight, cv::Scalar(255, 0, 0));
        cv::line(im_rgba, _bottomRight, _bottomLeft, cv::Scalar(255, 0, 0));
        cv::line(im_rgba, _bottomLeft, _topLeft, cv::Scalar(255, 0, 0));

    }
}
