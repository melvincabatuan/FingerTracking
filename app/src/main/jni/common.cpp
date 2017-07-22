#include "common.h"

#include <android/log.h>
#include <opencv2/imgproc.hpp>

#define _USE_MATH_DEFINES
#include <cmath>

#include <sys/time.h>
#include <time.h>
#include <vector>
#include <algorithm>


uint64_t ticks_ms()
{
    timeval t;
    gettimeofday(&t, 0);
    return static_cast<uint64_t>(t.tv_sec) * 1000 + t.tv_usec / 1000;
}


//todo : n*log(n) by sorting the second array and dichotomic search instead of n^2
std::vector<bool> in1d(const std::vector<int>& a, const std::vector<int>& b)
{
            std::vector<bool> result;
            for(unsigned int i = 0; i < a.size(); i++)
            {
                bool found = false;
                for(unsigned int j = 0; j < b.size(); j++)
                    if(a[i] == b[j])
                    {
                        found = true;
                        break;
                    }
                result.push_back(found);
            }
            return result;
}

std::vector<bool> in1dichotomic(const std::vector<int>& a, std::vector<int>& b)
{
	std::sort(std::begin(b), std::end(b));	
	std::vector<bool> result;
	for (auto element : a) {
		bool found = false;
		if (std::binary_search(b.begin(), b.end(), element)) {
              found = true;
        }
        result.push_back(found);
	}
	return result;
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
