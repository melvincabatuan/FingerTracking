//
// Created by cobalt on 1/5/16.
//

#ifndef MHEALTH_COMMON_H
#define MHEALTH_COMMON_H

#include <android/log.h>
#include <opencv2/imgproc.hpp>

#define _USE_MATH_DEFINES
#include <cmath>

#define LOG_TAG "mhealth_vision"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))
 
#include <sys/time.h>
#include <time.h>
#include <vector>

#if __cplusplus < 201103L  

    #include <limits>

    #ifndef NAN

        #define NAN std::numeric_limits<float>::quiet_NaN()

        template <T>
        bool isnan(T d)
        {
          return d != d;
        }
    #endif

#endif


typedef std::pair<int,int> PairInt;
typedef std::pair<float,int> PairFloat;

struct Cluster
{
    int first, second; 
    float dist;
    int num;
};

uint64_t ticks_ms();

/* Template definitions are meant to be in headers */
template<typename T>
bool comparatorPair ( const std::pair<T,int>& l, const std::pair<T,int>& r)
{
       return l.first < r.first;
}

template<typename T>
bool comparatorPairDesc ( const std::pair<T,int>& l, const std::pair<T,int>& r)
{
            return l.first > r.first;
}

template <typename T>
T sign(T t)
{
            if( t == 0 )
                return T(0);
            else
                return (t < 0) ? T(-1) : T(1);
}

template<typename T>
T median(std::vector<T> list)
{
            T val;
            std::nth_element(&list[0], &list[0]+list.size()/2, &list[0]+list.size());
            val = list[list.size()/2];
            if(list.size()%2==0)
            {
                std::nth_element(&list[0], &list[0]+list.size()/2-1, &list[0]+list.size());
                val = (val+list[list.size()/2-1])/2;
            }
            return val;
}

int argmax(const std::vector<int> &list);

std::vector<int> argSortInt(const std::vector<int> &list);

bool comparatorPairSecond(const std::pair<int, int> &l, const std::pair<int, int> &r);

std::vector<bool> in1d(const std::vector<int>& a, const std::vector<int>& b);

std::vector<bool> in1dichotomic(const std::vector<int>& a, std::vector<int>& b);

std::vector<int> binCount(const std::vector<int> &T);

float findMinSymetric(const std::vector<std::vector<float> > &dist, const std::vector<bool> &used, int limit, int &i, int &j);

std::vector<Cluster> linkage(const std::vector<cv::Point2f> &list);

///Hierarchical distance-based clustering
void fcluster_rec(std::vector<int> &data, const std::vector<Cluster> &clusters, float threshold,
                  const Cluster &currentCluster, int &binId);

std::vector<int> fcluster(const std::vector<Cluster> &clusters, float threshold);  


class Queue
{
private:
    const static int _Long = 30;
public:
    int Data[_Long][2];
    int Front;
    int rear;
    int num;
    Queue()
    {
        Front = 0;
        rear = 0;
        num = 0;
    }


    bool Empty()
    {
        return (Front == rear);
    }

    void append(int _data[2])
    {
        Data[rear][0] = _data[0];
        Data[rear][1] = _data[1];
        rear++;
        num++;
        if(rear > _Long - 1)
            rear = 0;
        if(num > _Long)
        {
            Front++;
            num = _Long;
        }        
        if(Front > _Long - 1)
            Front = 0;
    }

    void Draw_Trace(cv::Mat frame)
    {
        int index = Front + 1;
        int p_index = Front;
        if(num > 0)
        {
            for(int i = 1; i < num; i++)
            {
                if(index > _Long - 1)
                    index = 0;
                cv::Point p1 = cv::Point(Data[p_index][0], Data[p_index][1]);
                cv::Point p2 = cv::Point(Data[index][0], Data[index][1]);
                cv::line(frame, p1, p2, CV_RGB(255, 0, 0), 2);
                p_index = index;
                index ++;
            }
        }
    }
};  
        
#endif //MHEALTH_COMMON_H
