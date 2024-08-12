#include "enhancement.h"
///////////////////////////////////////////////////
#include "enhancer/enhancer.h"
////////////////////////////////////////
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <math.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>

using namespace std;
using namespace cv;

queue<sensor_msgs::ImageConstPtr> img0_buf;
queue<sensor_msgs::ImageConstPtr> img1_buf;
std::mutex m_buf;

bool STEREO = true;

ros::Publisher pub_img0_processed;
ros::Publisher pub_img1_processed;

void img0_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf.lock();
    img0_buf.push(img_msg);
    m_buf.unlock();
}

void img1_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf.lock();
    img1_buf.push(img_msg);
    m_buf.unlock();
}

cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
{
    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    cv::Mat img = ptr->image.clone();
    return img;
}

void publishImage(const cv::Mat &image, const std_msgs::Header &header, ros::Publisher &pub) {
    cv_bridge::CvImage out_msg;
    out_msg.header = header;
    out_msg.encoding = sensor_msgs::image_encodings::MONO8;
    out_msg.image = image;
    pub.publish(out_msg.toImageMsg());
}
//std::pair<cv::Mat, cv::Mat> processImages(cv::Mat image0, cv::Mat image1) {
cv::Mat processImage(cv::Mat image) {
    /****************************************** 
    //histogram 
    cv::Mat dstHist;
    int dims = 1;
    float hranges[] = {0,256};
    const float *ranges[] = {hranges};
    int size = 256;
    int channels = 0;

    cv::calcHist(&image,1,&channels,cv::Mat(),dstHist,dims,&size,ranges);
    cv::Mat dstImage(size, size, CV_8U, Scalar(0));

    double minValue = 0;
    double maxValue = 0;
    cv::minMaxLoc(image,&minValue, &maxValue, NULL, NULL);

    // This step is ref. from paper "Robust visual odometry based on image enhancement"
    // standardized
    float averge = 0;
    for(int i = 0; i<256; i++){
        float binValue = dstHist.at<float>(i);
        averge += binValue/255; 
    }

    float RM_error = 0;
    for(int i = 0; i<256; i++){
        float binValue = dstHist.at<float>(i);
        RM_error += pow((binValue - averge),2);
    }
    RM_error = sqrt(RM_error/255);

    float sum_of_SD_dst = 0;
    array<float,256> SD_dst;
    for(int i = 0; i<256 ;i++){
        float SD_value = (dstHist.at<float>(i)-averge)/RM_error;
        sum_of_SD_dst += SD_value;
        SD_dst[i] = SD_value;
    }

    // calculate the gamma(l)
    float gamma_value = 1;
    gamma_value = 1/(1 - sum_of_SD_dst);
    std::cout << "gamma: " << gamma_value << std::endl;

    // gamma_correction
    cv::Scalar mean_scaler = cv::mean(image);
    float img_Mean = mean_scaler.val[0];

    cv::Mat lookUpTable(1, 256, CV_8U);
    uchar* p = lookUpTable.ptr();
    for( int i = 0; i < 256; ++i)
        p[i] = saturate_cast<uchar>(pow(i / 255.0, gamma_value) * 255.0);
    cv::Mat res = image.clone();
    cv::LUT(image, lookUpTable, res);

    float* SD_maxValue = std::max_element(SD_dst.begin(),SD_dst.end());
    std::cout<<"Maximun Value: " << float(*SD_maxValue)<<std::endl;

    for(int i = 0; i < 256; i++)
    {
        float binValue = SD_dst[i];   
        int intensity = cvRound(binValue * (size-1) / float(*SD_maxValue));
        cv::line(dstImage,cv::Point(i, size - intensity),cv::Point(i, size - 1 ),Scalar(255));
    }
    //cv::imshow("一维直方图", dstImage);

    return res;
    **********************************/
    //return tplthe_enhancement(image);
    
    return adaptive_correction_mono_new(image);
    //return adaptive_correction_stereo(image0, image1);
    //return clahe_enhancement(image);
    
}
void showHistogram(const cv::Mat& image, const std::string& windowName) {
    // Calculate histogram
    int histSize = 256;    // bin size
    float range[] = { 0, 256 };
    const float* histRange = { range };
    cv::Mat hist;

    cv::calcHist(&image, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);

    // Normalize the histogram to fit the image height
    int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound((double) hist_w / histSize);
    cv::Mat histImage(hist_h, hist_w, CV_8UC1, cv::Scalar(0));

    cv::normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX);

    // Draw the histogram
    for(int i = 1; i < histSize; i++) {
        line(histImage, cv::Point(bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1))),
             cv::Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i))),
             cv::Scalar(255), 2, 8, 0);
    }

    // Display the histogram with a specific window name
    cv::imshow(windowName, histImage);
}

void sync_process(){
    while(1)
    {
        if(STEREO)
        {
            cv::Mat image0, image1;
            std_msgs::Header header;
            double time = 0;
            m_buf.lock();
            if (!img0_buf.empty() && !img1_buf.empty())
            {
                double time0 = img0_buf.front()->header.stamp.toSec();
                double time1 = img1_buf.front()->header.stamp.toSec();
                if(time0 < time1)
                {
                    img0_buf.pop();
                    printf("throw img0\n");
                }
                else if(time0 > time1)
                {
                    img1_buf.pop();
                    printf("throw img1\n");
                }
                else
                {
                    time = img0_buf.front()->header.stamp.toSec();
                    header = img0_buf.front()->header;
                    image0 = getImageFromMsg(img0_buf.front());
                    img0_buf.pop();
                    image1 = getImageFromMsg(img1_buf.front());
                    img1_buf.pop();
                    showHistogram(image0, "Original Image 0 Histogram");
                    // // 對 image0 進行處理
                    cv::Mat res0 = processImage(image0);

                    // // 對 image1 進行處理
                    cv::Mat res1 = processImage(image1);
                    // auto results = processImages(image0, image1);
                    // cv::Mat res0 = results.first;
                    // cv::Mat res1 = results.second;

                    if(!image0.empty()){
                        cv::imshow("image0",image0);
                        cv::imshow("correction_new_image0",res0);
                        publishImage(res0, header, pub_img0_processed);
                    }
                    if(!image1.empty()){
                        //cv::imshow("image1",image1);
                        //cv::imshow("correction_new_image1",res1);
                        publishImage(res1, header, pub_img1_processed);
                        // Display the histogram of res1
                        showHistogram(res0, "Processed Image 0 Histogram");

                    }
                }
            }
            m_buf.unlock();
        }
        else
        {
            cv::Mat image;
            std_msgs::Header header;
            double time = 0;
            m_buf.lock();
            if(!img0_buf.empty())
            {
                time = img0_buf.front()->header.stamp.toSec();
                header = img0_buf.front()->header;
                image = getImageFromMsg(img0_buf.front());
                img0_buf.pop();

                if(!image.empty()){
                    cv::imshow("image",image);
                }
            }
            m_buf.unlock();
        }

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
        cv::waitKey(1);
    }
}

int main(int argc, char** argv){

    ros::init(argc,argv,"image_enhance_node");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    ros::Subscriber sub_img0 = n.subscribe("/cam0/image_raw", 100, img0_callback);
    ros::Subscriber sub_img1 = n.subscribe("/cam1/image_raw", 100, img1_callback);

    pub_img0_processed = n.advertise<sensor_msgs::Image>("/cam0/image_processed", 10);
    pub_img1_processed = n.advertise<sensor_msgs::Image>("/cam1/image_processed", 10);

    std::thread sync_thread{sync_process};
    ros::spin();

    return 0;
}
