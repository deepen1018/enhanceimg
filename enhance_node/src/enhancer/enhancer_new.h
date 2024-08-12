#pragma once
//****** */
#ifndef ENHANCER_H
#define ENHANCER_H

#include <utility>
//******************* */
#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <numeric>
cv::Mat clahe_enhancement(cv::Mat inputImage);
//cv::Mat tplthe_enhancement(cv::Mat image0);
std::pair<cv::Mat,cv::Mat> adaptive_correction_stereo(cv::Mat image0, cv::Mat image1);
cv::Mat adaptive_correction_mono(cv::Mat image0);
#endif // ENHANCER_H