#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <cstdio>
#include <iostream>

using namespace cv;
using namespace std;

const int winHeight=600;
const int winWidth=800;


Point mousePosition= Point(winWidth>>1,winHeight>>1);

//mouse event callback
void mouseEvent(int event, int x, int y, int flags, void *param )
{
    if (event==EVENT_MOUSEMOVE) {
        mousePosition = Point(x,y);
    }
}

void kalManFilter(){
    RNG rng;
    //1.kalman filter setup
    const int stateNum=4;                                      //状态值4×1向量(x,y,△x,△y)
    const int measureNum=2;                                    //测量值2×1向量(x,y)
    KalmanFilter KF(stateNum, measureNum, 0);

    KF.transitionMatrix = (Mat_<float>(4, 4) <<1,0,1,0,0,1,0,1,0,0,1,0,0,0,0,1);  //转移矩阵A
    setIdentity(KF.measurementMatrix);                                             //测量矩阵H
    setIdentity(KF.processNoiseCov, Scalar::all(1e-5));                            //系统噪声方差矩阵Q
    setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));                        //测量噪声方差矩阵R
    setIdentity(KF.errorCovPost, Scalar::all(1));                                  //后验错误估计协方差矩阵P
    rng.fill(KF.statePost,RNG::UNIFORM,0,winHeight>winWidth?winWidth:winHeight);   //初始状态值x(0)
    Mat measurement = Mat::zeros(measureNum, 1, CV_32F);                           //初始测量值x'(0)，因为后面要更新这个值，所以必须先定义

    namedWindow("kalman");
    setMouseCallback("kalman",mouseEvent);

    Mat image(winHeight,winWidth,CV_8UC3,Scalar(0));

    while (true)
    {
        //2.kalman prediction
        Mat prediction = KF.predict();
        Point predict_pt = Point(prediction.at<float>(0),prediction.at<float>(1) );   //预测值(x',y')

        //3.update measurement
        measurement.at<float>(0) = (float)mousePosition.x;
        measurement.at<float>(1) = (float)mousePosition.y;

        //4.update
        KF.correct(measurement);

        //draw
        image.setTo(Scalar(255,255,255,0));
        circle(image,predict_pt,5,Scalar(0,255,0),3);    //predicted point with green
        circle(image,mousePosition,5,Scalar(255,0,0),3); //current position with red

        char buf[256];
        snprintf(buf,256,"predicted position:(%3d,%3d)",predict_pt.x,predict_pt.y);
        putText(image,buf,Point(10,30),FONT_HERSHEY_SCRIPT_COMPLEX,1,Scalar(0,0,0),1,8);
        snprintf(buf,256,"current position :(%3d,%3d)",mousePosition.x,mousePosition.y);
        putText(image,buf,Point(10,60),FONT_HERSHEY_SCRIPT_COMPLEX,1,Scalar(0,0,0),1,8);

        imshow("kalman", image);
        int key=waitKey(3);
        if (key==27){//esc
            break;
        }
    }
}

std::vector<Mat> TrackerSamplerCSCsampleImage(const Mat& img, int x, int y, int w, int h, float inrad, float outrad, int maxnum)
{
    auto rng = theRNG();
    int rowsz = img.rows - h - 1;
    int colsz = img.cols - w - 1;
    float inradsq = inrad * inrad;
    float outradsq = outrad * outrad;
    int dist;

    uint minrow = max(0, (int)y - (int)inrad);
    uint maxrow = min((int)rowsz - 1, (int)y + (int)inrad);
    uint mincol = max(0, (int)x - (int)inrad);
    uint maxcol = min((int)colsz - 1, (int)x + (int)inrad);

    fprintf(stderr,"inrad=%f minrow=%d maxrow=%d mincol=%d maxcol=%d\n",inrad,minrow,maxrow,mincol,maxcol);

    std::vector<Mat> samples;
    samples.resize((maxrow - minrow + 1) * (maxcol - mincol + 1));
    int i = 0;

    float prob = ((float)(maxnum)) / samples.size();

    Mat rectImg(img.size(),CV_8UC3,Scalar(0,0,0));
    for (int r = minrow; r <= int(maxrow); r++)
        for (int c = mincol; c <= int(maxcol); c++)
        {
            dist = (y - r) * (y - r) + (x - c) * (x - c);
            if (float(rng.uniform(0.f, 1.f)) < prob && dist < inradsq && dist >= outradsq)
            {
                samples[i] = img(Rect(c, r, w, h));
                i++;

                rectangle(rectImg,Rect(c, r, w, h),Scalar(rng.uniform(0.f, 1.f)*255,rng.uniform(0.f, 1.f)*255,90),1);
                namedWindow("rect",2);
                imshow("rect",rectImg);
                waitKey();
            }
        }

    samples.resize(min(i, maxnum));
    return samples;
}


int main ()
{
//    kalManFilter();

    float inrad = 13;
    float outrad = 4;
    int maxnum = 100;

    Mat img(300,300,0);
    randu(img,0,255);
    auto vmats = TrackerSamplerCSCsampleImage(img, 100, 100, 20, 20,inrad,outrad,maxnum);
}