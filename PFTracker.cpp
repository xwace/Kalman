#include "PFTracker.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <iostream>
#include<fstream>

using namespace std;
using namespace cv;

namespace PFTRACKER{

    Rect select;//选定区域。cv::Rect矩形类
    Point origin;//鼠标的点。opencv中提供的点的模板类
    bool select_flag = false;//选择标志
    bool tracking = false;//跟踪标志，用于判定是否已经选择了目标位置，只有它为true时，才开始跟踪
    bool select_show = false;
    Mat frame, hsv;//全局变量定义
    int after_select_frames = 0;
//*************************************hsv空间用到的变量***************************//
    int channels[] = { 0,1,2 };//需要统计的通道dim，第一个数组通道从0到image[0].channels()-1，第二个数组从image[0].channels()到images[0].channels()+images[1].channels()-1，
    int ZhiFangTuWeiShu = 3;//需要计算直方图的维度
    int hist_size[] = { 16,16,16 };// 每个维度的直方图尺寸的数组
    float hrange[] = { 0,180.0 };//色调H的取值范围
    float srange[] = { 0,256.0 };//饱和度S的取值范围
    float vrange[] = { 0,256.0 };//亮度V的取值范围
    const float *ranges[] = { hrange,srange,vrange };//每个维度中bin的取值范围


//****有关粒子窗口变化用到的相关变量****///Rob Hess里的参数，不太懂
    int A1 = 2;
    int A2 = -1;
    int B0 = 1;
    double sigmax = 1.0;
    double sigmay = 0.5;
    double sigmas = 0.001;


//****************************定义粒子数目**********************************//
#define PARTICLE_NUMBER 100

//*******************************定义粒子结构体类型************************//
    typedef struct particle//关于typedef struct和struct见下文补充
    {
        int orix, oriy;//原始粒子坐标
        int x, y;//当前粒子的坐标
        double scale;//当前粒子窗口的尺寸
        int prex, prey;//上一帧粒子的坐标
        double prescale;//上一帧粒子窗口的尺寸
        Rect rect;//当前粒子矩形窗口
        Mat hist;//当前粒子窗口直方图特征
        double weight;//当前粒子权值
    }PARTICLE;



    //函数声明
    void onMouse(int event, int x, int y, int, void*);
    void update_PARTICLES(PARTICLE* pParticle, RNG& rng, Mat& track_img, Mat& track_hist, Mat& target_hist, double& sum);
    int particle_decrease(const void* p1, const void* p2);
    void show_FPS(Mat& frame, VideoCapture capture);
    void save_to_txt(double x, double y, VideoCapture cap);
    void save_to_excel(double x, double y, VideoCapture cap);

    int run()
    {

        //打开摄像头或者特定视频
        VideoCapture cap;
        cap.open("/home/star/Desktop/robot.mp4");

        //读入视频是否为空
        if (!cap.isOpened())
        {
            return -1;
        }

        namedWindow("输出视频", 1);
        setMouseCallback("输出视频", onMouse, nullptr);//鼠标回调函数，响应鼠标以选择跟踪区域
        /*
        void setMouseCallback(const string& winname,     //图像视窗名称
        MouseCallback onMouse,     //鼠标响应函数，监视到鼠标操作后调用并处理相应动作
        void* userdata = 0        //鼠标响应处理函数的ID，识别号
        );
        */

        //定义一系列mian函数中的全局变量
        PARTICLE* pParticle;//定义一个指向粒子结构体类型的指针pParticle,指针默认指着int lizi = 0
        PARTICLE particles[PARTICLE_NUMBER];//这个粒子结构体里面有100个粒子
        Mat target_img;//将目标图像选定并截取赋给target_img
        Mat target_hist;//定义框选目标输出的直方图。注意这是一个二维数组
        Mat track_img;//要跟踪的目标
        Mat track_hist;//要跟踪的直方图


        while (true)
        {
            cap >> frame;
            if (frame.empty())
            {
                return -1;
            }

            blur(frame, frame, Size(2, 2));//先对原图进行均值滤波处理
            cvtColor(frame, hsv, COLOR_BGR2HSV);//从RGB到HSV空间的转换。粒子滤波在HSV空间来处理

            if (tracking)//跟踪位为正，表明目标已选定，开始进行跟踪步骤
            {

                if (after_select_frames == 1) //确认已经选择完目标，且做跟踪前的准备工作——初始化
                {//after_select_frames的设定是为了使得初始化粒子只有一次

                    //**********************计算目标模板的直方图特征******************************//
                    target_img = Mat(hsv, select);//将目标图像选定并截取赋给target_img。select是cv::Rect矩形类
                    /*目标跟踪最重要的就是特征，应该选取好的特征（拥有各种不变性的特征当然是最好的）；
                    另外考虑算法的效率，目标跟踪一般是实时跟踪，所以对算法实时性有一定的要求。
                    Rob Hess源码提取的是目标的颜色特征（颜色特征对图像本身的尺寸、方向、视角的依赖性较小，从而具有较高的鲁棒性），粒子与目标的直方图越相似，则说明越有可能是目标。*/

                    //上一句等同于Mat target_img = hsv(select)
                    //imshow("hhah",target_img);//观看截取的效果


                    calcHist(&target_img, 1, channels, Mat(), target_hist, ZhiFangTuWeiShu, hist_size, ranges);//计算目标图像的直方图。具体参见下文对于程序的补充。Mat()为空掩码
                    normalize(target_hist, target_hist);//做归一化处理



                    //*******************************初始化目标粒子****************************/
                    pParticle = particles;//指针初始化指向particles数组
                    for (int lizi = 0;lizi < PARTICLE_NUMBER;lizi++)//对于每个粒子
                    {
                        //选定目标矩形框中心为初始粒子窗口中心
                        // cvRound对一个double型的数进行四舍五入，并返回一个整型数
                        pParticle->x = cvRound(select.x + 0.5*select.width);//当前粒子的x坐标
                        pParticle->y = cvRound(select.y + 0.5*select.height);//当前粒子的y坐标

                        //粒子的原始坐标为选定矩形框(即目标)的中心
                        pParticle->orix = pParticle->x;
                        pParticle->oriy = pParticle->y;

                        //更新上一帧粒子的坐标
                        pParticle->prex = pParticle->x;
                        pParticle->prey = pParticle->y;

                        //当前粒子窗口的尺寸
                        pParticle->scale = 1;//初始化为1，然后后面粒子到搜索的时候才通过计算更新

                        //上一帧粒子窗口的尺寸
                        pParticle->prescale = 1;

                        //当前粒子矩形窗口
                        pParticle->rect = select;

                        //当前粒子窗口直方图特征
                        pParticle->hist = target_hist;

                        //当前粒子权值
                        pParticle->weight = 0;//权重初始为0

                        pParticle++;
                    }
                }



                    //*******************************开始跟踪，进行跟踪算法步骤****************************//
                else if (after_select_frames == 2)
                {
                    pParticle = particles;//指针初始化指向particles数组。指针首先指向数组第一个元素
                    RNG rng;//随机数产生器
                    //************************更新粒子参数*****************************//
                    double sum = 0;//粒子的权重
                    update_PARTICLES(pParticle, rng, track_img, track_hist, target_hist, sum);


                    //************************归一化粒子的权重*************************//
                    pParticle = particles;
                    for (int lizishu = 0;lizishu < PARTICLE_NUMBER;lizishu++)
                    {
                        pParticle->weight /= sum;
                        pParticle++;
                    }
                    pParticle = particles;
                    qsort(pParticle, PARTICLE_NUMBER, sizeof(PARTICLE), &particle_decrease);//降序排列，按照粒子权重。参见下文对于程序的补充。也可以用sort



                    //*********************重采样，根据粒子权重重采样********************//
                    PARTICLE newParticle[PARTICLE_NUMBER];//定义一个新的粒子数组
                    int np = 0;//阈值，只要np个粒子
                    int k = 0;
                    for (int i = 0;i < PARTICLE_NUMBER;i++)
                    {
                        np = cvRound(pParticle->weight*PARTICLE_NUMBER);//将权重较弱的粒子淘汰掉，保留权重在阈值以上的
                        for (int j = 0;j < np;j++)
                        {
                            newParticle[k++] = particles[i];
                            if (k == PARTICLE_NUMBER)
                            {
                                goto out;
                            }
                        }
                    }

                    while (k < PARTICLE_NUMBER)
                    {
                        newParticle[k++] = particles[0];//复制大的权值的样本填满空间
                    }

                    out:
                    for (int i = 0; i < PARTICLE_NUMBER; i++)
                    {
                        particles[i] = newParticle[i];
                    }

                }



                //***********计算最大权重目标的期望位置，采用权值最大的1/4个粒子数作为跟踪结果************//
                Rect rectTrackingTemp(0, 0, 0, 0);//初始化一个Rect作为跟踪的临时
                double weight_temp = 0.0;
                pParticle = particles;
                for (int i = 0; i<PARTICLE_NUMBER / 4; i++)
                {
                    weight_temp += pParticle->weight;
                    pParticle++;
                }
                pParticle = particles;
                for (int i = 0; i<PARTICLE_NUMBER / 4; i++)
                {
                    pParticle->weight /= weight_temp;
                    pParticle++;
                }
                pParticle = particles;
                for (int i = 0; i<PARTICLE_NUMBER / 4; i++)
                {
                    rectTrackingTemp.x += pParticle->rect.x*pParticle->weight;
                    rectTrackingTemp.y += pParticle->rect.y*pParticle->weight;
                    rectTrackingTemp.width += pParticle->rect.width*pParticle->weight;
                    rectTrackingTemp.height += pParticle->rect.height*pParticle->weight;
                    pParticle++;
                }

                Rect tracking_rect(rectTrackingTemp);//目标矩形区域
                pParticle = particles;

                for (int m = 0; m < PARTICLE_NUMBER; m++) {
                    pParticle++;
                }


                rectangle(frame, tracking_rect, Scalar(0, 255, 0), 3, 8, 0);//显示跟踪结果，框出
                after_select_frames = 2;//保证每次都可进入跟踪算法步骤

            }

            //rectangle(frame, select, Scalar(0, 255, 0), 3, 8, 0);//显示手动选择时的目标矩形框

            imshow("输出视频", frame);
            waitKey(30);

        }
        return 0;
    }


//**************************手动选择跟踪目标区域*************************//
//onMouse的响应函数
    void onMouse(int event, int x, int y, int, void*)//鼠标事件回调函数onMouse按照固定格式创建响应函数
    {
        /*
        Event:
        #define CV_EVENT_MOUSEMOVE 0             //滑动
        #define CV_EVENT_LBUTTONDOWN 1           //左键点击
        #define CV_EVENT_RBUTTONDOWN 2           //右键点击
        #define CV_EVENT_MBUTTONDOWN 3           //中键点击
        #define CV_EVENT_LBUTTONUP 4             //左键放开
        #define CV_EVENT_RBUTTONUP 5             //右键放开
        #define CV_EVENT_MBUTTONUP 6             //中键放开
        #define CV_EVENT_LBUTTONDBLCLK 7         //左键双击
        #define CV_EVENT_RBUTTONDBLCLK 8         //右键双击
        #define CV_EVENT_MBUTTONDBLCLK 9         //中键双击
        */

        if (select_flag)//只有当左键按下时，才计算ROI
        {
            //select是cv::Rect矩形类
            // origin为opencv中提供的点的模板类
            select.x = MIN(origin.x, x);//鼠标按下开始到弹起这段时间实时计算所选矩形框
            select.y = MIN(origin.y, y);
            select.width = abs(x - origin.x);//算矩形宽度和高度
            select.height = abs(y - origin.y);
            select &= Rect(0, 0, frame.cols, frame.rows);//保证所选矩形框在视频显示区域之内
        }
        if (event == EVENT_LBUTTONDOWN)//当鼠标左键按下（对应1）
        {
            select_flag = true;//鼠标按下的标志赋真值
            tracking = false;
            select_show = true;
            after_select_frames = 0;//还没开始选择，或者重新开始选择，计数为0
            origin = Point(x, y);//保存下来单击时捕捉到的点（最开始的那个点）
            select = Rect(x, y, 0, 0);//这里一定要初始化，因为在opencv中Rect矩形框类内的点是包含左上角那个点的，但是不含右下角那个点。
        }
        else if (event == EVENT_LBUTTONUP)//当鼠标左键放开（对应4）
        {
            select_flag = false;
            tracking = true;//选择完毕，跟踪标志位置1。只有它为1时，才可以开始跟踪
            select_show = false;
            after_select_frames = 1;//选择完后的那一帧当做第1帧
        }
    }


//************************************粒子更新函数*************************************//
    void update_PARTICLES(PARTICLE* pParticle, RNG& rng, Mat& track_img, Mat& track_hist, Mat& target_hist, double& sum)
    {
        int xpre, ypre;
        double pres, s;
        int x, y;
        for (int lizishu = 0;lizishu < PARTICLE_NUMBER;lizishu++)
        {
            //当前粒子的坐标
            xpre = pParticle->x;
            ypre = pParticle->y;

            //当前粒子窗口的尺寸
            pres = pParticle->scale;

            /*更新跟踪矩形框中心，即粒子中心*///使用二阶动态回归来自动更新粒子状态
            x = cvRound(A1*(pParticle->x - pParticle->orix) + A2*(pParticle->prex - pParticle->orix) +
                        B0*rng.gaussian(sigmax) + pParticle->orix);
            pParticle->x = max(0, min(x, frame.cols - 1));

            y = cvRound(A1*(pParticle->y - pParticle->oriy) + A2*(pParticle->prey - pParticle->oriy) +
                        B0*rng.gaussian(sigmay) + pParticle->oriy);
            pParticle->y = max(0, min(y, frame.rows - 1));

            s = A1*(pParticle->scale - 1) + A2*(pParticle->prescale - 1) + B0*(rng.gaussian(sigmas)) + 1.0;
            pParticle->scale = max(1.0, min(s, 3.0));//此处参数设置存疑

            pParticle->prex = xpre;
            pParticle->prey = ypre;
            pParticle->prescale = pres;

            /*计算更新得到矩形框数据*/
            pParticle->rect.x = max(0, min(cvRound(pParticle->x - 0.5*pParticle->scale*pParticle->rect.width), frame.cols));
            pParticle->rect.y = max(0, min(cvRound(pParticle->y - 0.5*pParticle->scale*pParticle->rect.height), frame.rows));
            pParticle->rect.width = min(cvRound(pParticle->rect.width), frame.cols - pParticle->rect.x);
            pParticle->rect.height = min(cvRound(pParticle->rect.height), frame.rows - pParticle->rect.y);

            /*计算粒子区域的直方图*/
            track_img = Mat(hsv, pParticle->rect);
            calcHist(&track_img, 1, channels, Mat(), track_hist, ZhiFangTuWeiShu, hist_size, ranges);//计算目标图像的直方图。具体参见下文对于程序的补充。Mat()为空掩码
            normalize(track_hist, track_hist);//做归一化处理

            /*用巴氏系数计算相似度，一直与最初的目标区域相比*/
            pParticle->weight = 1.0 - compareHist(target_hist, track_hist, HISTCMP_BHATTACHARYYA);//巴氏系数计算相似度效果最好，其余两种不适合

            /*粒子权重累加*/
            sum += pParticle->weight;
            pParticle++;//指针移向下一位
        }
    }



/****粒子权重值的降序排列****/
    int particle_decrease(const void* p1, const void* p2)
    {
        PARTICLE* _p1 = (PARTICLE*)p1;//指向PARTICLE的指针
        PARTICLE* _p2 = (PARTICLE*)p2;
        if (_p1->weight < _p2->weight) {
            return 1;
        }
        else if (_p1->weight > _p2->weight) {
            return -1;
        }
        return 0;//若权重相等返回0
    }
}
