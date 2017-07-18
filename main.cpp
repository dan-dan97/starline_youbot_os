#include <iostream>
#include <vector>
#include <cv.h>
#include <highgui.h>
#include <Eigen/Eigen>

using namespace std;
using namespace Eigen;

const int W = 1000;
const int H = 500;
const double EPS = 1e-10;

vector <CvPoint> point1;
vector <CvPoint> point2;
IplImage *img = cvCreateImage(cvSize(W,H),IPL_DEPTH_8U,3);

void MyMouse( int event, int x, int y, int flags, void* param )
{
    bool f = false;
    switch( event )
    {
        case CV_EVENT_LBUTTONDOWN:
                for (int i = 0; i < point1.size(); i++)
                {
                    if ((abs(point1[i].x-x)<5) && (abs(point1[i].y-y)<5))
                    {
                        for (int j = i; j < point1.size()-1; j++) point1[j] = point1[j+1];

                        point1.pop_back();

                        f = true;
                    }
                }

                if (!f) point1.push_back(cvPoint(x,y));
            break;

        case CV_EVENT_RBUTTONDOWN:
                for (int i = 0; i < point2.size(); i++)
                {
                    if ((abs(point2[i].x-x)<5) && (abs(point2[i].y-y)<5))
                    {
                        for (int j = i; j < point2.size()-1; j++) point2[j] = point2[j+1];

                        point2.pop_back();

                        f = true;
                    }
                }

                if (!f) point2.push_back(cvPoint(x,y));
            break;
    }
}

void draw()
{
    cvSetZero(img);

    for (int i = 0; i < point1.size(); i++) cvCircle(img, point1[i], 3, CV_RGB(0,0,255), -1);
    for (int i = 0; i < point2.size(); i++)
    {
        cvCircle(img, point2[i], 3, CV_RGB(255,0,0), -1);

        CvPoint p = cvPoint(1e5, 1e5);
        for (int j = 0; j < point1.size(); j++)
            if ((long long)(point2[i].x - p.x)*(point2[i].x - p.x) + (long long)(point2[i].y - p.y)*(point2[i].y - p.y) >
                (long long)(point2[i].x - point1[j].x)*(point2[i].x - point1[j].x) + (long long)(point2[i].y - point1[j].y)*(point2[i].y - point1[j].y)) p = point1[j];

        cvLine(img, p, point2[i], CV_RGB(0,255,0), 1);
    }

    cvShowImage("img", img);
    cvWaitKey(1);
}

void icp()
{
    int n = point2.size();
    MatrixXd A(2*n, 3);
    VectorXd b(2*n), x(n);
    vector <CvPoint> point(n);
    double a = 0, tx = 0, ty = 0;
    double oa = 1e10, otx = 1e10, oty = 1e10;

    while(abs(a-oa) + abs(tx-otx) + abs(ty-oty) > EPS)
    {
        for (int i = 0; i < n; i++)
        {
            point[i] = cvPoint(1e5, 1e5);
            for (int j = 0; j < point1.size(); j++)
                if ((long long)(point2[i].x - point[i].x)*(point2[i].x - point[i].x) + (long long)(point2[i].y - point[i].y)*(point2[i].y - point[i].y) >
                    (long long)(point2[i].x - point1[j].x)*(point2[i].x - point1[j].x) + (long long)(point2[i].y - point1[j].y)*(point2[i].y - point1[j].y)) point[i] = point1[j];

            A(2*i,0)   = -point2[i].y;
            A(2*i,1)   = 1;
            A(2*i,2)   = 0;
            A(2*i+1,0) = point2[i].x;
            A(2*i+1,1) = 0;
            A(2*i+1,2) = 1;
            b(2*i)     = point[i].x - point2[i].x;
            b(2*i+1)   = point[i].y - point2[i].y;
        }

        x = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b);
        oa = a;
        otx = tx;
        oty = ty;
        a = x(0);
        tx = x(1);
        ty = x(2);

        for (int i = 0; i < n; i++)
        {
            double real_x = point2[i].x*cos(a) - point2[i].y*sin(a) + tx;
            double real_y = point2[i].x*sin(a) + point2[i].y*cos(a) + ty;
            point2[i] = cvPoint((int)real_x, (int)real_y);
        }

        draw();
        cvWaitKey();
    }
    cout << "-------- THE END -------" << endl;
}

int main()
{
    cvNamedWindow("img", CV_WINDOW_AUTOSIZE);
    cvSetMouseCallback("img", MyMouse, NULL);

    while (1)
    {
        char c = cvWaitKey(1);
        if (c == 27) return 0;
        if (c == 10) icp();

        draw();

        cvShowImage("img", img);
    }

    return 0;
}