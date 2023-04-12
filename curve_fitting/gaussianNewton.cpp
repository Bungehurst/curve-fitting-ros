#include <iostream>
#include <ceres/ceres.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <matplot/matplot.h>
#include <cmath>

using namespace std;
using namespace Eigen;
using namespace matplot;

int main(int argc, char **argv){
    double ar = 1.0, br = 2.0, cr = 1.0;         // 真实参数值
    double ae = 2.5, be = -2.0, ce = 5.0;        // 估计参数值
    int N = 1000;                                 // 数据点
    double w_sigma = 1.0;                        // 噪声Sigma值
    double inv_sigma = 1.0 / w_sigma;
    cv::RNG rng;

    // generate data with noise
    vector<double> x_data, y_data;
    for (int i = 0; i < N; i++){
        double x = i/double(N);
        x_data.push_back(x);
        y_data.push_back(exp(ar*pow(x,2)+br*x+cr)+rng.gaussian(pow(w_sigma,2)));
    }
    
    int iterations = 1000;
    double cost = 0, last_cost = 0;

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    for(int iter = 0; iter < iterations; iter++){
        Matrix3d H = Matrix3d::Zero();
        Vector3d b = Vector3d::Zero();
        cost = 0;

        for(int i = 0; i < N; i++){
            double xi = x_data[i], yi = y_data[i];
            double error = yi - exp(ae * xi * xi + be * xi + ce);
            Vector3d J; // 雅可比矩阵
            J[0] = -xi * xi * exp(ae * xi * xi + be * xi + ce);  // de/da
            J[1] = -xi * exp(ae * xi * xi + be * xi + ce);  // de/db
            J[2] = -exp(ae * xi * xi + be * xi + ce);  // de/dc

            H += pow(inv_sigma,2) * J * J.transpose();
            b += -pow(inv_sigma,2) * error * J;

            cost += pow(error,2);
        }

        Vector3d dx = H.ldlt().solve(b);
        if(isnan(dx[0]) || isnan(dx[1] || isnan(dx[2]))){
            cout << "result nan!" << endl;
            break;
        }

        if(iter > 0 && cost >= last_cost)
        {
            cout << "cost: " << cost << ">= last cost: " << last_cost << ", break." << endl;
            break;
        }

        ae += dx[0];
        be += dx[1];
        ce += dx[2];
    
        last_cost = cost;

        cout << "total cost: " << cost << ", \t\tupdate: " << dx.transpose() <<
         "\t\testimated params: " << ae << "," << be << "," << ce << endl;
    }
    

    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> timecost = chrono::duration_cast<chrono::duration<double>>(t2 - t1);

    cout << "solve time cost = " << timecost.count() << " seconds. " << endl;

    cout << "estimated abc = " << ae << ", " << be << ", " << ce << endl;
    
    vector<double> y_new; 
    for (int i = 0; i < N; i++){
        y_new.push_back(exp(ae*pow(x_data[i],2)+be*x_data[i]+ce));
    }

    plot(x_data, y_data, "-o");
    hold(on);
    plot(x_data, y_new)->line_width(2).color("red");
    grid(on);
    show();

    return 0;
}