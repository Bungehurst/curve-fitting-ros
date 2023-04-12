#include <iostream>
#include <ceres/ceres.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <matplot/matplot.h>
#include <cmath>

// choose auto differientiate or anlytic differientiate
// #define AUTO_DIFF
#define MANUAL_DIFF

using namespace std;
using namespace matplot;

#ifdef AUTO_DIFF
struct CURVE_FITTING_COST 
{
    CURVE_FITTING_COST(double x, double y): _x(x), _y(y) {}

    template<typename T> 
    bool operator()(const T * const abc, T *res) const {
        res[0] = T(_y) - ceres::exp(abc[0]*pow(T(_x),2)+abc[1]*T(_x)+abc[2]);
        return true;
    }
    
    const double _x, _y;
};
#endif
#ifdef MANUAL_DIFF
class ExpCostFunction : public ceres::SizedCostFunction<1,3> 
{
public:
    template<typename T>
    ExpCostFunction(const T x, const T y) : xi(x), yi(y) {}
    virtual ~ExpCostFunction() {}
    virtual bool Evaluate(double const* const* parameters,
                    double* residuals,
                    double** jacobians) const override {
        const double ae = parameters[0][0];
        const double be = parameters[0][1];
        const double ce = parameters[0][2];

        residuals[0] = yi - ceres::exp(ae*pow(xi,2)+be*xi+ce);

        if (jacobians != nullptr && jacobians[0] != nullptr) {
            jacobians[0][0] = -xi * xi * exp(ae * xi * xi + be * xi + ce);  // de/da
            jacobians[0][1] = -xi * exp(ae * xi * xi + be * xi + ce);  // de/db
            jacobians[0][2] = -exp(ae * xi * xi + be * xi + ce);  // de/dc
        }
        return true;
    }
private:
    const double xi;
    const double yi;
};
#endif

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

    double abc[3] = {ae,be,ce};
    ceres::Problem problem;
    for(int i = 0; i < N; i++){
        #ifdef AUTO_DIFF
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 3>(
                new CURVE_FITTING_COST(x_data[i],y_data[i])
            ),
            nullptr,
            abc
        );
        #endif
        #ifdef MANUAL_DIFF
        ceres::CostFunction* costfunc = new ExpCostFunction(x_data[i],y_data[i]);
        problem.AddResidualBlock(
            costfunc,
            nullptr,
            abc
        );
        #endif
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    ceres::Solve(options, &problem, &summary);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> timecost = chrono::duration_cast<chrono::duration<double>>(t2-t1);
    cout << "solve time cost = " << timecost.count() << " seconds. " << endl;

    cout << summary.BriefReport() << endl;
    cout << "ae,be,ce = ";
    for(auto a:abc) cout << a << " ";
    cout << endl;

    vector<double> y_new; 
    for (int i = 0; i < N; i++){
        y_new.push_back(exp(abc[0]*pow(x_data[i],2)+abc[1]*x_data[i]+abc[2]));
    }

    plot(x_data, y_data, "-o");
    hold(on);
    plot(x_data, y_new)->line_width(2).color("red");
    grid(on);
    show();

    return 0;
}