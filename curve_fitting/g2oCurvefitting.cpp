#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <matplot/matplot.h>
#include <cmath>

#include "3rdparty/g2o/g2o/core/factory.h"
#include "3rdparty/g2o/g2o/core/base_vertex.h"
#include "3rdparty/g2o/g2o/core/base_binary_edge.h"
#include "3rdparty/g2o/g2o/core/base_multi_edge.h"
#include "3rdparty/g2o/g2o/core/base_unary_edge.h"
#include "3rdparty/g2o/g2o/types/types_sba.h"

#include "3rdparty/g2o/g2o/core/block_solver.h"
#include "3rdparty/g2o/g2o/core/solver.h"
#include "3rdparty/g2o/g2o/core/sparse_optimizer.h"

#include "3rdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "3rdparty/g2o/g2o/solvers/linear_solver_eigen.h"

#include "3rdparty/g2o/g2o/core/optimization_algorithm_gauss_newton.h"
#include "3rdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "3rdparty/g2o/g2o/core/optimization_algorithm_factory.h"

using namespace std;
using namespace Eigen;
using namespace matplot;

// construct vertex in graph
class CurveFittingVertex : public g2o::BaseVertex<3, Vector3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW 

    virtual void setToOriginImpl() override {
        _estimate << 0, 0, 0;
    }

    virtual void oplusImpl(const double *update) override {
        _estimate += Vector3d(update);
    }

    virtual bool read(istream &in) {}
    virtual bool write(ostream &out) const {}
};

class CurveFittingEdge : public g2o::BaseUnaryEdge<1, double, CurveFittingVertex> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    CurveFittingEdge(double x) : BaseUnaryEdge(), _x(x) {}

    virtual void computeError() override {
        const CurveFittingVertex *v = static_cast<const CurveFittingVertex *> (_vertices[0]);
        const Vector3d abc = v->estimate();
        _error(0,0) = _measurement - exp(abc(0,0) * pow(_x,2) + abc(1,0) *_x + abc(2,0));
    }

    virtual void linearizeOplus() override {
        const CurveFittingVertex *v = static_cast<const CurveFittingVertex *> (_vertices[0]);
        const Vector3d abc = v->estimate();
        double y = exp(abc[0] * pow(_x,2) + abc[1]*_x + abc[2]);
        _jacobianOplusXi[0] = - pow(_x,2) * y;
        _jacobianOplusXi[1] = - _x * y;
        _jacobianOplusXi[2] = - y;
    }

    virtual bool read(istream &in) {}
    virtual bool write(ostream &out) const {}

private:
    double _x;
};

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

    
    // 初始化g2o
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<3,1> > Block;  

    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    CurveFittingVertex *v = new CurveFittingVertex();
    v->setEstimate(Vector3d(ae,be,ce));
    v->setId(0);
    optimizer.addVertex(v);

    for(int i = 0; i < N; i++){
        CurveFittingEdge *edge = new CurveFittingEdge(x_data[i]);
        edge->setId(i);
        edge->setVertex(0, v);
        edge->setMeasurement(y_data[i]);
        edge->setInformation(Matrix<double, 1, 1>::Identity() * 1/pow(inv_sigma,2));
        optimizer.addEdge(edge);
    }

    cout << "start optimization" << endl;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

    Vector3d abc_e = v->estimate();
    cout << "estimated model: " << abc_e.transpose() << endl;

    vector<double> y_new; 
    for (int i = 0; i < N; i++){
        y_new.push_back(exp(abc_e[0]*pow(x_data[i],2)+abc_e[1]*x_data[i]+abc_e[2]));
    }

    plot(x_data, y_data, "-o");
    hold(on);
    plot(x_data, y_new)->line_width(2).color("red");
    grid(on);
    show();

    return 0;
}