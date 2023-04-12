# curve-fitting-ros
This repo contains three curve fitting realizations depends on ROS. 
- Gaussian Newton
- Ceres
- G2O
# Prerequisites
- ros noetic
- Ceres 2.1.0
- matplotplusplus
# How to run?
After you have done the `catkin_make`, you can run
```bash
rosrun curve_fitting cerescurvefit
rosrun curve_fitting gncurvefit
rosrun curve_fitting g2ocurvefit
```