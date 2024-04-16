#ifndef CERES_EXAMPLES_POSE_GRAPH_2D_POSE_GRAPH_2D_ERROR_TERM_H_
#define CERES_EXAMPLES_POSE_GRAPH_2D_POSE_GRAPH_2D_ERROR_TERM_H_

#include "Eigen/Core"

namespace ceres
{
  namespace pose_2d
  {

    template <typename T>
    Eigen::Matrix<T, 2, 2> RotationMatrix2D(T yaw_radians)
    {
      const T cos_yaw = ceres::cos(yaw_radians);
      const T sin_yaw = ceres::sin(yaw_radians);

      Eigen::Matrix<T, 2, 2> rotation;
      rotation << cos_yaw, -sin_yaw, sin_yaw, cos_yaw;
      return rotation;
    }

    // Computes the error term for two poses that have a relative pose measurement
    // between them. Let the hat variables be the measurement.
    //
    // residual =  information^{1/2} * [  r_a^T * (p_b - p_a) - \hat{p_ab}   ]
    //                                 [ Normalize(yaw_b - yaw_a - \hat{yaw_ab}) ]
    //
    // where r_a is the rotation matrix that rotates a vector represented in frame A
    // into the global frame, and Normalize(*) ensures the angles are in the range
    // [-pi, pi).
    class PoseGraph2dErrorTerm
    {
    public:
      PoseGraph2dErrorTerm(const double yaw_angle, const Eigen::Vector2d &trans_mat,
                           const Eigen::Matrix3d &sqrt_information) : 
                           yaw_angle_(yaw_angle),
                           trans_mat_(trans_mat),
                           sqrt_information_(sqrt_information)


      {

      }

      template <typename T>
      bool operator()(const T* const x_a, const T* const y_a, const T* const yaw_a,
                  const T* const x_b, const T* const y_b, const T* const yaw_b,
                  T* residuals_ptr) const
      {
        // 在这一部分构建残差函数
        const Eigen::Matrix<T, 2, 1> p_a(*x_a, *y_a);
        const Eigen::Matrix<T, 2, 1> p_b(*x_b, *y_b);

        Eigen::Map<Eigen::Matrix<T, 3, 1> > residuals_map(residuals_ptr);

        // Scale the residuals by the square root information matrix to account for
        // the measurement uncertainty.
        residuals_map = sqrt_information_.template cast<T>() * residuals_map;

        residuals_map.template head<2>() = p_b - p_a - trans_mat_.template cast<T>();
        residuals_map(2) = ceres::pose_2d::NormalizeAngle((*yaw_b - *yaw_a) - static_cast<T>(yaw_angle_));

        // Scale the residuals by the square root information matrix to account for
        // the measurement uncertainty.
        residuals_map = sqrt_information_.template cast<T>() * residuals_map;

        return true;
      }

      static ceres::CostFunction *Create(const double yaw_angle, const Eigen::Vector2d &trans_mat,
                           const Eigen::Matrix3d &sqrt_information)
      {
        return (new ceres::AutoDiffCostFunction<PoseGraph2dErrorTerm, 3, 1, 1, 1, 1, 1, 1>(new PoseGraph2dErrorTerm(
            yaw_angle, trans_mat, sqrt_information)));
      }

      EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    private:
      const Eigen::Matrix3d sqrt_information_;
      const Eigen::Vector2d trans_mat_;
      const double yaw_angle_;
    };

  } // namespace pose_2d
} // namespace ceres

#endif // CERES_EXAMPLES_POSE_GRAPH_2D_POSE_GRAPH_2D_ERROR_TERM_H_
