// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2016 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: vitus@google.com (Michael Vitus)
//
// An example of solving a graph-based formulation of Simultaneous Localization
// and Mapping (SLAM). It reads a 2D pose graph problem definition file in the
// g2o format, formulates and solves the Ceres optimization problem, and outputs
// the original and optimized poses to file for plotting.

#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "angle_local_parameterization.h"
#include "ceres/ceres.h"
#include "read_g2o.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "pose_graph_2d_error_term.h"
#include "types.h"

DEFINE_string(input, "", "The pose graph definition filename in g2o format.");

namespace ceres
{
    namespace pose_2d
    {
        namespace
        {

            // Constructs the nonlinear least squares optimization problem from the pose
            // graph constraints.
            void BuildOptimizationProblem(const std::vector<Constraint2d> &constraints,
                                          std::map<int, Pose2d> *poses,
                                          ceres::Problem *problem)
            {
                CHECK(poses != NULL);
                CHECK(problem != NULL);
                if (constraints.empty())
                {
                    LOG(INFO) << "No constraints, no problem to optimize.";
                    return;
                }

                ceres::LossFunction *loss_function = NULL;
                ceres::LocalParameterization *angle_local_parameterization =
                    AngleLocalParameterization::Create();

                for (std::vector<Constraint2d>::const_iterator constraints_iter =
                         constraints.begin();
                     constraints_iter != constraints.end(); ++constraints_iter)
                {
                    const Constraint2d &constraint = *constraints_iter;

                    std::map<int, Pose2d>::iterator pose_begin_iter =
                        poses->find(constraint.id_begin);

                    
                    CHECK(pose_begin_iter != poses->end())
                        << "Pose with ID: " << constraint.id_begin << " not found.";
                    std::map<int, Pose2d>::iterator pose_end_iter =
                        poses->find(constraint.id_end);
                    CHECK(pose_end_iter != poses->end())
                        << "Pose with ID: " << constraint.id_end << " not found.";

                    const Eigen::Matrix3d sqrt_information =
                        constraint.information.llt().matrixL();
                    // Ceres will take ownership of the pointer.

                    Eigen::Matrix<double, 2, 2> rot_v1 = RotationMatrix2D(constraint.vertex_1[2]);
                    Eigen::Matrix<double, 2, 1> tran_v1;
                    tran_v1 << constraint.vertex_1[0], constraint.vertex_1[1];
 
                    Eigen::Matrix<double, 2, 2> rot_v2 = RotationMatrix2D(constraint.vertex_2[2]);
                    Eigen::Matrix<double, 2, 1> tran_v2;
                    tran_v2 << constraint.vertex_2[0], constraint.vertex_2[1];

                    Eigen::Matrix<double, 2, 2> rot_est = RotationMatrix2D(constraint.est_pose[2]);
                    Eigen::Matrix<double, 2, 1> tran_est;
                    tran_est << constraint.est_pose[0], constraint.est_pose[1];

                    
                    Eigen::Matrix2d rot_mat;
                    rot_mat = rot_v1 * rot_est * rot_v2.transpose();
                    //这里目前存在一点问题，直接换算到估计的中心可能优化不出结果，最好是在estimation的地方计算误差
                    Eigen::Matrix<double,2,1> trans_mat_ = rot_v1*( rot_est * (-rot_v2.transpose() *tran_v2 ) + tran_est) + tran_v1;

                    double yaw_angle = std::atan2(rot_mat(1,0),rot_mat(0,0));

                    // std::cout<<"estimated robot center trans is "<<trans_mat_(0,0)<<"  "<<trans_mat_(1,0)<<std::endl;
                    // std::cout<<"estimated robot center yaw (in radian) is "<<yaw_angle<<std::endl;


                    ceres::CostFunction *cost_function = PoseGraph2dErrorTerm::Create(
                        yaw_angle, trans_mat_ , sqrt_information);

                    problem->AddResidualBlock(
                        cost_function, loss_function, &pose_begin_iter->second.x,
                        &pose_begin_iter->second.y, &pose_begin_iter->second.yaw_radians,
                        &pose_end_iter->second.x, &pose_end_iter->second.y,
                        &pose_end_iter->second.yaw_radians);

                    problem->SetParameterization(&pose_begin_iter->second.yaw_radians,
                                                 angle_local_parameterization);
                    problem->SetParameterization(&pose_end_iter->second.yaw_radians,
                                                 angle_local_parameterization);
                }

                // The pose graph optimization problem has three DOFs that are not fully
                // constrained. This is typically referred to as gauge freedom. You can apply
                // a rigid body transformation to all the nodes and the optimization problem
                // will still have the exact same cost. The Levenberg-Marquardt algorithm has
                // internal damping which mitigate this issue, but it is better to properly
                // constrain the gauge freedom. This can be done by setting one of the poses
                // as constant so the optimizer cannot change it.
                std::map<int, Pose2d>::iterator pose_start_iter =
                    poses->begin();
                CHECK(pose_start_iter != poses->end()) << "There are no poses.";
                problem->SetParameterBlockConstant(&pose_start_iter->second.x);
                problem->SetParameterBlockConstant(&pose_start_iter->second.y);
                problem->SetParameterBlockConstant(&pose_start_iter->second.yaw_radians);
            }

            // Returns true if the solve was successful.
            bool SolveOptimizationProblem(ceres::Problem *problem)
            {
                CHECK(problem != NULL);

                ceres::Solver::Options options;
                options.max_num_iterations = 1000;
                options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;

                ceres::Solver::Summary summary;
                ceres::Solve(options, problem, &summary);

                // std::cout << summary.FullReport() << '\n';

                return summary.IsSolutionUsable();
            }

            // Output the poses to the file with format: ID x y yaw_radians.
            bool OutputPoses(const std::string &filename,
                             const std::map<int, Pose2d> &poses)
            {
                std::fstream outfile;
                outfile.open(filename.c_str(), std::istream::out);
                if (!outfile)
                {
                    std::cerr << "Error opening the file: " << filename << '\n';
                    return false;
                }
                for (std::map<int, Pose2d>::const_iterator poses_iter = poses.begin();
                     poses_iter != poses.end(); ++poses_iter)
                {
                    const std::map<int, Pose2d>::value_type &pair = *poses_iter;
                    outfile << pair.first << " " << pair.second.x << " " << pair.second.y
                            << ' ' << pair.second.yaw_radians << '\n';
                }
                return true;
            }

        } // namespace
    }     // namespace pose_2d
} // namespace ceres

int main(int argc, char **argv)
{
    std::map<int, ceres::pose_2d::Pose2d> poses;           // pose map
    std::vector<ceres::pose_2d::Constraint2d> constraints; // constraints

    poses.clear();
    constraints.clear();

    std::string data_type;
    while (std::cin.good())
    {
        // Read whether the type is a node or a constraint.
        std::cin >> data_type;
        if (data_type == "VERTEX_SE2")
        {
            int id;
            ceres::pose_2d::Pose2d pose;
            std::cin >> id >> pose;
            // Ensure we don't have duplicate poses.
            if (poses.find(id) != poses.end())
            {
                LOG(ERROR) << "Duplicate vertex with ID: " << id;
                return false;
            }
            poses[id] = pose;
        }
        else if (data_type == "EDGE_SE2")
        {
            ceres::pose_2d::Constraint2d constraint;
            std::cin >> constraint;
            constraints.push_back(constraint);
        }
        else
        {
            LOG(ERROR) << "Unknown data type: " << data_type;
            return false;
        }

        // Clear any trailing whitespace from the line.
        std::cin >> std::ws;
    }

    ceres::Problem problem;
    ceres::pose_2d::BuildOptimizationProblem(constraints, &poses, &problem);

    CHECK(ceres::pose_2d::SolveOptimizationProblem(&problem))
        << "The solve was not successful, exiting.";

    for (std::map<int, ceres::pose_2d::Pose2d>::const_iterator poses_iter = poses.begin();
         poses_iter != poses.end(); ++poses_iter)
    {
        const std::map<int, ceres::pose_2d::Pose2d>::value_type &pair = *poses_iter;
        std::cout << pair.first << " " << pair.second.x << " " << pair.second.y
                  << ' ' << pair.second.yaw_radians << '\n';
    }

    
    return 0;
}
