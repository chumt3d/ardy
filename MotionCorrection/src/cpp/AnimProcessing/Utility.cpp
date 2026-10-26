/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "TrajectoryCorrector.h"
#include "InverseKinematics.h"

#include "Utility.h"

#include <map>
#include <cmath>
using Pose = std::vector<Math::Transform>;

static const float pos_weight = 0.001f;
static const float vel_weight = 1.0f;
static const float acc_weight = 10.0f;


namespace {

    void CorrectHipsY(
        std::vector<Pose>& poses,
        const std::vector<Pose>& targetPoses,
        const std::vector<float>& fullBodyMask
    )
    {
        // Correct the y coordinates of the root.
        auto N = poses.size();
        Eigen::MatrixXd x(N, 1);
        Eigen::MatrixXd observations(N, 1);
        Eigen::MatrixXd xfixed(N, 1);

        // Fill in the initial trajectory (x) and the values we want to hit when we
        // warp it (observations):
        Eigen::VectorXd yCorrectMargins(N);
        for(size_t frame = 0; frame < N; ++frame)
        {
            yCorrectMargins[frame] = fullBodyMask[frame] ? 0.0f : -1.0f;
            x(frame, 0) = ((float*)&poses[frame][0].GetTranslation())[1];
            observations(frame, 0) = ((float*)&targetPoses[frame][0].GetTranslation())[1];
        }

        TrajectoryCorrector ycorrector(
            yCorrectMargins,
            pos_weight * 10,
            vel_weight,
            acc_weight * 0.1f
        );
        ycorrector.Interpolate(
            xfixed,
            observations,
            x
        );

        // fill channel again:
        for (uint32_t frame = 0; frame < N; ++frame)
        {
            ((float*)&poses[frame][0].GetTranslation())[1] = float(xfixed(frame, 0));
        }
    }

    void SmoothChannels(
        Eigen::MatrixXd &x,
        const std::vector<float>& mask
    )
    {
        for( uint32_t i=0; i < mask.size(); ++i)
        {
            uint32_t i_prev = i == 0 ? 0 : i-1;
            uint32_t i_next = std::min(uint32_t(i+1), uint32_t(mask.size()-1));
            if(i > 0 && mask[i] > 0 && mask[i_prev] == 0)
            {
                // if the previous frame is unconstrained and the current frame is constrained,
                // replace the current frame with the average of its neighbors:
                for(long j=0; j < x.cols(); ++j)
                {
                    x(i, j) = 0.5f * (x(i_prev, j) + x(i_next, j));
                }
            }
            if(mask[i] > 0 && mask[i_next] == 0)
            {
                // if the next frame is unconstrained and the current frame is constrained,
                // replace the current frame with the average of its neighbors:
                for(long j=0; j < x.cols(); ++j)
                {
                    x(i, j) = 0.5f * (x(i_prev, j) + x(i_next, j));
                }
            }
        }
    }


    void CorrectHipsXZ(
        std::vector<Pose>& poses,
        const std::vector<Pose>& targetPoses,
        const std::vector<float>& fullBodyMask,
        const std::vector<float>& rootMask,
        const std::vector<Animation::ContactInfo>& endEffectorPins,
        const Eigen::VectorXd& velocity_weights,
        float root_margin
    )
    {
        auto N = poses.size();
        Eigen::VectorXd margins(N);
        for( size_t i = 0; i < N; ++i )
        {
            margins[i] = fullBodyMask[i] ? 0.0f : -1.0f;
        }

        std::vector<float> rootCombinedMask(N, 0.0f);
        for(size_t i = 0; i < N; ++i)
        {
            rootCombinedMask[i] = (fullBodyMask[i] > 0) || (rootMask[i] > 0);
            if(rootMask[i] > 0 && margins[i] != 0)
            {
                margins[i] = root_margin;
            }
            for (auto& c : endEffectorPins)
            {
                if (c.contactMask[i] && margins[i] != 0)
                {
                    margins[i] = root_margin;
                }
            }
        }
        TrajectoryCorrector xzcorrector(
            margins,
            pos_weight,
            vel_weight,
            acc_weight,
            velocity_weights
        );

        // Enforce pose constraints on root xz trajectory:
        Eigen::MatrixXd x(N, 2);
        Eigen::MatrixXd observations(N, 2);
        Eigen::MatrixXd x_fixed(N, 2);

        observations.setZero();
        for (uint32_t frame = 0; frame < N; ++frame)
        {
            x(frame, 0) = ((float*)&poses[frame][0].GetTranslation())[0];
            x(frame, 1) = ((float*)&poses[frame][0].GetTranslation())[2];

            observations(frame, 0) = ((float*)&targetPoses[frame][0].GetTranslation())[0];
            observations(frame, 1) = ((float*)&targetPoses[frame][0].GetTranslation())[2];
        }

        SmoothChannels(x, rootCombinedMask);

        xzcorrector.Interpolate(
            x_fixed,
            observations,
            x
        );

        // fill channels again:
        for (uint32_t frame = 0; frame < N; ++frame)
        {
            ((float*)&poses[frame][0].GetTranslation())[0] = float(x_fixed(frame, 0));
            ((float*)&poses[frame][0].GetTranslation())[2] = float(x_fixed(frame, 1));
        }
    }

    void CorrectRotationsForBone(
        std::vector<Pose>& poses,
        const std::vector<Pose>& targetPoses,
        const std::vector<float>& mask,
        const TrajectoryCorrector& corrector,
        int boneIdx,
        bool performChannelSmoothing)
    {
        auto N = poses.size();
        Eigen::MatrixXd x(N, 1);
        Eigen::MatrixXd observations(N, 1);
        observations.setZero();
        Eigen::MatrixXd x_fixed(N, 1);

        // Quaternion components can flip when they pass through 180 degree
        // rotations, so let's convert all the quaternions in this channel to
        // the forward/up vector representation, modify them, then convert back
        // to quaternions:

        // convert time series to 6d forward/up:
        std::vector<float> forwardUp(6 * N);
        std::vector<float> targetForwardUp(6 * N);
        for (uint32_t frame = 0; frame < N; ++frame)
        {
            auto q = poses[frame][boneIdx].GetRotation();
            auto forward = q.ZAxis();
            auto up = q.YAxis();
            forwardUp[N * 0 + frame] = forward.GetX();
            forwardUp[N * 1 + frame] = forward.GetY();
            forwardUp[N * 2 + frame] = forward.GetZ();
            forwardUp[N * 3 + frame] = up.GetX();
            forwardUp[N * 4 + frame] = up.GetY();
            forwardUp[N * 5 + frame] = up.GetZ();

            q = targetPoses[frame][boneIdx].GetRotation();
            forward = q.ZAxis();
            up = q.YAxis();
            targetForwardUp[N * 0 + frame] = forward.GetX();
            targetForwardUp[N * 1 + frame] = forward.GetY();
            targetForwardUp[N * 2 + frame] = forward.GetZ();
            targetForwardUp[N * 3 + frame] = up.GetX();
            targetForwardUp[N * 4 + frame] = up.GetY();
            targetForwardUp[N * 5 + frame] = up.GetZ();
        }

        // correct trajectories:
        for (uint32_t dim = 0; dim < 6; ++dim)
        {
            for (uint32_t frame = 0; frame < N; ++frame)
            {
                x(frame, 0) = forwardUp[N * dim + frame];
                observations(frame, 0) = mask[frame] * targetForwardUp[N * dim + frame];
            }

            if (performChannelSmoothing)
            {
                SmoothChannels(x, mask);
            }

            corrector.Interpolate(
                x_fixed,
                observations,
                x
            );

            // fill channel again:
            for (uint32_t frame = 0; frame < N; ++frame)
            {
                forwardUp[N * dim + frame] = float(x_fixed(frame, 0));
            }
        }

        for (uint32_t frame = 0; frame < N; ++frame)
        {
            Math::Vector forward = { forwardUp[N * 0 + frame] ,forwardUp[N * 1 + frame] ,forwardUp[N * 2 + frame] };
            Math::Vector up = { forwardUp[N * 3 + frame] ,forwardUp[N * 4 + frame] ,forwardUp[N * 5 + frame] };

            forward.Normalize3();
            up.Normalize3();

            poses[frame][boneIdx].SetRotation(Math::Quaternion::LookRotation(forward, up));
        }
    }

    void CorrectJointRotations(
        std::vector<Pose>& poses,
        const std::vector<Pose>& targetPoses,
        const std::vector<float>& fullBodyMask,
        const Eigen::VectorXd& velocity_weights
    )
    {
        auto N = poses.size();

        // Create a trajectory corrector for fixing the full body fullBodyMask positions:
        Eigen::VectorXd margins(N);
        for( size_t i = 0; i < N; ++i )
        {
            margins[i] = fullBodyMask[i] ? 0.0f : -1.0f;
        }
        TrajectoryCorrector corrector(
            margins,
            pos_weight * 10,
            vel_weight,
            acc_weight,
            velocity_weights
        );

        for (uint32_t boneIdx = 0; boneIdx < poses[0].size(); ++boneIdx)
        {
            CorrectRotationsForBone(
                poses,
                targetPoses,
                fullBodyMask,
                corrector,
                boneIdx,
                true
            );
        }
    }

    void DoEffectorIK(
        std::vector<Pose>& poses,
        const std::vector<Pose>& targetPoses,
        const std::vector<float>& fullBodyMask,
        const std::vector<Animation::ContactInfo>& endEffectorPins,
        const std::vector<int>& joint_parents_vec,
        const std::vector<Math::Transform>& defaultPose
    )
    {
        // Apply IK for effector pins
        auto N = poses.size();
        std::map<uint32_t, std::vector<float>> jointCorrectionMasks;
        std::vector<Pose> ikFixedPoses = poses;
        for (auto& c : endEffectorPins)
        {
            auto jointIdx = c.jointIndex;

            if(jointCorrectionMasks[jointIdx].empty())
            {
                // initialize to the full body constraint mask because we
                // want to constrain that anyway:
                jointCorrectionMasks[jointIdx] = fullBodyMask;
            }

            // Add a trajectory correction mask for the parent joint:
            auto parentIdx = joint_parents_vec[jointIdx];
            if(jointCorrectionMasks[parentIdx].empty())
            {
                // initialize to the full body constraint mask because we
                // want to constrain that anyway:
                jointCorrectionMasks[parentIdx] = fullBodyMask;
            }

            // Add a trajectory correction mask for its parent if this is
            // 2 bone IK:
            auto parentParentIdx = joint_parents_vec[parentIdx];
            if(c.contactType == Animation::kTwoBone)
            {
                if(jointCorrectionMasks[parentParentIdx].empty())
                {
                    // initialize to the full body constraint mask because we
                    // want to constrain that anyway:
                    jointCorrectionMasks[parentParentIdx] = fullBodyMask;
                }
            }

            for (uint32_t fixFrame = 0; fixFrame < fullBodyMask.size(); ++fixFrame)
            {
                if (c.contactMask[fixFrame])
                {
                    const auto targetGlobalTransform = Animation::JointLocalToGlobal(joint_parents_vec, jointIdx, targetPoses[fixFrame]);

                    // flag the parent joint as fixed in its correction mask:
                    jointCorrectionMasks[parentIdx][fixFrame] = 1;
                    switch(c.contactType)
                    {
                        case Animation::kOneBone:
                        {
                            IK::OneBoneIk(
                                ikFixedPoses[fixFrame],
                                Math::Transform::Identity,
                                jointIdx,
                                1.0,
                                targetGlobalTransform.GetTranslation(),
                                joint_parents_vec
                            );
                            break;
                        }
                        case Animation::kTwoBone:
                        {
                            // flag the parent parent joint as fixed in its correction mask:
                            jointCorrectionMasks[parentParentIdx][fixFrame] = 1;
                            IK::TwoBoneIk(
                                ikFixedPoses[fixFrame],
                                Math::Transform::Identity,
                                jointIdx,
                                1.0,
                                targetGlobalTransform.GetTranslation(),
                                joint_parents_vec,
                                c.hintOffset
                            );
                            break;
                        }
                    }

                    // now we need to fix things so the global rotation of the joint
                    // matches the input:
                    jointCorrectionMasks[jointIdx][fixFrame] = 1;
                    auto parentGlobalTransform = Animation::JointLocalToGlobal(joint_parents_vec, parentIdx, ikFixedPoses[fixFrame]);
                    ikFixedPoses[fixFrame][jointIdx].SetRotation(
                        targetGlobalTransform.GetRotation() * parentGlobalTransform.GetRotation().GetConjugate()
                    );

                }
            }
        }

        // Applying the effector pin IK introduces popping into the animation,
        // so let's apply the interpolator to all the joints we modified so as to
        // line the trajectory up properly again:
        Eigen::VectorXd margins(N);
        for( auto &kv : jointCorrectionMasks)
        {
            for( size_t i = 0; i < N; ++i )
            {
                margins[i] = kv.second[i] ? 0.0f : -1.0f;
            }
            TrajectoryCorrector corrector(margins, pos_weight * 10, vel_weight, acc_weight);

            CorrectRotationsForBone(
                poses,
                ikFixedPoses,
                kv.second,
                corrector,
                kv.first,
                false
            );
        }
    }

}


Math::Transform Animation::JointLocalToGlobal(
    const std::vector<int>& joint_parents_vec,
    int32_t index,
    const Pose& localPose,
    const Math::Transform& rootTx)
{
    Math::Transform worldTx = Math::Transform::Identity;
    while (index > -1)
    {
        worldTx = worldTx * localPose[index];
        index = joint_parents_vec[index];
    }

    return worldTx * rootTx;
}

void Animation::CorrectMotion(
    std::vector<Pose>& poses,
    const std::vector<Pose>& targetPoses,
    const std::vector<float>& fullBodyMask,
    const std::vector<float>& rootMask,
    const std::vector<ContactInfo>& endEffectorPins,
    const std::vector<int>& joint_parents_vec,
    const std::vector<Math::Transform>& defaultPose,
    float root_margin
)
{

    // Calculate some weights so we can preserve velocities more strongly on frames where
    // the root velocity is low
    const uint32_t N = poses.size();
    Eigen::VectorXd velocity_weights(N);
    for (uint32_t frame = 1; frame < N; ++frame)
    {
        // work out xz velocity for this frame:
        float xdiff = poses[frame][0].GetTranslation()[0] - poses[frame - 1][0].GetTranslation()[0];
        float zdiff = poses[frame][0].GetTranslation()[2] - poses[frame - 1][0].GetTranslation()[2];

        // find velocity magnitude, divided by a typical walking speed:
        float v_mag = sqrtf(xdiff*xdiff + zdiff*zdiff) / 0.05f;

        // weight lower velocities higher so that the corrector doesn't make the character drift around
        // when it's supposed to stand still:
        v_mag = std::max(v_mag, 1.0f/1000.0f);
        velocity_weights(frame) = 1.0f / v_mag;
    }
    velocity_weights[0] = velocity_weights[1];

    // Correct root y coordinates.
    // This will warp the root y coordinates in "poses" so they match the root y coordinates
    // in "targetPoses", on frames where the root y coordinates are constrained, ie the frames
    // where fullBodyMask = 1.
    CorrectHipsY(
        poses,
        targetPoses,
        fullBodyMask
    );

    // Correct root xz coordinates:
    // This will warp the root xz coordinates in "poses" so they match the xz coordinates
    // in "targetPoses" on frames where fullBodyMask = 1, and warp them so they're within
    // "root_margin" units of targetPoses on frames where rootMask = 1.
    CorrectHipsXZ(
        poses,
        targetPoses,
        fullBodyMask,
        rootMask,
        endEffectorPins,
        velocity_weights,
        root_margin
    );

    // Correct joint rotations by warping the rotations so they match targetPoses on frames
    // where fullBodyMask = 1:
    CorrectJointRotations(
        poses,
        targetPoses,
        fullBodyMask,
        velocity_weights
    );

    // Apply IK for end effector pins
    DoEffectorIK(
        poses,
        targetPoses,
        fullBodyMask,
        endEffectorPins,
        joint_parents_vec,
        defaultPose
    );

    // Automatic contact enforcement belongs to Studio's native final evaluator.
}
