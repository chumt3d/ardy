/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "Math/Transform.h"

#include <string>
#include <vector>

namespace Animation
{
    enum IKType {
        kOneBone,
        kTwoBone
    };

    Math::Transform JointLocalToGlobal(
        const std::vector<int>& joint_parents_vec,
        int32_t index,
        const std::vector<Math::Transform>& localPose,
        const Math::Transform& rootTx = Math::Transform::Identity
    );

    struct ContactInfo {
        // index IK contact joint:
        int jointIndex;
        // mask indicating which frames are in contact:
        std::vector<float> contactMask;
        // contact type:
        IKType contactType = kTwoBone;

        // Extra info for TwoBoneIK
        Math::Vector hintOffset = Math::Vector::Zero;
    };

    void CorrectMotion(
        std::vector< std::vector<Math::Transform> >& poses,
        const std::vector< std::vector<Math::Transform> >& targetPoses,
        const std::vector<float>& mask,
        const std::vector<float>& rootMask,
        const std::vector<ContactInfo>& endEffectorPins,
        const std::vector<int>& joint_parents_vec,
        const std::vector<Math::Transform>& defaultPose,
        float root_margin
    );
}
