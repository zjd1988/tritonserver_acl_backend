/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/********************************************
 * @Author: zhaojd-a
 * @Date: 2024-06-13
 * @LastEditTime: 2024-06-13
 * @LastEditors: zhaojd-a
 ********************************************/
#pragma once
#include <vector>
#include <memory>
#include <string>
#include "acl/acl.h"
#include "acl/acl_mdl.h"
#include "acl/acl_rt.h"
#include "acl_engine/non_copyable.h"

namespace ACL_ENGINE
{

    typedef struct AclDynamicShapeOptions
    {
        std::set<uint64_t>                              batch_size;
        std::set<std::pair<uint64_t, uint64_t>>         image_size;
        std::pair<aclmdlIODims*, size_t>                dynamic_dims;
        std::vector<Format>                             input_format;
        std::vector<std::vector<int64_t>>               input_shapes;
    } AclDynamicShapeOptions;

    class DynShapeProcess : NonCopyable
    {
    public:
        bool Init(const AclDynamicShapeOptions &options);
        bool CheckAndGetBatchSize(const std::vector<std::vector<int64_t>> &new_shapes, int32_t *batch_size);
        bool CheckAndGetDynamicDims(const std::vector<std::vector<int64_t>> &new_shapes, aclmdlIODims *dynamic_dims);
        bool CheckAndGetImageSize(const std::vector<std::vector<int64_t>> &new_shapes, int32_t *height, int32_t *width);

    private:
        bool CheckBatchSize(const std::vector<std::vector<int64_t>> &new_shapes);
        bool CheckDynamicDims(const std::vector<std::vector<int64_t>> &new_shapes);
        bool CheckImageSize(const std::vector<std::vector<int64_t>> &new_shapes);
        bool GetRealBatchSize(const std::vector<std::vector<int64_t>> &new_shapes, int32_t *batch_size);
        bool GetRealDynamicDims(const std::vector<std::vector<int64_t>> &new_shapes, aclmdlIODims *dynamic_dims);
        bool GetRealImageSize(const std::vector<std::vector<int64_t>> &new_shapes, int32_t *height, int32_t *width);

        AclDynamicShapeOptions acl_options_;
        size_t input_data_idx_ = 0;
    };

    using DynShapeProcPtr = std::shared_ptr<DynShapeProcess>;

}  // namespace ACL_ENGINE