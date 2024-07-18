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
#include <utility>
#include "acl_engine/log.h"
#include "acl_engine/dyn_shape_process.h"

namespace ACL_ENGINE
{

    namespace
    {
        constexpr auto kInputDimNum = 4;
        constexpr auto kNHWCNIdx = 0;
        constexpr auto kNHWCHeightIdx = 1;
        constexpr auto kNHWCWidthIdx = 2;
        constexpr auto kNHWCCIdx = 3;
        constexpr auto kNCHWNIdx = 0;
        constexpr auto kNCHWCIdx = 1;
        constexpr auto kNCHWHeightIdx = 2;
        constexpr auto kNCHWWidthIdx = 3;
        constexpr auto kImageSizeHwNum = 2;
        constexpr auto kUnknownDim = -1;
    }  // namespace

    bool DynShapeProcess::Init(const AclDynamicShapeOptions &options)
    {
        acl_options_ = options;
        for (size_t i = 0; i < options.input_shapes.size(); i++)
        {
            auto &shape = options.input_shapes[i];
            if (std::any_of(shape.begin(), shape.end(), [](auto dim) { return dim < 0; }))
            {
                input_data_idx_ = i;
                break;
            }
        }
        if (input_data_idx_ >= acl_options_.input_shapes.size())
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "input data index {} is invalid, inputs count: {}", input_data_idx_, 
                acl_options_.input_shapes.size());
            return false;
        }
        return true;
    }

    bool DynShapeProcess::CheckAndGetBatchSize(const std::vector<std::vector<int64_t>> &new_shapes, int32_t *batch_size)
    {
        if (acl_options_.batch_size.empty())
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "not support dynamic batch size");
            return false;
        }
        if (batch_size == nullptr)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "input parameter batch size cannot be nullptr");
            return false;
        }
        if (!CheckBatchSize(new_shapes))
        {
            return false;
        }
        return GetRealBatchSize(new_shapes, batch_size);
    }

    bool DynShapeProcess::CheckAndGetDynamicDims(const std::vector<std::vector<int64_t>> &new_shapes, aclmdlIODims *dynamic_dims)
    {
        if (dynamic_dims == nullptr)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "input parameter dynamic dims cannot be nullptr");
            return false;
        }
        if (!CheckDynamicDims(new_shapes))
        {
            return false;
        }
        return GetRealDynamicDims(new_shapes, dynamic_dims);
    }

    bool DynShapeProcess::CheckAndGetImageSize(const std::vector<std::vector<int64_t>> &new_shapes, int32_t *height, int32_t *width)
    {
        if (acl_options_.image_size.empty())
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "not support image batch size");
            return false;
        }
        if (height == nullptr || width == nullptr)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "input parameter image size cannot be nullptr");
            return false;
        }
        if (!CheckImageSize(new_shapes))
        {
            return false;
        }
        return GetRealImageSize(new_shapes, height, width);
    }

    bool DynShapeProcess::CheckBatchSize(const std::vector<std::vector<int64_t>> &new_shapes)
    {
        if (input_data_idx_ >= new_shapes.size())
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "input data index {} is larger than input size {}", input_data_idx_, 
                new_shapes.size());
            return false;
        }
        std::vector<int64_t> original_shape = acl_options_.input_shapes[input_data_idx_];
        std::vector<int64_t> cur_shape = new_shapes[input_data_idx_];
        if (cur_shape.empty() || original_shape.empty())
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "shape is empty, input index = {}", input_data_idx_);
            return false;
        }
        if (cur_shape.size() != original_shape.size())
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "curr shape size {} is not equal with original shape size {}", cur_shape.size(), 
                original_shape.size());
            return false;
        }
        for (size_t i = 1; i < cur_shape.size(); ++i)
        {
            if (cur_shape[i] <= 0)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "invalid new shape {} for input {}", cur_shape, i);
                return false;
            }
            if (original_shape[i] != kUnknownDim && (original_shape[i] != cur_shape[i]))
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "shape conflict: original shape:{} current shape:{}", 
                    spdlog::fmt_lib::join(original_shape, ", "), spdlog::fmt_lib::join(cur_shape, ", "));
                return false;
            }
        }
        return true;
    }

    bool DynShapeProcess::CheckDynamicDims(const std::vector<std::vector<int64_t>> &new_shapes)
    {
        std::vector<std::vector<int64_t>> original_shapes = acl_options_.input_shapes;
        if (original_shapes.size() != new_shapes.size() || new_shapes.empty())
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "new shape size is: {} not equal original shapes size: {}", 
                new_shapes.size(), original_shapes.size());
            return false;
        }
        for (size_t i = 0; i < new_shapes.size(); i++)
        {
            if (new_shapes[i].size() != original_shapes[i].size())
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "new shapes[{}] size:{}, not equal to original shapes[{}] size: {}", 
                    i, new_shapes[i].size(), i, original_shapes[i].size());
                return false;
            }
            for (size_t j = 0; j < new_shapes[i].size(); j++)
            {
                if (new_shapes[i][j] != original_shapes[i][j] && original_shapes[i][j] != -1)
                {
                    ACL_LOG(ACL_LOG_LEVEL_ERROR, "input shape is wrong");
                    return false;
                }
            }
        }

        return true;
    }

    bool DynShapeProcess::CheckImageSize(const std::vector<std::vector<int64_t>> &new_shapes)
    {
        if (input_data_idx_ >= new_shapes.size() || input_data_idx_ >= acl_options_.input_format.size())
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "input data index {} is invalid, inputs size {} input formats size {}", 
                input_data_idx_, new_shapes.size(), acl_options_.input_format.size());
            return false;
        }
        std::vector<int64_t> original_shape = acl_options_.input_shapes[input_data_idx_];
        std::vector<int64_t> cur_shape = new_shapes[input_data_idx_];
        if (original_shape.size() != kInputDimNum)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "shape size {} is invalid, input index = {}", original_shape.size(), 
                input_data_idx_);
            return false;
        }
        if (cur_shape.size() != original_shape.size())
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "curr shape size {} is not equal with original shape size {}", cur_shape.size(), 
                original_shape.size());
            return false;
        }
        for (size_t i = 1; i < cur_shape.size(); ++i)
        {
            if (cur_shape[i] <= 0)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "invalid new shape {} for input {}", cur_shape, i);
                return false;
            }
            if (original_shape[i] != kUnknownDim && (original_shape[i] != cur_shape[i]))
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "shape Conflict: original shape:{}, current shape:{}", 
                    spdlog::fmt_lib::join(original_shape, ", "), spdlog::fmt_lib::join(cur_shape, ", "));
                return false;
            }
        }
        auto format = acl_options_.input_format[input_data_idx_];
        if (format == mindspore::Format::NHWC)
        {
            if ((original_shape[kNHWCCIdx] != kUnknownDim && 
                (original_shape[kNHWCCIdx] != cur_shape[kNHWCCIdx])) ||
                (original_shape[kNHWCNIdx] != kUnknownDim && 
                (original_shape[kNHWCNIdx] != cur_shape[kNHWCNIdx])))
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "shape Conflict: original shape:{}, current shape:{}", 
                    spdlog::fmt_lib::join(original_shape, ", "), spdlog::fmt_lib::join(cur_shape, ", "));
                return false;
            }
        }
        else
        {
            if ((original_shape[kNCHWCIdx] != kUnknownDim && 
                (original_shape[kNCHWCIdx] != cur_shape[kNCHWCIdx])) ||
                (original_shape[kNCHWNIdx] != kUnknownDim && 
                (original_shape[kNCHWNIdx] != cur_shape[kNCHWNIdx])))
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "shape conflict: original shape:{}, current shape:{}", 
                    spdlog::fmt_lib::join(original_shape, ", "), spdlog::fmt_lib::join(cur_shape, ", "));
                return false;
            }
        }
        return true;
    }

    bool DynShapeProcess::GetRealBatchSize(const std::vector<std::vector<int64_t>> &new_shapes, int32_t *batch_size)
    {
        if (input_data_idx_ >= new_shapes.size())
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, " input data index {} is larger than input size {}", input_data_idx_ , 
                new_shapes.size());
            return false;
        }
        std::vector<int64_t> shape = new_shapes[input_data_idx_];
        if (shape.empty())
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "shape is empty, input index = {}", input_data_idx_);
            return false;
        }
        int32_t cur_batch_size = static_cast<uint64_t>(shape[0]);
        auto iter = acl_options_.batch_size.find(cur_batch_size);
        if (iter == acl_options_.batch_size.end())
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "current batch size {} is invalid, please check device info of context", 
                cur_batch_size);
            return false;
        }
        *batch_size = cur_batch_size;
        ACL_LOG(ACL_LOG_LEVEL_DEBUG, "current batch size {}", cur_batch_size);
        return true;
    }

    bool DynShapeProcess::GetRealDynamicDims(const std::vector<std::vector<int64_t>> &new_shapes, aclmdlIODims *dynamic_dims)
    {
        if (input_data_idx_ >= new_shapes.size())
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "input data index {} is larger than input size {}", input_data_idx_, 
                new_shapes.size());
            return false;
        }
        std::vector<int64_t> dims;
        for (auto shape : new_shapes)
        {
            for (auto dim : shape)
            {
                ACL_LOG(ACL_LOG_LEVEL_DEBUG, "input shape dim: {}", dim);
                dims.push_back(dim);
            }
        }
        dynamic_dims->dimCount = dims.size();
        for (size_t i = 0; i < dims.size(); i++)
        {
            ACL_LOG(ACL_LOG_LEVEL_DEBUG, "dynamic dim: {}", dims[i]);
            dynamic_dims->dims[i] = dims[i];
        }
        return true;
    }

    bool DynShapeProcess::GetRealImageSize(const std::vector<std::vector<int64_t>> &new_shapes, int32_t *height_p, int32_t *width_p)
    {
        if (input_data_idx_ >= new_shapes.size() || input_data_idx_ >= acl_options_.input_format.size())
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "input data index {} is invalid, inputs size {} input formats size {}", 
                input_data_idx_, new_shapes.size(), acl_options_.input_format.size());
            return false;
        }
        std::vector<int64_t> shape = new_shapes[input_data_idx_];
        if (shape.size() != kInputDimNum)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "shape size {} is invalid, input index = {}", shape.size(), input_data_idx_);
            return false;
        }
        auto format = acl_options_.input_format[input_data_idx_];
        int64_t height;
        int64_t width;
        if (format == mindspore::Format::NHWC)
        {
            height = shape[kNHWCHeightIdx];
            width = shape[kNHWCWidthIdx];
        }
        else
        {
            height = shape[kNCHWHeightIdx];
            width = shape[kNCHWWidthIdx];
        }
        auto cur_image_size = std::pair<int32_t, int32_t>(static_cast<int32_t>(height), static_cast<int32_t>(width));
        auto iter = acl_options_.image_size.find(cur_image_size);
        if (iter == acl_options_.image_size.end())
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "image size height {}, width is invalid, please check device info of context", 
                height, width);
            return false;
        }
        *height_p = LongToInt(height);
        *width_p = LongToInt(width);
        ACL_LOG(ACL_LOG_LEVEL_DEBUG, "current height {} width {}", height, width);
        return true;
    }

}  // namespace ACL_ENGINE