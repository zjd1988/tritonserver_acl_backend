/********************************************
 * @Author: zhaojd-a
 * @Date: 2024-06-13
 * @LastEditTime: 2024-06-13
 * @LastEditors: zhaojd-a
 ********************************************/
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <vector>
#include "acl_engine/engine_tensor.h"

namespace ACL_ENGINE
{

    /** tensor utils */
    class TensorUtils
    {
    public:

        /**
         * @brief reshape tensor to a new shape
         * @param origin          tensor.
         * @param new_shape       new shape info.
         */
        static int reshapeTensor(EngineTensor* origin, const std::vector<int64_t>& new_shape);

        /**
         * @brief compare tensor to expected with tolerance.
         * @param compareTensor comparing tensor.
         * @param toTensor      expected tensor.
         * @param tolerance     tolerable error, any error less than this value will be ignored.
         *                      for integer types, compare with `abs(v1 - v2) > tolerance`;
         *                      for float types, see `overallTolerance`.
         * @param overall       for float types only. compare with `abs(v1 - v2) / max(abs(allExpectValues))` if true,
         *                      `abs(v1 - v2) / abs(v2)` otherwise.
         * @param printsError   print error data or not.
         * @return equals within tolerance or not.
         */
        static bool compareTensors(const EngineTensor* compareTensor, const EngineTensor* toTensor, float tolerance = 0,
            bool overall = false, bool printsErrors = true);

        /**
         * @brief calculate number of bytes of string tensor.
         * @return 0:success, -1:fail
        */
        static int getStringTensorByteSize(const EngineTensor* tensor, size_t& bytes_num);

        /**
         * @brief get string tensor content.
         * @param tensor, tensor ptr
         * @param buffer, to store all string buffer
         * @param buffer_length, all string buffer len
         * @param offsets, sub string buffer offset 
         * @param offsets_count, number of sub string
         * @return 0:success, -1:fail
        */
        static int getStringTensorContent(const EngineTensor* tensor, char* buffer, size_t buffer_length, 
            size_t* offsets, size_t offsets_count);

        /**
         * @brief set string tensor content.
         * @param tensor, tensor ptr
         * @param string_ptrs, buffer sotre string ptrs
         * @param string_size, string ptrs size
         * @return 0:success, -1:fail
        */
        static int setStringTensorContent(const EngineTensor* tensor, const char* const* string_ptrs, size_t ptrs_size);

    };

} // namespace ACL_ENGINE