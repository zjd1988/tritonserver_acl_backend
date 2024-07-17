/********************************************
 * @Author: zhaojd-a
 * @Date: 2024-06-13
 * @LastEditTime: 2024-06-13
 * @LastEditors: zhaojd-a
 ********************************************/
#include <float.h>
#include "acl_engine/log.h"
#include "acl_engine/engine_tensor_utils.h"

namespace ACL_ENGINE
{

    int TensorUtils::reshapeTensor(EngineTensor* origin, const std::vector<int64_t>& new_shape)
    {
        if (nullptr == origin)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "input tensor is nullptr");
            return -1;
        }
        return origin->reshape(new_shape);
    }

    static const EngineTensor* createHostPlanar(const EngineTensor* source)
    {
        // check
        bool device = source->host<void>() ? true : false;
        // no convert needed
        if (!device)
        {
            return source;
        }
        // convert
        return source->createHostTensorFromDevice(source, true);
    }

    static bool equals(const double* pa, const double* pb, size_t size, double tolerance, double epsilon, bool overall, bool prints) 
    {
        // get max if using overall torelance
        double max = fabs(pb[0]);
        if (overall)
        {
            for (size_t i = 1; i < size; i++)
            {
                max = std::max(max, fabs(pb[i]));
            }
        }

        // compare
        for (size_t i = 0; i < size; i++)
        {
            float va = pa[i], vb = pb[i];
            if (std::isinf(va) && std::isinf(vb))
            {
                continue;
            }
            if (fabs(va) < epsilon && fabs(vb) < epsilon)
            {
                continue;
            }
            float div = overall ? max : fabsf(vb);
            if (fabsf(va - vb) / div > tolerance)
            {
                if (prints)
                {
                    ACL_LOG(ACL_LOG_LEVEL_ERROR, "{}: {} != {}", i, va, vb);
                }
                return false;
            }
        }
        return true;
    }

    template <typename T>
    static void copyTensorToFloat(const EngineTensor* source, double* dest)
    {
        auto srcData = source->host<T>();
        auto size    = source->elementSize();
        for (int i = 0; i < size; ++i)
        {
            dest[i] = srcData[i];
        }
    }

    bool TensorUtils::compareTensors(const EngineTensor* compareTensor, const EngineTensor* toTensor, float tolerance,
        bool overall, bool printsErrors)
    {
        // type
        if (compareTensor->getTensorDataType() != toTensor->getTensorDataType())
        {
            if (printsErrors)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "not equal in type: {} - {}", 
                    int(compareTensor->getTensorDataType()), int(toTensor->getTensorDataType()));
            }
            return false;
        }

        // dimensions
        auto compare_dims = compareTensor->dimensions();
        auto to_dims = toTensor->dimensions();
        if (compare_dims != to_dims)
        {
            if (printsErrors)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "NOT equal in dimensions: {} - {}", 
                compareTensor->dimensions(), toTensor->dimensions());
            }
            return false;
        }
        auto compare_shape = compareTensor->shape();
        auto to_shape = toTensor->shape();
        for (size_t i = 0; i < compare_shape.size(); i++)
        {
            if (compare_shape[i] == to_shape[i]) 
            {
                continue;
            }
            if (printsErrors)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "NOT equal in dimensions[{}]: {} - {}", i, 
                    compare_shape[i], to_shape[i]);
            }
            return false;
        }

        // convert to host if needed
        auto a = createHostPlanar(compareTensor), b = createHostPlanar(toTensor);

        // get value as double
        auto size = toTensor->elementSize();
        std::vector<double> expectValue(toTensor->elementSize(), 0.0f);
        std::vector<double> compareValue(compareTensor->elementSize(), 0.0f);

        auto result = false;
        auto tensor_type = compareTensor->getTensorDataType();
        switch (tensor_type)
        {
            case EngineTensor::TENSOR_DATA_TYPE_UINT8:
                copyTensorToFloat<uint8_t>(a, compareValue.data());
                copyTensorToFloat<uint8_t>(b, expectValue.data());
                break;
            case EngineTensor::TENSOR_DATA_TYPE_UINT16:
                copyTensorToFloat<uint16_t>(a, compareValue.data());
                copyTensorToFloat<uint16_t>(b, expectValue.data());
                break;
            case EngineTensor::TENSOR_DATA_TYPE_UINT32:
                copyTensorToFloat<uint32_t>(a, compareValue.data());
                copyTensorToFloat<uint32_t>(b, expectValue.data());
                break;
            case EngineTensor::TENSOR_DATA_TYPE_UINT64:
                copyTensorToFloat<uint64_t>(a, compareValue.data());
                copyTensorToFloat<uint64_t>(b, expectValue.data());
                break;
            case EngineTensor::TENSOR_DATA_TYPE_INT8:
                copyTensorToFloat<int8_t>(a, compareValue.data());
                copyTensorToFloat<int8_t>(b, expectValue.data());
                break;
            case EngineTensor::TENSOR_DATA_TYPE_INT16:
                copyTensorToFloat<int16_t>(a, compareValue.data());
                copyTensorToFloat<int16_t>(b, expectValue.data());
                break;
            case EngineTensor::TENSOR_DATA_TYPE_INT32:
                copyTensorToFloat<int32_t>(a, compareValue.data());
                copyTensorToFloat<int32_t>(b, expectValue.data());
                break;
            case EngineTensor::TENSOR_DATA_TYPE_INT64:
                copyTensorToFloat<int64_t>(a, compareValue.data());
                copyTensorToFloat<int64_t>(b, expectValue.data());
                break;
            case EngineTensor::TENSOR_DATA_TYPE_FLOAT32:
                copyTensorToFloat<float>(a, compareValue.data());
                copyTensorToFloat<float>(b, expectValue.data());
                break;
            default:
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "unsupported data type");
                break;
        }
        auto epsilon = FLT_EPSILON;
        if ((NULL != compareValue.data()) && (NULL != expectValue.data()))
        {
            result = equals(compareValue.data(), expectValue.data(), size, tolerance, epsilon, overall, printsErrors);
        }
        return result;
    }

    int TensorUtils::getStringTensorByteSize(const EngineTensor* tensor, size_t& bytes_num)
    {
        if (nullptr == tensor)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "input tensor is nullptr");
            return -1;
        }
        if (EngineTensor::TENSOR_DATA_TYPE_STRING != tensor->getTensorDataType())
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "input tensor is not a string tensor");
            return -1;
        }
        char** string_buffer = tensor->host<char*>();
        int buffer_count = tensor->elementSize();
        int byte_sum = 0;
        for (int i = 0; i < buffer_count; i++)
        {
            char* str_ptr = string_buffer[i];
            if (nullptr == str_ptr)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "string tensor get buffer size check fail, str buffer is nullptr");
                return -1;
            }
            int str_len = strlen(str_ptr);
            if (0 > str_len || 0 > (byte_sum + str_len))
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "string tensor get buffer size check fail, byte_sum:{} str_len:{}", 
                    byte_sum, str_len);
                return -1;
            }
            byte_sum += str_len;
        }
        bytes_num = (size_t)byte_sum;
        return 0;
    }

    int TensorUtils::getStringTensorContent(const EngineTensor* tensor, char* buffer, size_t buffer_length, 
        size_t* offsets, size_t offsets_count)
    {
        if (nullptr == tensor)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "input tensor is nullptr");
            return -1;
        }
        if (EngineTensor::TENSOR_DATA_TYPE_STRING != tensor->getTensorDataType())
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "input tensor is not a string tensor");
            return -1;
        }
        size_t string_bytes_num = 0;
        if (0 != getStringTensorByteSize(tensor, string_bytes_num))
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "get string tensor byte size fail");
            return -1;
        }
        if (buffer_length != string_bytes_num)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "string tensor byte size {} not equal to input buffer lenght {}",
                string_bytes_num, buffer_length);
            return -1;
        }
        auto element_count = (size_t)tensor->elementSize();
        if (offsets_count != element_count)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "string tensor element count {} not equal to input offsets count {}",
                element_count, offsets_count);
            return -1;
        }
        char** string_buffer = tensor->host<char*>();
        int offset_sum = 0;
        for (size_t i = 0; i < offsets_count; i++)
        {
            char* str_ptr = string_buffer[i];
            int str_len = strlen(str_ptr);
            offsets[i] = offset_sum;
            if (size_t(offset_sum) > buffer_length || 
                0 > (offset_sum + str_len) || 
                size_t(offset_sum + str_len) > buffer_length)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "string tensor copy to buffer check fail, offset_sum:{} str_len:{} buffer_length:{}",
                    offset_sum, str_len, buffer_length);
                return -1;
            }
            memcpy(buffer + offset_sum, str_ptr, str_len);
            offset_sum += str_len;
        }
        return 0;
    }

    int TensorUtils::setStringTensorContent(const EngineTensor* tensor, const char* const* string_ptrs, size_t ptrs_size)
    {
        if (nullptr == tensor)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "input tensor is nullptr");
            return -1;
        }
        if (EngineTensor::TENSOR_DATA_TYPE_STRING != tensor->getTensorDataType())
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "input tensor is not a string tensor");
            return -1;
        }
        if (true != tensor->buffer().own_flag)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "input tensor own_flag is false");
            return -1;
        }
        auto element_count = (size_t)tensor->elementSize();
        if (ptrs_size != element_count)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "string tensor element count {} not equal to input ptrs size {}", 
                element_count, ptrs_size);
            return -1;
        }

        const char** string_buffer = tensor->host<const char*>();
        for (size_t i = 0; i < ptrs_size; i++)
        {
            string_buffer[i] = string_ptrs[i];
        }
        return 0;
    }

} // namespace ACL_ENGINE