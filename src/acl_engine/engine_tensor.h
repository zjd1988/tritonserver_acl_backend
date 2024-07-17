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
#include "acl_engine/log.h"

namespace ACL_ENGINE
{

    class EngineTensor
    {
    public:
        enum TensorCopyKindType
        {
            TENSOR_COPY_HOST_TO_HOST     = 0,
            TENSOR_COPY_HOST_TO_DEVICE   = 1,
            TENSOR_COPY_DEVICE_TO_HOST   = 2,
            TENSOR_COPY_DEVICE_TO_DEVICE = 3,
        };
        /** dimension type used to create tensor */
        enum TensorDataType 
        {
            TENSOR_DATA_TYPE_VOID        = 0,
            TENSOR_DATA_TYPE_INT8        = 1,
            TENSOR_DATA_TYPE_UINT8       = 2,
            TENSOR_DATA_TYPE_INT16       = 3,
            TENSOR_DATA_TYPE_UINT16      = 4,
            TENSOR_DATA_TYPE_INT32       = 5,
            TENSOR_DATA_TYPE_UINT32      = 6,
            TENSOR_DATA_TYPE_INT64       = 7,
            TENSOR_DATA_TYPE_UINT64      = 8,
            TENSOR_DATA_TYPE_FLOAT16     = 9,
            TENSOR_DATA_TYPE_FLOAT32     = 10,
            TENSOR_DATA_TYPE_FLOAT64     = 11,
            TENSOR_DATA_TYPE_STRING      = 12,
            TENSOR_DATA_TYPE_MAX         ,
        };
        /** dimension type used to create tensor */
        enum TensorFormatType 
        {
            TENSOR_FORMAT_TYPE_NCHW        = 0,
            TENSOR_FORMAT_TYPE_NHWC        = 1,
            TENSOR_FORMAT_TYPE_MAX         ,
        };
    private:
        struct TensorBuffer 
        {
            std::vector<int64_t>    dim;
            TensorDataType          type;
            TensorFormatType        format;
            void*                   host;
            uint64_t                device;
            int                     device_id;
            bool                    own_flag = false;
            size_t elementBytes() const
            {
                if (type == TENSOR_DATA_TYPE_INT8 || type == TENSOR_DATA_TYPE_UINT8)
                    return sizeof(uint8_t);
                else if (type == TENSOR_DATA_TYPE_INT16 || type == TENSOR_DATA_TYPE_UINT16)
                    return sizeof(uint16_t);
                else if (type == TENSOR_DATA_TYPE_INT32 || type == TENSOR_DATA_TYPE_UINT32)
                    return sizeof(uint32_t);
                else if (type == TENSOR_DATA_TYPE_INT64 || type == TENSOR_DATA_TYPE_UINT64)
                    return sizeof(uint64_t);
                else if (type == TENSOR_DATA_TYPE_FLOAT32)
                    return sizeof(float);
                else if (type == TENSOR_DATA_TYPE_FLOAT64)
                    return sizeof(double);
                else if (type == TENSOR_DATA_TYPE_STRING)
                    return sizeof(void*);
                else
                {
                    ACL_LOG(ACL_LOG_LEVEL_FATAL, "unspported data type:{}.", int(type));
                    return -1;
                }
            }
        };

    public:
        /**
         * @brief create a tensor with same shape as given tensor.
         * @param tensor        shape provider.
         * @param type          data type.
         * @param format        data format.
         * @param alloc_host    acquire host memory for data or not.
         * @param alloc_device  acquire device memory for data or not.
         * @warning tensor data won't be copied.
         */
        EngineTensor(const std::vector<int64_t>& dims, TensorDataType type, 
            TensorFormatType format = TENSOR_FORMAT_TYPE_NCHW, bool alloc_host = true, 
            bool alloc_device = false, int device_id = -1);

        /**
         * @brief create a tensor with same shape as given tensor.
         * @param tensor        shape provider.
         * @param type          data type.
         * @param alloc_host    acquire host memory for data or not.
         * @param alloc_device  acquire device memory for data or not.
         * @warning tensor data won't be copied.
         */
        EngineTensor(const std::vector<int64_t>& dims, TensorDataType type, bool alloc_host = true, 
            bool alloc_device = false, int device_id = -1);

        /** deinitializer */
        ~EngineTensor();

    private:
        // remove all assignment operator
        EngineTensor(const EngineTensor& tensor)  = delete;
        EngineTensor(const EngineTensor&& tensor) = delete;
        EngineTensor& operator=(const EngineTensor&) = delete;
        EngineTensor& operator=(const EngineTensor&&) = delete;

    public:
        /**
         * @brief create tensor with shape, data type and dimension type.
         * @param shape     tensor shape.
         * @param type      data type.
         * @return created tensor.
         * @warning memory for data won't be acquired. call backend's onAcquireBuffer to get memory ready.
         */
        static EngineTensor* createDevice(const std::vector<int64_t>& shape, TensorDataType type, 
            TensorFormatType format = TENSOR_FORMAT_TYPE_NCHW, void* device_data = NULL, int device_id = -1);

        /**
         * @brief create tensor with shape, data type, data and dimension type.
         * @param shape     tensor shape.
         * @param type      data type.
         * @param data      data to save.
         * @return created tensor.
         */
        static EngineTensor* create(const std::vector<int64_t>& shape, TensorDataType type, 
            TensorFormatType format = TENSOR_FORMAT_TYPE_NCHW, void* host_data = NULL);

        /**
         * @brief copy a new tensor have same shape, data type, data(or data ptr) and dimension type
         * @param tensor, origin tensor to be copy
         * @return copyed tensor.
         */
        static EngineTensor* copy(EngineTensor* tensor);

        /**
         * @brief clone a new tensor have same shape, data type, data(or data ptr) and dimension type
         * @param tensor, origin tensor to be cloned
         * @param deep_copy, copy data or not
         * @return cloned tensor.
         */
        static EngineTensor* clone(EngineTensor* tensor, bool deep_copy = false);

        /**
         * @brief create HOST tensor from DEVICE tensor, with or without data copying.
         * @param deviceTensor  given device tensor.
         * @param copyData      copy data or not.
         * @return created host tensor.
         */
        static EngineTensor* createHostTensorFromDevice(const EngineTensor* device_tensor, bool copy_data = true);

    public:
        const TensorBuffer& buffer() const
        {
            return m_buffer;
        }

        TensorBuffer& buffer()
        {
            return m_buffer;
        }

        /**
         * @brief get tensor data type.
         * @return data type.
         */
        TensorDataType getTensorDataType() const;

        /**
         * @brief get tensor format type.
         * @return format type.
         */
        TensorFormatType getTensorFormatType() const;

        /**
         * @brief visit host memory, data type is represented by `T`.
         * @return data point in `T` type.
         */
        template <typename T>
        T* host() const {
            return (T*)m_buffer.host;
        }

        /**
         * @brief visit device memory.
         * @return device data ptr. what the ptr means varies between backends.
         */
        uint64_t devicePtr() const 
        {
            return m_buffer.device;
        }

        /**
         * @brief visit device id.
         * @return device id. what the id means device memory locate in which device.
         */
        uint64_t deviceId() const 
        {
            return m_buffer.device_id;
        }

        /**
         * @brief reshape tensor to a new valid dims
         * @param new_shape, new dims
         * @return 0 success, else fail
         */
        int reshape(const std::vector<int64_t> new_shape);

        /**
         * @brief for DEVICE tensor, copy data from given host tensor.
         * @param hostTensor host tensor, the data provider.
         * @return true for DEVICE tensor, and false for HOST tensor.
         */
        int copyFromHostTensor(const EngineTensor* hostTensor);

        /**
         * @brief for DEVICE tensor, copy data to given host tensor.
         * @param hostTensor host tensor, the data consumer.
         * @return true for DEVICE tensor, and false for HOST tensor.
         */
        int copyToHostTensor(EngineTensor* hostTensor) const;

    public:
        int dimensions() const 
        {
            return m_buffer.dim.size();
        }

        /**
         * @brief get all dimensions' extent.
         * @return dimensions' extent.
         */
        std::vector<int64_t> shape() const;

        /**
         * @brief calculate number of bytes needed to store data taking reordering flag into account.
         * @return bytes needed to store data
         */
        int size() const;

        /**
         * @brief calculate number of elements needed to store data taking reordering flag into account.
         * @return elements needed to store data
         */
        inline int elementSize() const
        {
            return size() / m_buffer.elementBytes();
        }

    public:
        /**
         * @brief print tensor data. for DEBUG use only.
         */
        void print() const;
        
        /**
         *@brief print tensor shape
        */
        void printShape() const;

    private:
        TensorBuffer m_buffer;
    };

    typedef struct EngineTensorInfo
    {
        std::string                               name;
        EngineTensor::TensorDataType              type;
        std::vector<int64_t>                      shape;
    } EngineTensorInfo;

} // namespace ACL_ENGINE