/********************************************
 * @Author: zhaojd-a
 * @Date: 2024-06-13
 * @LastEditTime: 2024-06-13
 * @LastEditors: zhaojd-a
 ********************************************/
#include <string.h>
#include <numeric>
#include "acl_engine/engine_tensor.h"
#include "acl_engine/engine_memory_utils.h"
#ifdef ENGINE_SUPPORT_CUDA
#include <cuda_runtime.h>
#endif

namespace ACL_ENGINE
{

    static void* mallocDeviceMem(int mem_size, int& device_id)
    {
        void* device_ptr = nullptr;
#ifdef ENGINE_SUPPORT_CUDA
        int current_device = -1;
        cudaError_t cuda_ret = cudaGetDevice(&current_device);
        if (cuda_ret != cudaSuccess || -1 == current_device)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "get current device id failed with {}", cudaGetErrorString(cudaGetLastError()));
            return nullptr;
        }
        if (-1 == device_id)
            device_id = current_device;
        if (device_id != current_device)
        {
            cuda_ret = cudaSetDevice(device_id);
            if (cuda_ret != cudaSuccess)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "set current device id to {} failed with {}", 
                    device_id, cudaGetErrorString(cudaGetLastError()));
                return nullptr;
            }
        }
        cuda_ret = cudaMalloc(&device_ptr, mem_size);
        if (cuda_ret != cudaSuccess || nullptr == device_ptr)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "malloc device tensor buffer failed with {}", cudaGetErrorString(cudaGetLastError()));
            return nullptr;
        }
        if (device_id != current_device)
        {
            cuda_ret = cudaSetDevice(current_device);
            if (cuda_ret != cudaSuccess)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "reset current device id to {} failed with {}", 
                    current_device, cudaGetErrorString(cudaGetLastError()));
                cuda_ret = cudaFree(device_ptr);
                if (cuda_ret != cudaSuccess)
                {
                    ACL_LOG(ACL_LOG_LEVEL_FATAL, "free device mem fail with {}", 
                        cudaGetErrorString(cudaGetLastError()));
                }
                return nullptr;
            }
        }
#else
        ACL_LOG(ACL_LOG_LEVEL_ERROR, "please rebuild with ENGINE_SUPPORT_CUDA");
#endif
        return device_ptr;
    }

    static void freeDeviceMem(void* device_ptr, int& device_id)
    {
#ifdef ENGINE_SUPPORT_CUDA
        int current_device = -1;
        cudaError_t cuda_ret = cudaGetDevice(&current_device);
        if (cuda_ret != cudaSuccess || -1 == current_device)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "get current device id failed with {}", cudaGetErrorString(cudaGetLastError()));
            return;
        }
        if (-1 == device_id)
            device_id = current_device;
        if (device_id != current_device)
        {
            cuda_ret = cudaSetDevice(device_id);
            if (cuda_ret != cudaSuccess)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "set current device id to {} failed with {}", 
                    device_id, cudaGetErrorString(cudaGetLastError()));
                return;
            }
        }
        cuda_ret = cudaFree(device_ptr);
        if (cuda_ret != cudaSuccess)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "free device tensor buffer failed with {}", cudaGetErrorString(cudaGetLastError()));
            return;
        }
        if (device_id != current_device)
        {
            cuda_ret = cudaSetDevice(current_device);
            if (cuda_ret != cudaSuccess)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "reset current device id to {} failed with {}", 
                    current_device, cudaGetErrorString(cudaGetLastError()));
                return;
            }
        }
#else
        ACL_LOG(ACL_LOG_LEVEL_ERROR, "please rebuild with ENGINE_SUPPORT_CUDA");
#endif
        return;
    }

    static void* memcpyDeviceMem(void* dst_ptr, void* src_ptr, int mem_size, 
        int device_id, EngineTensor::TensorCopyKindType kind = EngineTensor::TENSOR_COPY_HOST_TO_HOST)
    {
        if (nullptr == src_ptr)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "input src_ptr param is nullptr, please check");
            return nullptr;
        }
        if (nullptr == dst_ptr)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "input dst_ptr param is nullptr, please check");
            return nullptr;
        }
        if (EngineTensor::TENSOR_COPY_HOST_TO_HOST == kind)
        {
            return memcpy(dst_ptr, src_ptr, mem_size);
        }
#ifdef ENGINE_SUPPORT_CUDA
        int current_device = -1;
        cudaError_t cuda_ret = cudaGetDevice(&current_device);
        if (cuda_ret != cudaSuccess || -1 == current_device)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "get current device id failed with {}", cudaGetErrorString(cudaGetLastError()));
            return nullptr;
        }
        if (-1 == device_id)
            device_id = current_device;
        if (device_id != current_device)
        {
            cuda_ret = cudaSetDevice(device_id);
            if (cuda_ret != cudaSuccess)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "set current device id to {} failed with {}", 
                    device_id, cudaGetErrorString(cudaGetLastError()));
                return nullptr;
            }
        }
        auto copy_kind = cudaMemcpyHostToDevice;
        switch (kind)
        {
            case EngineTensor::TENSOR_COPY_HOST_TO_DEVICE:
                copy_kind = cudaMemcpyHostToDevice;
                break;
            case EngineTensor::TENSOR_COPY_DEVICE_TO_HOST:
                copy_kind = cudaMemcpyDeviceToHost;
                break;
            case EngineTensor::TENSOR_COPY_DEVICE_TO_DEVICE:
                copy_kind = cudaMemcpyDeviceToDevice;
                break;
            default:
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "unsupported tensor copy kind type {}", kind);
                return nullptr;
            }
        }
        cuda_ret = cudaMemcpy(dst_ptr, src_ptr, mem_size, copy_kind);
        if (cuda_ret != cudaSuccess)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "copy device tensor data failed with {}", cudaGetErrorString(cudaGetLastError()));
            return nullptr;
        }
        if (device_id != current_device)
        {
            cuda_ret = cudaSetDevice(current_device);
            if (cuda_ret != cudaSuccess)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "reset current device id to {} failed with {}", 
                    current_device, cudaGetErrorString(cudaGetLastError()));
                return nullptr;
            }
        }
#else
        ACL_LOG(ACL_LOG_LEVEL_ERROR, "please rebuild with ENGINE_SUPPORT_CUDA");
#endif
        return dst_ptr;
    }

    EngineTensor::EngineTensor(const std::vector<int64_t>& dims, EngineTensor::TensorDataType type, 
        TensorFormatType format, bool alloc_host, bool alloc_device, int device_id)
    {
        m_buffer.type       = type;
        m_buffer.format     = format;
        m_buffer.dim        = dims;
        m_buffer.device     = 0;
        m_buffer.device_id  = -1;
        m_buffer.host       = nullptr;
        if (alloc_host || alloc_device)
        {
            auto memory_size = size();
            if (memory_size > 0 && alloc_host)
            {
                m_buffer.host = (uint8_t*)memoryAllocAlign(size(), ENGINE_MEMORY_ALIGN_DEFAULT);
                if (m_buffer.host == nullptr)
                {
                    ACL_LOG(ACL_LOG_LEVEL_FATAL, "malloc tensor buffer fail");
                }
            }
            if (memory_size > 0 && alloc_device)
            {
                m_buffer.device = (uint64_t)mallocDeviceMem(memory_size, device_id);
                m_buffer.device_id = device_id;
            }
            m_buffer.own_flag = true;
        }
    }

    EngineTensor::EngineTensor(const std::vector<int64_t>& dims, EngineTensor::TensorDataType type, 
        bool alloc_host, bool alloc_device, int device_id)
    {
        m_buffer.type       = type;
        m_buffer.format     = TENSOR_FORMAT_TYPE_NCHW;
        m_buffer.dim        = dims;
        m_buffer.device     = 0;
        m_buffer.device_id  = -1;
        m_buffer.host       = nullptr;
        if (alloc_host || alloc_device)
        {
            auto memory_size = size();
            if (memory_size > 0 && alloc_host)
            {
                m_buffer.host = (uint8_t*)memoryAllocAlign(size(), ENGINE_MEMORY_ALIGN_DEFAULT);
                if (m_buffer.host == nullptr)
                {
                    ACL_LOG(ACL_LOG_LEVEL_FATAL, "malloc host tensor buffer fail");
                }
            }
            if (memory_size > 0 && alloc_device)
            {
                m_buffer.device = (uint64_t)mallocDeviceMem(memory_size, device_id);
                m_buffer.device_id = device_id;
            }
            m_buffer.own_flag = true;
        }
    }

    EngineTensor::~EngineTensor()
    {
        if (nullptr != m_buffer.host && true == m_buffer.own_flag)
        {
            memoryFreeAlign(m_buffer.host);
        }
        if (nullptr != (void*)m_buffer.device && true == m_buffer.own_flag)
        {
            void* device_ptr = (void*)m_buffer.device;
            int device_id = m_buffer.device_id;
            if (device_ptr != nullptr)
            {
                freeDeviceMem(device_ptr, device_id);
            }
        }
    }

    EngineTensor* EngineTensor::createDevice(const std::vector<int64_t>& dims, EngineTensor::TensorDataType type, 
        TensorFormatType format, void* device_data, int device_id)
    {
        std::unique_ptr<EngineTensor> tensor;
        bool malloc_host = false;
        bool malloc_device = device_data == nullptr;
        tensor.reset(new EngineTensor(dims, type, format, malloc_host, malloc_device, device_id));
        if (nullptr != tensor.get() && ((malloc_host && nullptr == tensor->host<void>()) ||
            (malloc_device && nullptr == (void*)tensor->devicePtr())))
        {
            return nullptr;
        }
        if (nullptr != device_data)
        {
            tensor->buffer().device = (uint64_t)device_data;
            if (-1 == device_id)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "create device tensor should speicfic valid device id, when device_ptr is not nullptr");
                return nullptr;
            }
            tensor->buffer().device_id = device_id;
        }
        return tensor.release();
    }

    EngineTensor* EngineTensor::create(const std::vector<int64_t>& dims, EngineTensor::TensorDataType type, 
        TensorFormatType format, void* host_data)
    {
        std::unique_ptr<EngineTensor> tensor;
        bool malloc_host = host_data == nullptr;
        tensor.reset(new EngineTensor(dims, type, format, malloc_host));
        if (nullptr == tensor.get() && nullptr == tensor->host<void>())
            return nullptr;
        if (nullptr != host_data)
        {
            tensor->buffer().host = (uint8_t*)host_data;
        }
        return tensor.release();
    }

    EngineTensor* EngineTensor::copy(EngineTensor* tensor)
    {
        std::unique_ptr<EngineTensor> copy_tensor;
        if(nullptr == tensor)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "input tensor is nullptr, when to copy");
            return nullptr;
        }
        std::vector<int64_t> tensor_shape = tensor->shape();
        EngineTensor::TensorDataType tensor_type = tensor->getTensorDataType(); 
        TensorFormatType tensor_format = tensor->getTensorFormatType();
        void* host_data = tensor->host<void>();
        void* device_data = (void*)tensor->devicePtr();
        bool malloc_host = (nullptr != host_data);
        bool malloc_device = (nullptr != device_data);
        copy_tensor.reset(new EngineTensor(tensor_shape, tensor_type, tensor_format, malloc_host, malloc_device));
        if (nullptr != copy_tensor.get() && 
            ((malloc_host && nullptr == copy_tensor->host<void>()) ||
            (malloc_device && nullptr == (void*)copy_tensor->devicePtr())))
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "copy tensor host_data/device_data is nullptr, when want malloc_host/malloc_device");
            copy_tensor.reset();
        }
        return copy_tensor.release();
    }

    EngineTensor* EngineTensor::clone(EngineTensor* tensor, bool deep_copy)
    {
        std::unique_ptr<EngineTensor> clone_tensor;
        if(nullptr == tensor)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "input tensor is nullptr, when to clone");
            return nullptr;
        }
        std::vector<int64_t> tensor_shape = tensor->shape();
        EngineTensor::TensorDataType tensor_type = tensor->getTensorDataType();
        TensorFormatType tensor_format = tensor->getTensorFormatType();
        void* host_data = tensor->host<void>();
        void* device_data = (void*)tensor->devicePtr();
        int device_id = tensor->deviceId();
        if (deep_copy)
        {
            clone_tensor.reset(EngineTensor::copy(tensor));
        }
        else
        {
            clone_tensor.reset(new EngineTensor(tensor_shape, tensor_type, tensor_format, false, false));
        }
        if (nullptr != clone_tensor.get())
        {
            if (deep_copy)
            {
                void* clone_host = clone_tensor->host<void>();
                void* clone_device = (void*)clone_tensor->devicePtr();
                auto device_kind = EngineTensor::TENSOR_COPY_DEVICE_TO_DEVICE;
                if ((nullptr != host_data && 
                    clone_host != memcpy(clone_host, host_data, tensor->size())) || 
                    (nullptr != device_data && 
                    clone_device != memcpyDeviceMem(clone_device, device_data, 
                    tensor->size(), device_id, device_kind)))
                {
                    ACL_LOG(ACL_LOG_LEVEL_ERROR, "copy data from src tensor to clone tensor fail");
                    clone_tensor.reset();
                }
            }
            else
            {
                clone_tensor->buffer().host = host_data;
                clone_tensor->buffer().device = (uint64_t)device_data;
                clone_tensor->buffer().device_id = device_id;
            }
        }
        return clone_tensor.release();
    }

    EngineTensor* EngineTensor::createHostTensorFromDevice(const EngineTensor* device_tensor, bool copy_data)
    {
        std::unique_ptr<EngineTensor> host_tensor;
        std::vector<int64_t> tensor_shape = device_tensor->shape();
        EngineTensor::TensorDataType tensor_type = device_tensor->getTensorDataType();
        TensorFormatType tensor_format = device_tensor->getTensorFormatType();
        host_tensor.reset(EngineTensor::create(tensor_shape, tensor_type, tensor_format, nullptr));
        if (host_tensor.get() && copy_data)
        {
            if (0 != device_tensor->copyToHostTensor(host_tensor.get()))
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "copy device tensor data to host tensor fail");
                host_tensor.reset();
            }
        }
        return host_tensor.release();
    }

    int EngineTensor::copyFromHostTensor(const EngineTensor* host_tensor)
    {
        if (nullptr == host_tensor)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "host tensor is nullptr, please check");
            return -1;
        }
        if (host_tensor->size() != size())
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "host tensor size {} not equal to device tensor size {}", 
                host_tensor->size(), size());
            return -1;
        }
        void* host = host_tensor->host<void>();
        void* device = (void*)devicePtr();
        int mem_size = size();
        int device_id = deviceId();
        auto kind = EngineTensor::TENSOR_COPY_HOST_TO_DEVICE;
        return (nullptr != memcpyDeviceMem(device, host, mem_size, device_id, kind)) ? 0 : 1;
    }

    int EngineTensor::copyToHostTensor(EngineTensor* host_tensor) const
    {
        if (nullptr == host_tensor)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "host tensor is nullptr, please check");
            return -1;
        }
        if (host_tensor->size() != size())
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "host tensor size {} not equal to device tensor size {}", 
                host_tensor->size(), size());
            return -1;
        }
        void* host = host_tensor->host<void>();
        void* device = (void*)devicePtr();
        int mem_size = size();
        int device_id = deviceId();
        auto kind = EngineTensor::TENSOR_COPY_DEVICE_TO_HOST;
        return (nullptr != memcpyDeviceMem(host, device, mem_size, device_id, kind)) ? 0 : 1;
    }

    int EngineTensor::reshape(const std::vector<int64_t> new_shape)
    {
        int new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<int>());
        int ori_size = size();
        if (new_size == ori_size)
        {
            m_buffer.dim = new_shape;
        }
        else
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "ori tensor size {} not compatible with new size {}", ori_size, new_size);
            return -1;
        }
        return 0;
    }

    EngineTensor::TensorDataType EngineTensor::getTensorDataType() const 
    {
        return m_buffer.type;
    }

    EngineTensor::TensorFormatType EngineTensor::getTensorFormatType() const
    {
        return m_buffer.format;
    }

    int EngineTensor::size() const 
    {
        auto data_size = m_buffer.elementBytes();
        if (data_size < 1)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "element bytes size is {}", data_size);
            return -1;
        }
        for (size_t i = 0; i < m_buffer.dim.size(); i++)
        {
            data_size *= m_buffer.dim[i];
        }
        return data_size;
    }

    std::vector<int64_t> EngineTensor::shape() const 
    {
        return m_buffer.dim;
    }

    template <typename T>
    void printData(const EngineTensor* tensor, const void* data, const char* fmt)
    {
        const T* buffer = (const T*)data;
        std::string fmt_str;
        auto size = tensor->elementSize();
        for (int i = 0; i < size; i++)
        {
            if (i != 0 && i % 10 == 0)
            {
                ACL_LOG(ACL_LOG_LEVEL_INFO, "{}", fmt_str);
                fmt_str.clear();
            }
            else
            {
                T buffer_val = buffer[i];
                fmt_str += std::to_string(buffer_val);
            }
        }
        if (!fmt_str.empty())
            ACL_LOG(ACL_LOG_LEVEL_INFO, "{}", fmt_str);
        return;
    }   

    void EngineTensor::print() const 
    {
        // print dimensions
        std::string dim_str;
        ACL_LOG(ACL_LOG_LEVEL_INFO, "====== Tensor ======");
        for (size_t i = 0; i < m_buffer.dim.size(); i++) 
        {
            dim_str += std::to_string(m_buffer.dim[i]);
            if (i < m_buffer.dim.size() - 1)
                dim_str += "x";
        }
        ACL_LOG(ACL_LOG_LEVEL_INFO, "Dimension: {}", dim_str);
        auto printee = this;
        auto data_buffer = m_buffer.host;
        ACL_LOG(ACL_LOG_LEVEL_INFO, "Data: ");
        if (m_buffer.type == TENSOR_DATA_TYPE_INT8)          // int8
            printData<int8_t>(printee, data_buffer, "%d, ");
        else if (m_buffer.type == TENSOR_DATA_TYPE_INT16)    // int16
            printData<int16_t>(printee, data_buffer, "%d, ");
        else if (m_buffer.type == TENSOR_DATA_TYPE_INT32)    // int32
            printData<int32_t>(printee, data_buffer, "%d, ");
        else if (m_buffer.type == TENSOR_DATA_TYPE_UINT8)    // uint8
            printData<uint8_t>(printee, data_buffer, "%d, ");
        else if (m_buffer.type == TENSOR_DATA_TYPE_UINT16)   // uint16
            printData<uint16_t>(printee, data_buffer, "%d, ");
        else if (m_buffer.type == TENSOR_DATA_TYPE_UINT32)   // uint32
            printData<uint32_t>(printee, data_buffer, "%d, ");
        else if (m_buffer.type == TENSOR_DATA_TYPE_FLOAT32)  // float32
            printData<float>(printee, data_buffer, "%f, ");
        else if (m_buffer.type == TENSOR_DATA_TYPE_FLOAT64)  // float64
            printData<double>(printee, data_buffer, "%f, ");
        else
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "unsupported data type");
    }

    void EngineTensor::printShape() const 
    {
        ACL_LOG(ACL_LOG_LEVEL_INFO, "====== Tensor shape ======");
        if (m_buffer.dim.size() == 0)
        {
            ACL_LOG(ACL_LOG_LEVEL_INFO, "shape dim is zero");
            return;
        }
        std::string log_str;
        for (size_t i = 0; i < m_buffer.dim.size(); i++) 
        {
            log_str += std::to_string(m_buffer.dim[i]);
            if (i != m_buffer.dim.size() - 1)
                log_str += ", ";
        }
        ACL_LOG(ACL_LOG_LEVEL_INFO, "{}", log_str);
    }

} // namespace ACL_ENGINE