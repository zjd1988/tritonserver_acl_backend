/********************************************
 * @Author: zhaojd-a
 * @Date: 2024-06-13
 * @LastEditTime: 2024-06-13
 * @LastEditors: zhaojd-a
 ********************************************/
#include <mutex>
#include <numeric>
#include "acl_engine/file_stream.h"
#include "acl_engine/acl_engine.h"

namespace ACL_ENGINE
{

    static std::once_flag s_flag;
    void initAclResource()
    {
        std::call_once(s_flag, [&]() {
            auto acl_inst = AscendCLInitSingleton::Instance();
        });
    }

    std::map<EngineTensor::TensorDataType, std::string> gEngineTensorDataTypeToStrMap = {
        {EngineTensor::TENSOR_DATA_TYPE_INT8,       "tensor(int8)"},
        {EngineTensor::TENSOR_DATA_TYPE_UINT8,      "tensor(uint8)"},
        {EngineTensor::TENSOR_DATA_TYPE_INT16,      "tensor(int16)"},
        {EngineTensor::TENSOR_DATA_TYPE_UINT16,     "tensor(uint16)"},
        {EngineTensor::TENSOR_DATA_TYPE_INT32,      "tensor(int32)"},
        {EngineTensor::TENSOR_DATA_TYPE_UINT32,     "tensor(uint32)"},
        {EngineTensor::TENSOR_DATA_TYPE_INT64,      "tensor(int64)"},
        {EngineTensor::TENSOR_DATA_TYPE_UINT64,     "tensor(uint64)"},
        {EngineTensor::TENSOR_DATA_TYPE_FLOAT16,    "tensor(float16)"},
        {EngineTensor::TENSOR_DATA_TYPE_FLOAT32,    "tensor(float32)"},
        {EngineTensor::TENSOR_DATA_TYPE_FLOAT64,    "tensor(float64)"},
        {EngineTensor::TENSOR_DATA_TYPE_STRING,     "tensor(string)"},
        {EngineTensor::TENSOR_DATA_TYPE_MAX,        "tensor(unknown)"}
    };

    std::map<std::string, EngineTensor::TensorDataType> gEngineTensorStrToDataTypeMap = {
        {"tensor(int8)",       EngineTensor::TENSOR_DATA_TYPE_INT8},
        {"tensor(uint8)",      EngineTensor::TENSOR_DATA_TYPE_UINT8},
        {"tensor(int16)",      EngineTensor::TENSOR_DATA_TYPE_INT16},
        {"tensor(uint16)",     EngineTensor::TENSOR_DATA_TYPE_UINT16},
        {"tensor(int32)",      EngineTensor::TENSOR_DATA_TYPE_INT32},
        {"tensor(uint32)",     EngineTensor::TENSOR_DATA_TYPE_UINT32},
        {"tensor(int64)",      EngineTensor::TENSOR_DATA_TYPE_INT64},
        {"tensor(uint64)",     EngineTensor::TENSOR_DATA_TYPE_UINT64},
        {"tensor(float16)",    EngineTensor::TENSOR_DATA_TYPE_FLOAT16},
        {"tensor(float32)",    EngineTensor::TENSOR_DATA_TYPE_FLOAT32},
        {"tensor(float64)",    EngineTensor::TENSOR_DATA_TYPE_FLOAT64},
        {"tensor(string)",     EngineTensor::TENSOR_DATA_TYPE_STRING},
        {"tensor(unknown)",    EngineTensor::TENSOR_DATA_TYPE_MAX},
    };

    static aclDataType convertTensorTypeToAscendCLType(EngineTensor::TensorDataType tensor_type)
    {
        aclDataType acl_dtype;
        switch (tensor_type)
        {
            case EngineTensor::TENSOR_DATA_TYPE_INT8:
            {
                acl_dtype = ACL_INT8;
                break; 
            }
            case EngineTensor::TENSOR_DATA_TYPE_INT16:
            {
                acl_dtype = ACL_INT16;
                break; 
            }
            case EngineTensor::TENSOR_DATA_TYPE_INT32:
            {
                acl_dtype = ACL_INT32;
                break; 
            }
            case EngineTensor::TENSOR_DATA_TYPE_INT64:
            {
                acl_dtype = ACL_INT64;
                break; 
            }
            case EngineTensor::TENSOR_DATA_TYPE_UINT8:
            {
                acl_dtype = ACL_UINT8;
                break; 
            }
            case EngineTensor::TENSOR_DATA_TYPE_UINT16:
            {
                acl_dtype = ACL_UINT16;
                break; 
            }
            case EngineTensor::TENSOR_DATA_TYPE_UINT32:
            {
                acl_dtype = ACL_UINT32;
                break; 
            }
            case EngineTensor::TENSOR_DATA_TYPE_UINT64:
            {
                acl_dtype = ACL_UINT64;
                break; 
            }
            case EngineTensor::TENSOR_DATA_TYPE_FLOAT16:
            {
                acl_dtype = ACL_FLOAT16;
                break; 
            }
            case EngineTensor::TENSOR_DATA_TYPE_FLOAT32:
            {
                acl_dtype = ACL_FLOAT;
                break; 
            }
            case EngineTensor::TENSOR_DATA_TYPE_FLOAT64:
            {
                acl_dtype = ACL_DOUBLE;
                break; 
            }
            default:
            {
                acl_dtype = ACL_DT_UNDEFINED; // never reach
                ACL_LOG(ACL_LOG_LEVEL_FATAL, "unsupported engine tensor type:{}", int(tensor_type));
            }
        }
        return acl_dtype;
    }

    static EngineTensor::TensorDataType convertAscendCLTypeToTensorType(aclDataType acl_type)
    {
        EngineTensor::TensorDataType tensor_type;
        switch (acl_type)
        {
            case ACL_INT8:
            {
                tensor_type = EngineTensor::TENSOR_DATA_TYPE_INT8;
                break; 
            }
            case ACL_INT16:
            {
                tensor_type = EngineTensor::TENSOR_DATA_TYPE_INT16;
                break; 
            }
            case ACL_INT32:
            {
                tensor_type = EngineTensor::TENSOR_DATA_TYPE_INT32;
                break; 
            }
            case ACL_INT64:
            {
                tensor_type = EngineTensor::TENSOR_DATA_TYPE_INT64;
                break; 
            }
            case ACL_UINT8:
            {
                tensor_type = EngineTensor::TENSOR_DATA_TYPE_UINT8;
                break; 
            }
            case ACL_UINT16:
            {
                tensor_type = EngineTensor::TENSOR_DATA_TYPE_UINT16;
                break; 
            }
            case ACL_UINT32:
            {
                tensor_type = EngineTensor::TENSOR_DATA_TYPE_UINT32;
                break; 
            }
            case ACL_UINT64:
            {
                tensor_type = EngineTensor::TENSOR_DATA_TYPE_UINT64;
                break; 
            }
            case ACL_FLOAT16:
            {
                tensor_type = EngineTensor::TENSOR_DATA_TYPE_FLOAT16;
                break; 
            }
            case ACL_FLOAT:
            {
                tensor_type = EngineTensor::TENSOR_DATA_TYPE_FLOAT32;
                break; 
            }
            case ACL_DOUBLE:
            {
                tensor_type = EngineTensor::TENSOR_DATA_TYPE_FLOAT64;
                break; 
            }
            default:
            {
                tensor_type = EngineTensor::TENSOR_DATA_TYPE_MAX; // never reach
                ACL_LOG(ACL_LOG_LEVEL_FATAL, "unsupported acl tensor type:{}", int(acl_type));
            }
        }
        return tensor_type;
    }

    AscendCLEngine::AscendCLEngine(const EngineConfig& config, const std::vector<std::string>& model_files)
    {
        initAclResource();
        if (0 != loadModelFromFile(config, model_files))
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "init acl engine fail");
            return;
        }
        m_status = true;
    }

    AscendCLEngine::AscendCLEngine(const EngineConfig& config, const std::vector<const char*>& model_datas, 
        const std::vector<size_t>& data_lens)
    {
        initAclResource();
        if (0 != loadModelFromBuffer(config, model_datas, data_lens))
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "init acl engine fail");
            return;
        }
        m_status = true;
    }

    AscendCLEngine::~AscendCLEngine()
    {

        auto ret = aclmdlUnload(m_model_id);
        if (ACL_ERROR_NONE != ret)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "unload model failed, ret:{}, msg:{}", int(ret), aclGetRecentErrMsg());
            assert(0);
        }

        if (m_model_desc != nullptr)
        {
            ret = aclmdlDestroyDesc(m_model_desc);
            if (ACL_ERROR_NONE != ret)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "unload model failed, ret:{}, msg:{}", int(ret), aclGetRecentErrMsg());
                assert(0);
            }
            m_model_desc = nullptr;
        }
        destroyInputsBuffer();
        destroyOutputsBuffer();

        // destroy stream
        if (nullptr != m_stream)
        {
            ret = aclrtDestroyStream(m_stream);
            if (ACL_ERROR_NONE != ret)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "destroy stream failed, ret:{}, msg:{}", int(ret), aclGetRecentErrMsg());
                assert(0);
            }
            m_stream = nullptr;
        }

        // destroy context
        if (nullptr != m_context)
        {
            ret = aclrtDestroyContext(m_context);
            if (ACL_ERROR_NONE != ret)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "destroy context failed, ret:{}, msg:{}", int(ret), aclGetRecentErrMsg());
                assert(0);
            }
            m_context = nullptr;
        }

        rt_ret = aclrtResetDevice(m_engine_config.device_id);
        if (ACL_ERROR_NONE != rt_ret)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "reset device {} failed, ret:{}, msg:{}", m_engine_config.device_id, int(ret), 
                aclGetRecentErrMsg());
            assert(0);
        }

        return;
    }

    void AscendCLEngine::printEngineInfo()
    {
        // engine config
        auto& config = m_engine_config;
        ACL_LOG(ACL_LOG_LEVEL_INFO, "device_id                      : {}", config.device_id);
        ACL_LOG(ACL_LOG_LEVEL_INFO, "config file                    : {}", config.config_file);

        // log input tensor infos
        std::vector<EngineTensorInfo> input_tensor_infos;
        if (0 != getInputTensorInfos(input_tensor_infos))
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "get input tensor infos fail");
            return;
        }
        for (size_t index = 0; index < input_tensor_infos.size(); index++)
        {
            auto& tensor_info = input_tensor_infos[index];
            auto tensor_type = gEngineTensorDataTypeToStrMap[tensor_info.type];
            ACL_LOG(ACL_LOG_LEVEL_INFO, "input tensor index={}, name={}, type={}, dim={}, shape=[{}]", 
                index, tensor_info.name, tensor_type, tensor_info.shape.size(), 
                spdlog::fmt_lib::join(tensor_info.shape, ","));
        }

        // log output tensor infos
        std::vector<EngineTensorInfo> output_tensor_infos;
        if (0 != getOutputTensorInfos(output_tensor_infos))
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "get output tensor infos fail");
            return;
        }
        for (size_t index = 0; index < output_tensor_infos.size(); index++)
        {
            auto& tensor_info = output_tensor_infos[index];
            auto tensor_type = gEngineTensorDataTypeToStrMap[tensor_info.type];
            ACL_LOG(ACL_LOG_LEVEL_INFO, "output tensor index={}, name={}, type={}, dim={}, shape=[{}]", 
                index, tensor_info.name, tensor_type, tensor_info.shape.size(), 
                spdlog::fmt_lib::join(tensor_info.shape, ","));
        }
        return;
    }

    int AscendCLEngine::checkEngineConfig(const EngineConfig& config)
    {

        // check device id valid
        uint32_t device_count;
        aclError ret = aclrtGetDeviceCount(&device_count);
        if (ACL_SUCCESS != ret)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl get device count fail, ret:{}", int(ret));
            return -1;
        }

        if (0 > config.device_id || device_count <= config.device_id)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "invalid acl device id:{}, total device count:{}", config.device_id, device_count);
            return -1;
        }

        return 0;
    }

    std::set<uint64_t> AscendCLEngine::getDynamicBatch()
    {
        if (nullptr == m_model_desc)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl model desc is nullptr");
            return std::set<uint64_t>();
        }
        aclmdlBatch dynamic_batch;
        if (ACL_SUCCESS != aclmdlGetDynamicBatch(m_model_desc, &dynamic_batch))
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "failed to get dynamic batch");
            return std::set<uint64_t>();
        }
        size_t batch_count = dynamic_batch.batchCount;
        if (batch_count > ACL_MAX_BATCH_NUM)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "real batch count {} is larger than max {}", batch_count, ACL_MAX_BATCH_NUM);
            return std::set<uint64_t>();
        }
        std::set<uint64_t> batch;
        for (size_t i = 0; i < dynamic_batch.batchCount; ++i)
        {
            batch.insert(dynamic_batch.batch[i]);
        }
        return batch;
    }

    std::set<std::pair<uint64_t, uint64_t>> AscendCLEngine::getDynamicImage()
    {
        if (nullptr == m_model_desc)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl model desc is nullptr");
            return std::set<std::pair<uint64_t, uint64_t>>();
        }
        aclmdlHW dynamic_hw;
        if (ACL_SUCCESS != aclmdlGetDynamicHW(m_model_desc, 0, &dynamic_hw))
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "failed to get dynamic hw");
            return std::set<std::pair<uint64_t, uint64_t>>();
        }
        size_t hw_count = dynamic_hw.hwCount;
        if (hw_count > ACL_MAX_HW_NUM)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "Real hw count {} is larger than max {}", hw_count, ACL_MAX_HW_NUM);
            return std::set<std::pair<uint64_t, uint64_t>>();
        }
        std::set<std::pair<uint64_t, uint64_t>> image;
        for (size_t i = 0; i < dynamic_hw.hwCount; ++i)
        {
            image.insert(std::pair<uint64_t, uint64_t>(dynamic_hw.hw[i][0], dynamic_hw.hw[i][1]));
        }
        return image;
    }

    std::pair<aclmdlIODims*, size_t> AscendCLEngine::getDynamicDims()
    {
        if (nullptr == m_model_desc)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl model desc is nullptr");
            return std::make_pair(nullptr, 0);
        }
        size_t gear_conut = 0;
        auto ret = aclmdlGetInputDynamicGearCount(m_model_desc, -1, &gear_conut);
        if (ACL_SUCCESS != ret)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "aclmdlGetInputDynamicGearCount failed");
            return std::make_pair(nullptr, 0);
        }
        ACL_LOG(ACL_LOG_LEVEL_DEBUG, "gear_conut is: {}", gear_conut);
        if (gear_conut == 0)
        {
            ACL_LOG(ACL_LOG_LEVEL_DEBUG, "gear_conut is zero");
            return std::make_pair(nullptr, 0);
        }
        m_dynamic_dims = new aclmdlIODims[gear_conut];
        if (nullptr == m_dynamic_dims)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "new aclmldIODims failed");
            return std::make_pair(nullptr, 0);
        }
        if (ACL_SUCCESS != aclmdlGetInputDynamicDims(m_model_desc, -1, m_dynamic_dims, gear_conut))
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "aclmdlGetInputDynamicDims failed");
            delete[] m_dynamic_dims;
            m_dynamic_dims = nullptr;
            return std::make_pair(nullptr, 0);
        }
        return std::make_pair(m_dynamic_dims, gear_conut);
    }

    std::vector<Format> AscendCLEngine::getInputFormat()
    {
        if (m_model_desc == nullptr)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl model desc is nullptr");
            return std::vector<Format>();
        }
        std::vector<Format> input_formats;
        static const std::map<aclFormat, enum Format> acl_format_map = {
            {ACL_FORMAT_NCHW, NCHW}, 
            {ACL_FORMAT_NHWC, NHWC},
            {ACL_FORMAT_ND, NCHW}};

        for (size_t i = 0; i < m_data_input_num; ++i)
        {
            aclFormat format = aclmdlGetInputFormat(m_model_desc, i);
            auto iter = acl_format_map.find(format);
            if (iter != acl_format_map.end())
            {
                input_formats.emplace_back(iter->second);
            }
            else
            {
                ACL_LOG(ACL_LOG_LEVEL_DEBUG, "aclFormat " << format << " not found in map, please double check and add...using default format");
                input_formats.emplace_back(DEFAULT_FORMAT);
            }
            MS_LOG(DEBUG) << "Format of Input " << i << " is " << static_cast<int32_t>(format);
        }
        return input_formats;
    }

    std::vector<Format> AscendCLEngine::GetOutputFormat()
    {
        if (m_model_desc == nullptr)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl model desc is nullptr");
            return std::vector<Format>();
        }
        std::vector<Format> output_formats;
        static const std::map<aclFormat, enum Format> acl_format_map = {
            {ACL_FORMAT_NCHW, NCHW}, 
            {ACL_FORMAT_NHWC, NHWC}, 
            {ACL_FORMAT_ND, NCHW}};

        for (size_t i = 0; i < m_output_infos.size(); ++i)
        {
            aclFormat format = aclmdlGetOutputFormat(m_model_desc, i);
            auto iter = acl_format_map.find(format);
            if (iter != acl_format_map.end())
            {
                output_formats.emplace_back(iter->second);
            }
            else
            {
                ACL_LOG(ACL_LOG_LEVEL_DEBUG, "aclFormat " << format << " not found in map, please double check and add...using default format");
                output_formats.emplace_back(DEFAULT_FORMAT);
            }
            MS_LOG(DEBUG) << "Format of Output " << i << " is " << static_cast<int32_t>(format);
        }
        return output_formats;
    }

    bool AscendCLEngine::isDynamicShape() { return isDynamicBatchSize() || isDynamicImageSize() || isDynamicDims(); }

    bool AscendCLEngine::isDynamicBatchSize() { return !m_dynamic_shape_options.batch_size.empty(); }

    bool AscendCLEngine::isDynamicImageSize() { return !m_dynamic_shape_options.image_size.empty(); }

    bool AscendCLEngine::isDynamicDims() { return 0 != m_dynamic_shape_options.dynamic_dims.second; }

    const std::vector<TypeId> AscendCLEngine::getOutputDataType()
    {
        std::vector<TypeId> data_types;
        for (size_t i = 0; i < m_output_infos.size(); ++i)
        {
            TypeId data_type = TransToDataType(m_output_infos[i].data_type);
            data_types.emplace_back(data_type);
        }
        return data_types;
    }

    const std::vector<std::vector<int64_t>> AscendCLEngine::getOutputShape()
    {
        std::vector<std::vector<int64_t>> shapes;
        for (size_t i = 0; i < m_output_infos.size(); ++i)
        {
            shapes.emplace_back(m_output_infos[i].dims);
        }
        return shapes;
    }

    const std::vector<TypeId> AscendCLEngine::getInputDataType()
    {
        std::vector<TypeId> data_types;
        for (size_t i = 0; i < m_data_input_num; ++i)
        {
            TypeId data_type = TransToDataType(m_input_infos[i].data_type);
            data_types.emplace_back(data_type);
        }
        return data_types;
    }

    const std::vector<std::vector<int64_t>> AscendCLEngine::getInputShape()
    {
        std::vector<std::vector<int64_t>> shapes;
        for (size_t i = 0; i < m_data_input_num; ++i)
        {
            shapes.push_back(m_input_infos[i].dims);
        }
        return shapes;
    }

    bool AscendCLEngine::checkAndSetDynFlag()
    {
        if (nullptr == m_model_desc)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl model desc is nullptr");
            return false;
        }

        // check and set input dynamic flag
        aclError ret;
        size_t input_size = aclmdlGetNumInputs(m_model_desc);
        for (size_t i = 0; i < input_size; ++i)
        {
            auto buffer_size = aclmdlGetInputSizeByIndex(m_model_desc, i);
            aclmdlIODims input_dims;
            ret = aclmdlGetInputDimsV2(m_model_desc, i, &input_dims);
            if (ACL_ERROR_NONE != ret)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl model get input dims failed");
                return false;
            }
            for (size_t j = 0; j < input_dims.dimCount; ++j)
            {
                if (input_dims.dims[j] < 0)
                {
                    if (buffer_size == 0)
                    {
                        m_is_dynamic_input = true;
                        ACL_LOG(ACL_LOG_LEVEL_DEBUG, "the input of acl model is dynamic");
                        break;
                    }
                    else
                    {
                        if (!isDynamicShape())
                        {
                            m_is_dynamic_shape_range = true;
                            ACL_LOG(ACL_LOG_LEVEL_DEBUG, "the input of acl model is dynamic shape range");
                        }
                    }
                }
            }
            if (m_is_dynamic_input || m_is_dynamic_shape_range)
            {
                break;
            }
        }

        // check and set output dynamic flag
        size_t output_size = aclmdlGetNumOutputs(m_model_desc);
        for (size_t i = 0; i < output_size; ++i)
        {
            aclmdlIODims output_dims;
            ret = aclmdlGetOutputDims(m_model_desc, i, &output_dims);
            if (ACL_ERROR_NONE != ret)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl model get output dims failed");
                return false;
            }
            for (size_t j = 0; j < output_dims.dimCount; ++j)
            {
                if (output_dims.dims[j] < 0)
                {
                    m_is_dynamic_output = true;
                    ACL_LOG(ACL_LOG_LEVEL_DEBUG, "the output of acl model is dynamic");
                    return true;
                }
            }
        }
        return true;
    }

    bool AscendCLEngine::createDataBuffer(void** data_mem_buffer, size_t buffer_size, aclmdlDataset* dataset)
    {
        aclError ret;
        auto free_data_buffer = [this](void *dataMemBuffer) {
            if (!m_is_run_on_device)
            {
                (void)aclrtFree(dataMemBuffer);
            }
            else
            {
                (void)aclrtFreeHost(dataMemBuffer);
            }
        };
        // The model with dynamic input do not need to malloc the memory of output
        if (0 != buffer_size)
        {
            if (nullptr == data_mem_buffer)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "data mem buffer is nullptr");
                return false;
            }
            if (!m_is_run_on_device)
            {
                ret = aclrtMalloc(data_mem_buffer, buffer_size, ACL_MEM_MALLOC_HUGE_FIRST);
                if (ACL_ERROR_NONE != ret)
                {
                    ACL_LOG(ACL_LOG_LEVEL_ERROR, "malloc device buffer failed, buffer size {}", buffer_size);
                    return false;
                }
            }
            else
            {
                ret = aclrtMallocHost(data_mem_buffer, buffer_size);
                if (ACL_ERROR_NONE != ret)
                {
                    ACL_LOG(ACL_LOG_LEVEL_ERROR, "malloc host buffer failed, buffer size {}", buffer_size);
                    return false;
                }
            }
        }
        auto data_buffer = aclCreateDataBuffer(*data_mem_buffer, buffer_size);
        if (nullptr == data_buffer)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "create Data Buffer failed");
            if (data_mem_buffer != nullptr)
            {
                free_data_buffer(*data_mem_buffer);
            }
            aclDestroyDataBuffer(data_buffer);
            return false;
        }
        ret = aclmdlAddDatasetBuffer(dataset, data_buffer);
        if (ACL_ERROR_NONE != ret)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "add data buffer failed");
            if (data_mem_buffer != nullptr)
            {
                free_data_buffer(*data_mem_buffer);
            }
            aclDestroyDataBuffer(data_buffer);
            return false;
        }
        return true;
    }

    bool AscendCLEngine::initInputsBuffer()
    {
        if (nullptr == m_model_desc)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl model desc is nullptr");
            return false;
        }

        aclError ret;
        m_input_dataset = aclmdlCreateDataset();
        if (nullptr == m_input_dataset)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "create input dataset failed");
            return false;
        }
        size_t input_size = aclmdlGetNumInputs(m_model_desc);
        ACL_LOG(ACL_LOG_LEVEL_DEBUG, "model input_size = {}", input_size);
        for (size_t i = 0; i < input_size; ++i)
        {
            aclmdlIODims dims;
            // To get correct dims with static AIPP configured, same result as aclmdlGetInputDims without static AIPP
            if (m_is_dynamic_output)
            {  // There is a bug for aclmdlGetInputDimsV2 when output is dynamic shape.
                ret = aclmdlGetInputDims(m_model_desc, i, &dims);
            }
            else
            {
                ret = aclmdlGetInputDimsV2(m_model_desc, i, &dims);
            }
            if (ACL_ERROR_NONE != ret)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "get input shape failed, ret:{}", int(ret));
                return false;
            }

            auto buffer_size = aclmdlGetInputSizeByIndex(m_model_desc, i);
            void *data_mem_buffer = nullptr;
            if (!m_is_dynamic_input && !createDataBuffer(&data_mem_buffer, buffer_size, m_input_dataset))
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "add input data buffer failed, buffer size {}", buffer_size);
                return false;
            }
            aclDataType data_type = aclmdlGetInputDataType(m_model_desc, i);
            std::vector<int64_t> shape(dims.dims, dims.dims + dims.dimCount);
            std::string input_name = aclmdlGetInputNameByIndex(m_model_desc, i);
            if (!m_is_dynamic_input)
            {
                aclFormat input_format = aclmdlGetInputFormat(m_model_desc, i);
                aclTensorDesc *desc = aclCreateTensorDesc(data_type, dims.dimCount, dims.dims, input_format);
                ret = aclmdlSetDatasetTensorDesc(m_input_dataset, desc, i);
                if (ACL_ERROR_NONE != ret)
                {
                    ACL_LOG(ACL_LOG_LEVEL_ERROR, "aclmdlSetDatasetTensorDesc failed, ret:{}", int(ret));
                    return false;
                }
            }
            if (input_name.empty())
            {
                ACL_LOG(ACL_LOG_LEVEL_WARN, "get name of input {} failed", i);
            }
            ACL_LOG(ACL_LOG_LEVEL_DEBUG, "name of input {} is {}", i, input_name);
            m_input_infos.emplace_back(AclTensorInfo{data_mem_buffer, data_mem_buffer, buffer_size, buffer_size, 
                data_type, shape, input_name});
        }
        ACL_LOG(ACL_LOG_LEVEL_DEBUG, "create model inputs success");
        return true;
    }

    bool AscendCLEngine::initOutputsBuffer()
    {
        if (nullptr == m_model_desc)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl model desc is nullptr");
            return false;
        }

        aclError ret;
        m_output_dataset = aclmdlCreateDataset();
        if (nullptr == m_output_dataset)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "create output dataset failed");
            return false;
        }
        size_t output_size = aclmdlGetNumOutputs(m_model_desc);
        ACL_LOG(ACL_LOG_LEVEL_DEBUG, "model output_size = {}", output_size);
        for (size_t i = 0; i < output_size; ++i)
        {
            aclmdlIODims dims;
            ret = aclmdlGetOutputDims(m_model_desc, i, &dims);
            if (ACL_ERROR_NONE != ret)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "get output shape failed, ret:{}", int(ret));
                return false;
            }
            bool is_dynamic_output = false;
            for (size_t dim_idx = 0; dim_idx < dims.dimCount; dim_idx++)
            {
                is_dynamic_output = (dims.dims[dim_idx] < 0) ? true : false;
            }
            size_t buffer_size = 0;
            if (!is_dynamic_output)
            {
                buffer_size = aclmdlGetOutputSizeByIndex(m_model_desc, i);
            }
            void *data_mem_buffer = nullptr;
            if (!createDataBuffer(&data_mem_buffer, buffer_size, m_output_dataset))
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "add output data buffer failed, buffer size {}", buffer_size);
                return false;
            }
            aclFormat format = aclmdlGetOutputFormat(m_model_desc, i);
            ACL_LOG(ACL_LOG_LEVEL_DEBUG, "the output format of om is ", int(format));
            aclDataType data_type = aclmdlGetOutputDataType(m_model_desc, i);
            std::vector<int64_t> shape(dims.dims, dims.dims + dims.dimCount);
            if (is_dynamic_output)
            {
                shape = std::vector<int64_t>({-1});
            }
            std::string output_name = aclmdlGetOutputNameByIndex(m_model_desc, i);
            if (output_name.empty())
            {
                ACL_LOG(ACL_LOG_LEVEL_WARN, "get name of output {} failed", i);
            }
            ACL_LOG(ACL_LOG_LEVEL_DEBUG, "name of om output {} is {} buffer size {}", i, output_name, buffer_size);
            m_output_infos.emplace_back(AclTensorInfo{data_mem_buffer, data_mem_buffer, buffer_size, buffer_size, 
                data_type, shape, output_name});
        }
        ACL_LOG(ACL_LOG_LEVEL_DEBUG, "create model output success");
        return true;
    }

    void AscendCLEngine::destroyInputsBuffer()
    {
        for (const auto &item : m_input_infos)
        {
            if (item.device_data != nullptr)
            {
                if (!m_is_run_on_device)
                {
                    aclrtFree(item.device_data);
                }
                else
                {
                    aclrtFreeHost(item.device_data);
                }
            }
            if (nullptr != item.dynamic_acl_tensor_desc)
            {
                aclDestroyTensorDesc(item.dynamic_acl_tensor_desc);
            }
        }
        m_input_infos.clear();

        if (nullptr == m_input_dataset)
        {
            return;
        }
        for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(m_input_dataset); i++)
        {
            auto dataBuffer = aclmdlGetDatasetBuffer(m_input_dataset, i);
            aclDestroyDataBuffer(dataBuffer);
        }
        aclmdlDestroyDataset(m_input_dataset);
        m_input_dataset = nullptr;
    }

    void AscendCLEngine::destroyOutputsBuffer()
    {
        if (!m_is_dynamic_output)
        {
            for (const auto &item : m_output_infos)
            {
                if (item.device_data != nullptr)
                {
                    if (!m_is_run_on_device)
                    {
                        aclrtFree(item.device_data);
                    }
                    else
                    {
                        aclrtFreeHost(item.device_data);
                    }
                }
            }
        }
        m_output_infos.clear();

        if (nullptr == m_output_dataset)
        {
            return;
        }
        for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(m_output_dataset); i++)
        {
            auto dataBuffer = aclmdlGetDatasetBuffer(m_output_dataset, i);
            aclDestroyDataBuffer(dataBuffer);
        }
        aclmdlDestroyDataset(m_output_dataset);
        m_output_dataset = nullptr;
    }

    int AscendCLEngine::initAclModelFromBuffer(const char* model_data, const size_t& data_len, 
        const EngineConfig& acl_config)
    {

        // set device 
        auto device_id = m_engine_config.device_id;
        aclError ret = aclrtSetDevice(device_id);
        if (ACL_ERROR_NONE != ret)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl set device:{} failed, ret:{}, msg:{}", device_id, int(ret), 
                aclGetRecentErrMsg());
            return -1;
        }

        // create context
        ret = aclrtCreateContext(&m_context, device_id);
        if (ACL_ERROR_NONE != ret)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl create context failed, ret:{}, msg:{}", int(ret), aclGetRecentErrMsg());
            return -1;
        }

        // get run mode stream
        aclrtRunMode run_mode;
        ret = aclrtGetRunMode(&run_mode);
        if (ACL_ERROR_NONE != ret)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl get run mode failed, ret:{}, msg:{}", int(ret), aclGetRecentErrMsg());
            return -1;
        }
        m_is_run_on_device = (run_mode == ACL_DEVICE);
        ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl model is running in {} mode", int(m_run_mode));

        // create stream
        ret = aclrtCreateStream(&m_stream);
        if (ACL_ERROR_NONE != ret)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl create stream failed, ret:{}, msg:{}", int(ret), aclGetRecentErrMsg());
            return -1;
        }

        // set context
        ret = aclrtSetCurrentContext(m_context);
        if (ACL_ERROR_NONE != ret)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl set context failed, ret:{}, msg:{}", int(ret), aclGetRecentErrMsg());
            return -1;
        }

        // load model from memory
        ret = aclmdlLoadFromMem(model_data, data_len, &m_model_id);
        if (ACL_ERROR_NONE != ret)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "load acl model failed, ret:{}, msg:{}", int(ret), aclGetRecentErrMsg());
            return -1;
        }

        // create and get model desc
        m_model_desc = aclmdlCreateDesc();
        ret = aclmdlGetDesc(m_model_desc, m_model_id);
        if (ACL_ERROR_NONE != ret)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "read model desc failed, ret:{}, msg:{}", int(ret), aclGetRecentErrMsg());
            return -1;
        }

        // init dynamic shape options
        m_dynamic_shape_options.batch_size = getDynamicBatch();
        m_dynamic_shape_options.image_size = getDynamicImage();
        m_dynamic_shape_options.dynamic_dims = getDynamicDims();

        // check and set dynamic flag
        if (!checkAndSetDynFlag())
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "check and set dynamic flag failed");
            return -1;
        }

        // init input buffers
        if (!initInputsBuffer())
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "create input buffer failed");
            return -1;
        }

        // init output buffers
        if (!initOutputsBuffer())
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "create output buffer failed");
            return -1;
        }

        // 
        if (m_is_dynamic_input)
        {
            m_data_input_num = m_input_infos.size();
            return 0;
        }

        m_data_input_num = m_input_infos.size();
        if (isDynamicShape() && 0 < m_data_input_num)
        {
            m_data_input_num -= 1;
        }

        m_dynamic_shape_options.input_format = getInputFormat();
        m_dynamic_shape_options.input_shapes = getInputShape();

        if (!m_dyn_shape_proc.Init(m_dynamic_shape_options))
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "init DynShapeProcess failed");
            return -1;
        }

        m_status = true;
        return 0;
    }

    int AscendCLEngine::loadModelFromFile(const EngineConfig& config, const std::vector<std::string>& model_files)
    {
        if (1 != model_files.size())
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl engine expect 1 model files, but get {} files", model_files.size());
            return -1;
        }

        std::vector<std::shared_ptr<FileInputStream>> file_streams;
        std::vector<const char *> model_datas;
        std::vector<size_t> data_lens;
        for (size_t i = 0; i < model_files.size(); i++)
        {
            std::shared_ptr<FileInputStream> file_stream(new FileInputStream(model_files[i]));
            if (nullptr == file_stream.get() || !file_stream->isOpen())
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl engine open file {} fail", model_files[i]);
                return -1;
            }
            const char *file_data = file_stream->getFileData();
            size_t file_len = file_stream->getFileSize();
            if (nullptr == file_data)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl engine get file {} data fail", model_files[i]);
                return -1;
            }
            model_datas.push_back(file_data);
            data_lens.push_back(file_len);
            file_streams.push_back(file_stream);
        }

        if (0 != loadModelFromBuffer(config, model_datas, data_lens))
        {
            ACL_LOG(ACL_LOG_LEVEL_INFO, "acl engine init from file {} fail", model_files[0]);
            return -1;
        }
        ACL_LOG(ACL_LOG_LEVEL_INFO, "acl engine init from file {} success", model_files[0]);
        return 0;
    }

    int AscendCLEngine::loadModelFromBuffer(const EngineConfig& config, const std::vector<const char*>& model_datas, 
        const std::vector<size_t>& data_lens)
    {
        if (1 != model_datas.size() || 1 != data_lens.size())
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "expect model_datas size:1 and data_lens size:1, but get "
                "model_datas size:{} data_len size:{}", model_datas.size(), data_lens.size());
            return -1;
        }
        if (0 == checkEngineConfig(config))
        {
            m_engine_config = config;
            if (0 != initAclModelFromBuffer(model_datas[0], data_lens[0], m_engine_config))
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "init acl model fail");
                return -1;
            }
            printEngineInfo();
            ACL_LOG(ACL_LOG_LEVEL_INFO, "init acl engine from buffer success");
        }
        else
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "check acl engine config fail");
            return -1;
        }
        return 0;
    }

    int AscendCLEngine::setEngineInputTensors(std::map<std::string, EngineTensor*>& input_tensors_map)
    {
        // check model valid
        if (!m_status)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "model has not been loaded");
            return false;
        }

        // check input tensors
        if (0 == input_tensors_map.size())
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl engine input tensors is empty");
            return -1;
        }

        // construct new shapes
        std::vector<std::vector<int64_t>> new_shapes;
        for (auto& input : m_input_infos)
        {
            std::string input_name = input.name;
            if (input_tensors_map.find(input_name) == input_tensors_map.end())
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl engine input tensors map cannot find tensor: {}", input_name);
                return -1;
            }
            std::vector<int64_t>& input_shape = input_tensors_map[input_name]->shape();
            new_shapes.push_back(input_shape);
        }

        // if input tensor shape not equal to acl engine shape, need to resize acl engine
        if (0 != resizeEngine(new_shapes))
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl engine resize with new shapes fail");
            return -1;
        }

        // set input tensors
        for (auto& it : input_tensors_map)
        {
            std::string tensor_name = it.first;
            auto& input_tensor = it.second;
            auto tensor_dtype = input_tensor->getTensorDataType();
            auto tensor_format = input_tensor->getTensorFormatType();
            auto tensor_shape = input_tensor->shape();
            void* tensor_data = input_tensor->host<void>();
            std::shared_ptr<EngineTensor> temp_tensor;
            temp_tensor.reset(EngineTensor::create(tensor_shape, tensor_dtype, tensor_format, tensor_data));
            if (nullptr == temp_tensor.get() || nullptr == temp_tensor->host<void>())
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl engine create input engine tensor {} fail", 
                    tensor_name);
                return -1;
            }
            m_input_tensors_map[tensor_name] = temp_tensor;
        }

        std::vector<mindspore::MSTensor> model_outputs = m_model->GetOutputs();
        for (size_t i = 0; i < model_outputs.size(); i++)
        {
            auto& model_output = model_outputs[i];
            std::string model_output_name = model_output.Name();
            auto model_output_type = model_output.DataType();
            auto model_output_shape = model_output.Shape();

            // current not support dynamic output shape, when set fix input shape
            if (0 >= model_output.ElementNum())
            {
                ACL_LOG(ACL_LOG_LEVEL_WARN, "acl engine output tensor {} has dyanmic dims:{}", 
                    model_output_name, spdlog::fmt_lib::join(model_output_shape, ","));
                continue;
            }

            // check output shape
            if ((m_output_tensors_map.end() == m_output_tensors_map.find(model_output_name) || 
                m_output_tensors_map[model_output_name]->shape() != model_output_shape) && 
                "cpu" != m_engine_config.device_type)
            {
                std::shared_ptr<EngineTensor> output_tensor;
                auto tensor_shape = model_output_shape;
                auto tensor_type = convertACLTypeToTensorType(model_output_type);
                output_tensor.reset(EngineTensor::create(tensor_shape, tensor_type, EngineTensor::TENSOR_FORMAT_TYPE_NCHW));
                if (nullptr == output_tensor.get() || nullptr == output_tensor->host<void>())
                {
                    ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl engine create output engine tensor {} fail", 
                        model_output_name);
                    return -1;
                }
                m_output_tensors_map[model_output_name] = output_tensor;
            }
        }
        return 0;
    }

    int AscendCLEngine::getEngineOutputTensors(std::map<std::string, EngineTensor*>& output_tensors_map)
    {
        // check model/context valid
        if (nullptr == m_model.get() || nullptr == m_context.get())
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl model or context is nullptr");
            return -1;
        }

        // clear history
        output_tensors_map.clear();

        // get output tensors from model
        for (auto& it : m_output_tensors_map)
        {
            std::string tensor_name = it.first;
            output_tensors_map[tensor_name] = m_output_tensors_map[tensor_name].get();
        }

        // The device_data is malloced by acl, user need to free the addr
        if (m_is_dynamic_output)
        {
            freeResourceOutput(&m_output_infos, outputs);
        }

        return 0;
    }

    int AscendCLEngine::resizeEngine(std::map<std::string, std::vector<int64_t>>& new_shapes)
    {
        // check model valid
        if (!m_status)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "model has not been loaded");
            return false;
        }

        // acl model resize
        if (!resize(new_shapes))
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl model resize fail");
            return -1;
        }

        return 0;
    }

    int AscendCLEngine::runEngine()
    {
        // check model valid
        if (!m_status)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "model has not been loaded");
            return false;
        }

        // set current context
        auto ret = aclrtSetCurrentContext(m_context);
        if (ACL_ERROR_NONE != ret)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl set context failed, ret:{}, msg:{}", int(ret), aclGetRecentErrMsg());
            return -1;
        }

        // model execute
        auto ret = aclmdlExecute(m_model_id, m_input_dataset, m_output_dataset);
        if (ACL_ERROR_NONE != ret)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "execute model failed, ret:{}, msg:{}", int(ret), aclGetRecentErrMsg());
            return -1;
        }

        // set input host data
        auto model_inputs = m_model->GetInputs();
        auto model_outputs = m_model->GetOutputs();
        for (size_t index = 0; index < model_inputs.size(); index++)
        {
            auto& model_input = model_inputs[index];
            std::string tensor_name = model_input.Name();
            auto& input_tensor = m_input_tensors_map[tensor_name];
            model_input.SetData(input_tensor->host<void>(), false);
            model_input.SetDeviceData(nullptr);
        }

        return 0;
    }

    int AscendCLEngine::runEngine(std::map<std::string, EngineTensor*>& input_tensors_map, 
        std::map<std::string, EngineTensor*>& output_tensors_map)
    {
        // check model
        if (nullptr == m_model.get() || nullptr == m_context.get())
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl model or context is nullptr");
            return -1;
        }

        // set input tensors
        if (0 != setEngineInputTensors(input_tensors_map))
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "set acl engine input tensors fail");
            return -1;
        }

        // run engine
        if (0 != runEngine())
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "run acl engine fail");
            return -1;
        }

        // get output tensors
        if (0 != getEngineOutputTensors(output_tensors_map))
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "get acl engine output tensors fail");
            return -1;
        }
        return 0;
    }

    int AscendCLEngine::getInputTensorInfos(std::vector<EngineTensorInfo>& input_tensor_infos)
    {
        // check context/model_desc valid
        if (nullptr == m_context || nullptr == m_model_desc)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl context or model desc is nullptr");
            return -1;
        }
        input_tensor_infos.clear();

        size_t input_count = aclmdlGetNumInputs(m_model_desc);
        for (auto index = 0; index < input_count; index++)
        {
            // get input tensor name
            std::string tensor_name = std::string(aclmdlGetInputNameByIndex(m_model_desc, index));

            // get input tensor shape
            std::vector<int64_t> tensor_shape;
            aclmdlIODims tensor_dim;
            aclError ret = aclmdlGetInputDims(m_model_desc, index, &tensor_dim);
            if (ACL_SUCCESS != ret)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl get input_{} dims failed, ret:{}, msg:{}", index, int(ret), 
                    aclGetRecentErrMsg());
                return -1;
            }
            for (int dim_index = 0; dim_index < tensor_dim.dimCount; dim_index++)
                tensor_shape.push_back(tensor_dim.dims[dim_index]);

            // get input tensor datatype
            aclDataType tensor_dtype = aclmdlGetInputDataType(m_model_desc, index);

            EngineTensorInfo tensor_info;
            tensor_info.name = tensor_name;
            tensor_info.type = convertAscendCLTypeToTensorType(tensor_dtype);
            tensor_info.shape = tensor_shape;
            input_tensor_infos.push_back(tensor_info);
        }
        return 0;
    }

    int AscendCLEngine::getOutputTensorInfos(std::vector<EngineTensorInfo>& output_tensor_infos)
    {
        // check context/model_desc valid
        if (nullptr == m_context || nullptr == m_model_desc)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl context or model desc is nullptr");
            return -1;
        }
        output_tensor_infos.clear();

        size_t output_count = aclmdlGetNumInputs(m_model_desc);
        for (auto index = 0; index < output_count; index++)
        {
            // get output tensor name
            std::string tensor_name = std::string(aclmdlGetOutputNameByIndex(m_model_desc, index));
            // 由于ascend_cl输出名称会自动加前缀且用:分开
            auto pos = tensor_name.rfind(":");
            if (pos != std::string::npos)
                tensor_name = tensor_name.substr(pos + 1);

            // get output tensor shape
            std::vector<int64_t> tensor_shape;
            aclmdlIODims tensor_dim;
            aclError ret = aclmdlGetOutputDims(m_model_desc, index, &tensor_dim);
            if (ACL_SUCCESS != ret)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl get input_{} dims failed, ret:{}, msg:{}", index, int(ret), 
                    aclGetRecentErrMsg());
                return -1;
            }
            for (int dim_index = 0; dim_index < tensor_dim.dimCount; dim_index++)
                tensor_shape.push_back(tensor_dim.dims[dim_index]);

            // get output tensor datatype
            aclDataType tensor_dtype = aclmdlGetOutputDataType(m_model_desc, index);

            EngineTensorInfo tensor_info;
            tensor_info.name = tensor_name;
            tensor_info.type = convertACLTypeToTensorType(tensor_dtype);
            tensor_info.shape = tensor_shape;
            output_tensor_infos.push_back(tensor_info);
        }
        return 0;
    }

    bool AscendCLEngine::resetInputSize(const std::vector<std::vector<int64_t>>& new_shapes)
    {
        for (size_t index = 0; index < new_shapes.size(); index++)
        {
            std::vector<int64_t> shape = new_shapes[index];
            size_t elem_count = 1;
            for (size_t i = 0; i < shape.size(); i++)
            {
                if (shape[i] < 0)
                {
                    elem_count = 0;
                    break;
                }
                elem_count *= shape[i];
            }
            m_input_infos[index].dims = shape;
            auto data_type = aclmdlGetInputDataType(m_model_desc, index);
            auto new_buffer_size = elem_count * aclDataTypeSize(data_type);
            if (!m_is_dynamic_input)
            {
                m_input_infos[index].buffer_size = new_buffer_size;
            }
            else if (new_buffer_size > m_input_infos[index].buffer_size)
            {
                m_is_dynamic_resize_input = true;
                m_input_infos[index].buffer_size = new_buffer_size;
            }
        }
        return true;
    }

    bool AscendCLEngine::resetOutputSize()
    {
        if (m_model_desc == nullptr)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, " Model desc is nullptr");
            return false;
        }
        aclDataType data_type;
        aclError ret;
        size_t output_size = aclmdlGetNumOutputs(m_model_desc);
        for (size_t index = 0; index < output_size; index++)
        {
            struct aclmdlIODims dims;
            ret = aclmdlGetCurOutputDims(m_model_desc, index, &dims);
            if (ret != ACL_ERROR_NONE)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "get output dim error.");
                return false;
            }
            std::vector<int64_t> shape(dims.dims, dims.dims + dims.dimCount);
            size_t elem_count = 1;
            for (size_t i = 0; i < dims.dimCount; i++)
            {
                if (dims.dims[i] < 0)
                {
                    elem_count = 0;
                    break;
                }
                elem_count *= dims.dims[i];
            }
            data_type = aclmdlGetOutputDataType(m_model_desc, index);
            m_output_infos[index].dims = shape;
            m_output_infos[index].buffer_size = elem_count * aclDataTypeSize(data_type);
        }
        return true;
    }

    bool AscendCLEngine::resizeDynamicInputShape(const std::vector<std::vector<int64_t>>& new_shapes)
    {
        ACL_LOG(ACL_LOG_LEVEL_DEBUG, "Start to resize dynamic input shape");
        // If it is not the first time to resize input shape, the old addr need to be free
        resetInputSize(new_shapes);
        freeResourceInput(m_input_infos);
        if (m_is_dynamic_resize_input)
        {
            m_input_dataset = aclmdlCreateDataset();
            if (m_input_dataset == nullptr)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "Create input dataset failed");
                return false;
            }
        }
        for (size_t i = 0; i < new_shapes.size(); ++i)
        {
            if (m_is_dynamic_resize_input)
            {
                void *data_buf = nullptr;
                if (!createDataBuffer(&data_buf, m_input_infos[i].buffer_size, m_input_dataset))
                {
                    ACL_LOG(ACL_LOG_LEVEL_ERROR, "add input data buffer failed");
                    return false;
                }
                auto data_type = aclmdlGetInputDataType(m_model_desc, i);
                std::string input_name = aclmdlGetInputNameByIndex(m_model_desc, i);
                if (input_name.empty())
                {
                    ACL_LOG(ACL_LOG_LEVEL_ERROR, "get name of input {} failed", i);
                    return false;
                }
                ACL_LOG(ACL_LOG_LEVEL_DEBUG, "name of input {} is {}", i, input_name);
                m_input_infos[i].cur_device_data = data_buf;
                m_input_infos[i].device_data = data_buf;
                m_input_infos[i].data_type = data_type;
                m_input_infos[i].name = input_name;
                auto data_buffer = aclmdlGetDatasetBuffer(m_input_dataset, i);
                m_input_infos[i].dynamic_acl_data_buffer = data_buffer;
            }

            aclTensorDesc *input_desc = aclCreateTensorDesc(ACL_FLOAT, new_shapes[i].size(), &new_shapes[i][0], ACL_FORMAT_NCHW);
            auto ret = aclmdlSetDatasetTensorDesc(m_input_dataset, input_desc, i);
            m_input_infos[i].dynamic_acl_tensor_desc = input_desc;
            if (ret != ACL_ERROR_NONE)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl set dataset tensor desc failed");
                return false;
            }
        }
        m_is_dynamic_resize_input = false;
        ACL_LOG(ACL_LOG_LEVEL_DEBUG, "resize dynamic input shape success");
        return true;
    }

    bool AscendCLEngine::resizeDynamicInputShapeRange(const std::vector<std::vector<int64_t>>& new_shapes)
    {
        ACL_LOG(ACL_LOG_LEVEL_DEBUG, "start to resize dynamic input shape range");
        for (size_t i = 0; i < new_shapes.size(); ++i)
        {
            std::vector<int64_t> shape = new_shapes[i];
            auto buffer_size = aclmdlGetInputSizeByIndex(m_model_desc, i);
            auto data_type = aclmdlGetInputDataType(m_model_desc, i);
            size_t elem_count = 1;
            for (size_t j = 0; j < shape.size(); ++j)
            {
                if (shape[j] < 0)
                {
                    ACL_LOG(ACL_LOG_LEVEL_ERROR, "the resize shape has the dim less than 0");
                    return false;
                }
                elem_count *= shape[j];
            }
            auto new_buffer_size = elem_count * aclDataTypeSize(data_type);
            if (new_buffer_size > buffer_size)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "the resize shape is over shape range");
                return false;
            }
            m_input_infos[i].dims = shape;
            aclTensorDesc *input_desc = aclCreateTensorDesc(ACL_FLOAT, new_shapes[i].size(), &new_shapes[i][0], ACL_FORMAT_NCHW);
            auto ret = aclmdlSetDatasetTensorDesc(m_input_dataset, input_desc, i);
            if (ret != ACL_ERROR_NONE)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl set dataset tensor desc failed");
                return false;
            }
        }
        ACL_LOG(ACL_LOG_LEVEL_DEBUG, "resize dynamic input shape range success");
        return true;
    }

    bool AscendCLEngine::resizeDynamicBatchAndImageSize(const std::vector<std::vector<int64_t>>& new_shapes)
    {
        if (m_model_desc == nullptr || m_input_dataset == nullptr)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "model is not inited");
            return false;
        }
        size_t index;
        auto ret = aclmdlGetInputIndexByName(m_model_desc, ACL_DYNAMIC_TENSOR_NAME, &index);
        if (ret != ACL_ERROR_NONE)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "get index of dynamic tensor failed");
            return false;
        }
        if (isDynamicBatchSize())
        {
            int32_t batch_size = 0;
            if (!m_dyn_shape_proc.CheckAndGetBatchSize(new_shapes, &batch_size))
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "failed to check batch size");
                return false;
            }
            ACL_LOG(ACL_LOG_LEVEL_DEBUG, "set Batch size({}) of input {}", batch_size, index);
            ret = aclmdlSetDynamicBatchSize(m_model_id, m_input_dataset, index, batch_size);
            if (ret != ACL_ERROR_NONE)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "Set dynamic batch size failed, model_id is {}", m_model_id);
                return false;
            }
        }
        else if (isDynamicImageSize())
        {
            int32_t height = 0;
            int32_t width = 0;
            if (!m_dyn_shape_proc.CheckAndGetImageSize(new_shapes, &height, &width))
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "failed to check image size");
                return false;
            }
            ACL_LOG(ACL_LOG_LEVEL_DEBUG, "set Image size({},{}) of input {}", height, width, index);
            ret = aclmdlSetDynamicHWSize(m_model_id, m_input_dataset, index, height, width);
            if (ret != ACL_ERROR_NONE)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "set dynamic batch size failed, model_id is {}", m_model_id);
                return false;
            }
        }
        else if (isDynamicDims())
        {
            aclmdlIODims dynamic_dims;
            if (!m_dyn_shape_proc.CheckAndGetDynamicDims(new_shapes, &dynamic_dims))
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "CheckAndGetDynamicDims failed");
                return false;
            }
            ret = aclmdlSetInputDynamicDims(m_model_id, m_input_dataset, index, &dynamic_dims);
            if (ret != ACL_ERROR_NONE)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "aclmdlSetInputDynamicDims failed");
                return false;
            }
        }
        else
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "not support dynamic input");
            return false;
        }
        if (!resetInputSize(new_shapes))
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "reset input size failed");
            return false;
        }
        if (!resetOutputSize())
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "reset output size failed");
            return false;
        }
        return true;
    }

    bool AscendCLEngine::resize(const std::vector<std::vector<int64_t>>& new_shapes)
    {

        // set current context
        auto ret = aclrtSetCurrentContext(m_context);
        if (ACL_ERROR_NONE != ret)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl set context failed, ret:{}, msg:{}", int(ret), aclGetRecentErrMsg());
            return -1;
        }

        auto input_shapes = getInputShape();
        if (input_shapes.size() != new_shapes.size())
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "invalid new input size {}, expect input size {}", new_shapes.size(), 
                input_shapes.size());
            return false;
        }

        bool input_shape_changed = false;
        for (size_t i = 0; i < new_shapes.size(); i++)
        {
            auto new_shape = new_shapes[i];
            if (std::any_of(new_shape.begin(), new_shape.end(), [](auto dim) { return dim < 0; }))
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "new shape of input {} cannot be dynamic, new shape:{}", i, 
                    spdlog::fmt_lib::join(new_shape, ","));
                return false;
            }
            if (input_shapes[i] != new_shape)
            {
                input_shape_changed = true;
            }
        }

        if (!input_shape_changed)
        {
            return true;
        }

        if (m_is_dynamic_input)
        {
            return resizeDynamicInputShape(new_shapes);
        }

        if (m_is_dynamic_shape_range)
        {
            return resizeDynamicInputShapeRange(new_shapes);
        }

        if (!isDynamicShape())
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "not support dynamic input");
            return false;
        }

        if (!resizeDynamicBatchAndImageSize(new_shapes))
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "resize dynamic batch and image size failed");
            return false;
        }

        return true;
    }

    bool AscendCLEngine::checkInputTensors(const std::vector<EngineTensor*> &input_tensors)
    {
        if (m_data_input_num != input_tensors.size())
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "expect input size to be {}, but got {}", m_data_input_num, input_tensors.size());
            return false;
        }
        for (size_t i = 0; i < input_tensors.size(); ++i)
        {
            auto &tensor = input_tensors[i];
            auto &info = m_input_infos[i];
            if (tensor->shape() != info.dims)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "note: input {} shape not match, required {}, given {}."
                    "Please check input shape has been modified by DVPP method.", i, spdlog::fmt_lib::join(info.dims, ","), 
                    spdlog::fmt_lib::join(tensor->shape()));
                return false;
            }
            if (tensor->GetDtype() != TransToDataType(info.data_type))
            {
                MS_LOG(ERROR) << "Note: input " << i << " data type not match, required "
                                << static_cast<int>(TransToDataType(info.data_type)) << ", given "
                                << static_cast<int>(tensor->GetDtype());
                return false;
            }
            auto device_data = tensor->GetData();
            auto host_data = tensor->GetHostData();
            if (device_data != nullptr && device_data->addr != nullptr)
            {
                if (!is_dynamic_input_ && !is_dynamic_shape_range_ && device_data->size != info.buffer_size)
                {
                    MS_LOG(ERROR) << "Input " << i << " data size not match, required size " << info.buffer_size << ", given count "
                                << device_data->size;
                    return false;
                }
            }
            else if (host_data != nullptr && host_data->addr != nullptr)
            {
                if (!m_is_dynamic_input && !m_is_dynamic_shape_range && host_data->size != info.buffer_size)
                {
                    MS_LOG(ERROR) << "Input " << i << " data size not match, required size " << info.buffer_size << ", given count "
                                << host_data->size;
                    return false;
                }
            }
            else
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "failed to get data from input {}", i);
                return false;
            }
        }
        return true;
    }

    bool AscendCLEngine::checkOutputTensors(const std::vector<KernelTensorPtr> &outputs)
    {
        if (outputs.size() != m_output_infos.size())
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "actual tensor count not match, required count {}, but got {}", output_infos_.size(), 
                outputs.size());
            return false;
        }
        if (m_is_dynamic_output)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "this model has dynamic output shape");
            return true;
        }
        for (size_t i = 0; i < outputs.size(); ++i)
        {
            auto &tensor = outputs[i];
            auto &info = output_infos_[i];
            if (tensor->GetShapeVector() != info.dims)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "note: output {} shape not match, required {}, given {}."
                    "please check output shape", i, spdlog::fmt_lib::join(info.dims, ","), 
                    spdlog::fmt_lib::join(tensor->GetShapeVector(), ","));
            }
            if (tensor->GetDtype() != TransToDataType(info.data_type))
            {
                MS_LOG(ERROR) << "Note: output " << i << " data type not match, required "
                                << static_cast<int>(TransToDataType(info.data_type)) << ", given "
                                << static_cast<int>(tensor->GetDtype());
                return false;
            }
            auto device_data = tensor->GetData();
            auto host_data = tensor->GetHostData();
            if (device_data != nullptr && device_data->addr != nullptr)
            {
                if (device_data->size != info.buffer_size)
                {
                    MS_LOG(ERROR) << "Output " << i << " device data size not match, required size " << info.buffer_size
                                << ", given count " << tensor->GetData()->size;
                    return false;
                }
            } 
            else if (host_data != nullptr && host_data->addr != nullptr)
            {
                if (host_data->size != info.buffer_size) {
                    MS_LOG(ERROR) << "Output " << i << " host data size not match, required size " << info.buffer_size
                                << ", given count " << tensor->GetData()->size;
                    return false;
                }
            }
            else
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "failed to get data from output {}", i);
                return false;
            }
        }
        return true;
    }

    void AscendCLEngine::heckAndInitDynOutputDeviceBuf(const EngineTensor* output, const AclTensorInfo& output_info,
        void** output_device_buffer, size_t* output_buf_size, size_t output_idx)
    {
        auto device_data = output->GetData();
        auto host_data = output->GetHostData();
        if ((host_data == nullptr) || (dyn_out_sys_buf_addr_.find(host_data->addr) != dyn_out_sys_buf_addr_.end()) ||
            (host_data->size == 0))
        {
            MS_LOG(DEBUG) << "host_data->addr: " << host_data->addr
                        << ", user not defined dynamic output buffer on host, using system defined buffer";
            user_defined_output_buf_[output_idx] = false;
        }
        if (user_defined_output_buf_[output_idx])
        {
            *output_device_buffer = output_info.device_data;
            auto addr = (host_data != nullptr) ? host_data->addr : device_data->addr;
            auto size = (host_data != nullptr) ? host_data->size : device_data->size;
            *output_buf_size = size;
            MS_LOG(DEBUG) << "found user buffer with addr: " << addr << " with size: " << size
                        << ". init output device addr: " << output_info.device_data;
        }
    }

    bool AscendCLEngine::checkAndInitInput(const std::vector<EngineTensor*>& inputs)
    {
        // check inputs
        if (!checkInputTensors(inputs))
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "check input tensor failed");
            return false;
        }
        aclError ret;
        // copy inputs
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            auto &info = m_input_infos[i];
            auto input = inputs[i];
            void *input_buffer = nullptr;
            auto data = input->host<void>();
            auto size = input->size();
            if (!m_is_run_on_device)
            {
                ret = aclrtMemcpy(info.device_data, info.buffer_size, data, size, ACL_MEMCPY_HOST_TO_DEVICE);
                if (ACL_ERROR_NONE != ret)
                {
                    ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl memcpy input {} data to device failed, src input size: {}"
                        ", dst device buffer size: {}", i, size, info.buffer_size);
                    return false;
                }
                input_buffer = info.device_data;
            }
            else
            {
                input_buffer = data;
            }
            auto data_buffer = aclmdlGetDatasetBuffer(m_input_dataset, i);
            if (nullptr == data_buffer)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "failed to get dataset buffer of input {}", i);
                return false;
            }
            ret = aclUpdateDataBuffer(data_buffer, input_buffer, info.buffer_size);
            if (ACL_ERROR_NONE != ret)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "failed to update data buffer of input {}, buffer size: {}, input shape: {}", 
                    i, info.buffer_size, spdlog::fmt_lib::join(input->shape(), ","));
                return false;
            }
        }
        return true;
    }

    bool AscendCLEngine::checkAndInitOutput(const std::vector<EngineTensor*>& outputs)
    {
        // check outputs
        if (!checkOutputTensors(outputs))
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "check output tensor failed");
            return false;
        }
        aclError ret;
        // copy outputs
        for (size_t i = 0; i < outputs.size(); ++i)
        {
            auto &info = m_output_infos[i];
            auto output = outputs[i];
            void *output_device_buffer = nullptr;
            auto host_data = output->host<void>();
            auto output_device_buffer_size = info.buffer_size;
            bool is_dynamic = m_is_dynamic_input || m_is_dynamic_shape_range || m_is_dynamic_output;

            if (host_data && m_is_run_on_device)
            {
                output_device_buffer = host_data->addr;
            }
            else
            {
                output_device_buffer = info.device_data;
                if (is_dynamic)
                {
                    output_device_buffer = nullptr;  // in dynamic output shape, setting nullptr allows acl to alloc memory
                    output_device_buffer_size = 0;
                    checkAndInitDynOutputDeviceBuf(output, info, &output_device_buffer, &output_device_buffer_size, i);
                }
            }
            auto data_buffer = aclmdlGetDatasetBuffer(m_output_dataset, i);
            if (data_buffer == nullptr)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "failed to get dataset buffer of output {}", i);
                return false;
            }
            ret = aclUpdateDataBuffer(data_buffer, output_device_buffer, output_device_buffer_size);
            if (ACL_ERROR_NONE != ret)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "failed to update data buffer of output {}, buffer size: {}, output shape: {}", 
                    i, info.buffer_size, spdlog::fmt_lib::join(output->shape(), ","));
                return false;
            }
        }
        return true;
    }

    void AscendCLEngine::freeResourceInput(std::vector<AclTensorInfo>& acl_tensor_info)
    {
        for (const auto &item : acl_tensor_info)
        {
            if (item.dynamic_acl_tensor_desc != nullptr)
            {
                aclDestroyTensorDesc(item.dynamic_acl_tensor_desc);
            }
            if (m_is_dynamic_resize_input)
            {
                if (item.device_data != nullptr)
                {
                    if (!m_is_run_on_device)
                    {
                        aclrtFree(item.device_data);
                    }
                    else
                    {
                        aclrtFreeHost(item.device_data);
                    }
                }
                if (item.dynamic_acl_data_buffer != nullptr)
                {
                    aclDestroyDataBuffer(item.dynamic_acl_data_buffer);
                }
            }
        }
        if (m_is_dynamic_resize_input)
        {
            aclmdlDestroyDataset(m_input_dataset);
            m_input_dataset = nullptr;
        }
    }

    void AscendCLEngine::freeResourceOutput(std::vector<AclTensorInfo>& acl_tensor_info)
    {
        for (size_t i = 0; i < acl_tensor_info->size(); i++)
        {
            auto &item = (*acl_tensor_info)[i];
            if (item.device_data != nullptr)
            {
                MS_LOG(DEBUG) << "freeing device buffer at addr: " << item.device_data;
                if (!m_is_run_on_device)
                {
                    aclrtFree(item.device_data);
                }
                else
                {
                    aclrtFreeHost(item.device_data);
                }
                item.device_data = nullptr;
            }
            if (item.dynamic_acl_data_buffer != nullptr)
            {
                aclDestroyDataBuffer(item.dynamic_acl_data_buffer);
            }
            if (item.dynamic_acl_tensor_desc != nullptr)
            {
                aclDestroyTensorDesc(item.dynamic_acl_tensor_desc);
            }
        }
    }

} // namespace ACL_ENGINE