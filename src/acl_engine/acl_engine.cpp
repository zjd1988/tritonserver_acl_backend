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
        // check model valid
        if (!m_status)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl model has not been loaded");
            return -1;
        }

        // engine config
        auto& config = m_engine_config;
        ACL_LOG(ACL_LOG_LEVEL_INFO, "device_id                      : {}", config.device_id);
        ACL_LOG(ACL_LOG_LEVEL_INFO, "config file                    : {}", config.config_file);

        // log input tensor infos
        for (size_t index = 0; index < m_input_infos.size(); index++)
        {
            auto& tensor_info = m_input_infos[index];
            auto tensor_type = gEngineTensorDataTypeToStrMap[tensor_info.data_type];
            ACL_LOG(ACL_LOG_LEVEL_INFO, "input tensor index={}, name={}, type={}, dim={}, shape=[{}]", 
                index, tensor_info.name, tensor_type, tensor_info.dims.size(), 
                spdlog::fmt_lib::join(tensor_info.dims, ", "));
        }

        // log output tensor infos
        for (size_t index = 0; index < m_output_infos.size(); index++)
        {
            auto& tensor_info = m_output_infos[index];
            auto tensor_type = gEngineTensorDataTypeToStrMap[tensor_info.data_type];
            ACL_LOG(ACL_LOG_LEVEL_INFO, "output tensor index={}, name={}, type={}, dim={}, shape=[{}]", 
                index, tensor_info.name, tensor_type, tensor_info.dims.size(), 
                spdlog::fmt_lib::join(tensor_info.dims, ", "));
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

    int AscendCLEngine::getDynamicBatch(std::set<uint64_t>& dynamic_batch_set)
    {
        // check model desc valid
        if (nullptr == m_model_desc)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl model desc is nullptr");
            dynamic_batch_set = std::set<uint64_t>();
            return -1;
        }

        // get dynamic batch info
        aclmdlBatch dynamic_batch;
        if (ACL_SUCCESS != aclmdlGetDynamicBatch(m_model_desc, &dynamic_batch))
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "failed to get dynamic batch");
            return -1;
        }

        // check dynamic batch count valid
        size_t batch_count = dynamic_batch.batchCount;
        if (batch_count > ACL_MAX_BATCH_NUM)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "real batch count {} is larger than max {}", batch_count, ACL_MAX_BATCH_NUM);
            return -1;
        }

        // get dynamic batch set
        std::set<uint64_t> batch;
        for (size_t i = 0; i < dynamic_batch.batchCount; ++i)
        {
            batch.insert(dynamic_batch.batch[i]);
        }
        dynamic_batch_set = batch;
        return 0;
    }

    int AscendCLEngine::getDynamicImage(std::set<std::pair<uint64_t, uint64_t>>& dynamic_image_set)
    {
        // check model desc valid
        if (nullptr == m_model_desc)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl model desc is nullptr");
            dynamic_image_set = std::set<std::pair<uint64_t, uint64_t>>();
            return 0;
        }

        // get dynamic hw info
        aclmdlHW dynamic_hw;
        if (ACL_SUCCESS != aclmdlGetDynamicHW(m_model_desc, -1, &dynamic_hw))
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "failed to get dynamic hw");
            return -1;
        }

        // check dynamic hw count valid
        size_t hw_count = dynamic_hw.hwCount;
        if (hw_count > ACL_MAX_HW_NUM)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "Real hw count {} is larger than max {}", hw_count, ACL_MAX_HW_NUM);
            return -1;
        }

        // get dynamic hw set
        std::set<std::pair<uint64_t, uint64_t>> image;
        for (size_t i = 0; i < dynamic_hw.hwCount; ++i)
        {
            image.insert(std::pair<uint64_t, uint64_t>(dynamic_hw.hw[i][0], dynamic_hw.hw[i][1]));
        }
        dynamic_image_set = image;
        return 0;
    }

    int AscendCLEngine::getDynamicDims(std::pair<aclmdlIODims*, size_t>& dynamic_dims)
    {
        // check model desc valid
        if (nullptr == m_model_desc)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl model desc is nullptr");
            dynamic_dims = std::make_pair(nullptr, 0);
            return -1;
        }

        // get input dynamic gear count
        size_t gear_conut = 0;
        auto ret = aclmdlGetInputDynamicGearCount(m_model_desc, -1, &gear_conut);
        if (ACL_SUCCESS != ret)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "aclmdlGetInputDynamicGearCount failed");
            return -1;
        }
        ACL_LOG(ACL_LOG_LEVEL_DEBUG, "gear_conut is: {}", gear_conut);

        // if gear count is zero, return <nullptr, 0> pair
        if (0 == gear_conut)
        {
            ACL_LOG(ACL_LOG_LEVEL_DEBUG, "gear_conut is zero");
            return 0;
        }

        // if gear count not zero, get real dynamic dims pair
        m_dynamic_dims = new aclmdlIODims[gear_conut];
        if (nullptr == m_dynamic_dims)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "new aclmldIODims failed");
            return -1;
        }
        if (ACL_SUCCESS != aclmdlGetInputDynamicDims(m_model_desc, -1, m_dynamic_dims, gear_conut))
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "aclmdlGetInputDynamicDims failed");
            delete[] m_dynamic_dims;
            m_dynamic_dims = nullptr;
            return -1;
        }
        dynamic_dims = std::make_pair(m_dynamic_dims, gear_conut);
        return 0;
    }

    int AscendCLEngine::getInputFormat(std::vector<EngineTensor::TensorFormatType>& input_formats)
    {
        if (nullptr == m_model_desc)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl model desc is nullptr");
            return -1;
        }
        static const std::map<aclFormat, EngineTensor::TensorFormatType> acl_format_map = {
            {ACL_FORMAT_NCHW, EngineTensor::TENSOR_FORMAT_TYPE_NCHW}, 
            {ACL_FORMAT_NHWC, EngineTensor::TENSOR_FORMAT_TYPE_NHWC},
            {ACL_FORMAT_ND, EngineTensor::TENSOR_FORMAT_TYPE_NCHW}};

        for (size_t index = 0; index < m_data_input_num; ++index)
        {
            aclFormat format = aclmdlGetInputFormat(m_model_desc, index);
            auto iter = acl_format_map.find(format);
            if (iter != acl_format_map.end())
            {
                input_formats.emplace_back(iter->second);
            }
            else
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "aclFormat {} not found in map, please double check and add...using default format", 
                    static_cast<int32_t>(format));
                return -1;
            }
            ACL_LOG(ACL_LOG_LEVEL_DEBUG, "format of input {} is {}", index, static_cast<int32_t>(format));
        }
        return 0;
    }

    int AscendCLEngine::getOutputFormat(std::vector<EngineTensor::TensorFormatType>& output_formats);
    {
        if (nullptr == m_model_desc)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl model desc is nullptr");
            return -1;
        }
        static const std::map<aclFormat, EngineTensor::TensorFormatType> acl_format_map = {
            {ACL_FORMAT_NCHW, EngineTensor::TENSOR_FORMAT_TYPE_NCHW}, 
            {ACL_FORMAT_NHWC, EngineTensor::TENSOR_FORMAT_TYPE_NHWC},
            {ACL_FORMAT_ND, EngineTensor::TENSOR_FORMAT_TYPE_NCHW}};

        for (size_t index = 0; index < m_output_infos.size(); ++index)
        {
            aclFormat format = aclmdlGetOutputFormat(m_model_desc, index);
            auto iter = acl_format_map.find(format);
            if (iter != acl_format_map.end())
            {
                output_formats.emplace_back(iter->second);
            }
            else
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "aclFormat {} not found in map, please double check and add...using default format", 
                    static_cast<int32_t>(format));
                return -1;
            }
            ACL_LOG(ACL_LOG_LEVEL_DEBUG, "format of output {} is {}", index, static_cast<int32_t>(format));
        }
        return 0;
    }

    bool AscendCLEngine::isDynamicShape() { return isDynamicBatchSize() || isDynamicImageSize() || isDynamicDims(); }

    bool AscendCLEngine::isDynamicBatchSize() { return !m_dynamic_shape_options.batch_size.empty(); }

    bool AscendCLEngine::isDynamicImageSize() { return !m_dynamic_shape_options.image_size.empty(); }

    bool AscendCLEngine::isDynamicDims() { return 0 != m_dynamic_shape_options.dynamic_dims.second; }

    int AscendCLEngine::getOutputDataType(std::vector<EngineTensor::TensorDataType>& output_dtypes)
    {
        for (size_t index = 0; index < m_output_infos.size(); ++index)
        {
            EngineTensor::TensorDataType data_type = convertAscendCLTypeToTensorType(m_output_infos[index].data_type);
            output_dtypes.emplace_back(data_type);
        }
        return 0;
    }

    int AscendCLEngine::getOutputShape(std::vector<std::vector<int64_t>>& output_shapes)
    {
        for (size_t index = 0; index < m_output_infos.size(); ++index)
        {
            output_shapes.emplace_back(m_output_infos[index].dims);
        }
        return 0;
    }

    int AscendCLEngine::getInputDataType(std::vector<EngineTensor::TensorDataType>& input_dtypes)
    {
        for (size_t index = 0; index < m_data_input_num; ++index)
        {
            EngineTensor::TensorDataType data_type = convertAscendCLTypeToTensorType(m_input_infos[index].data_type);
            input_dtypes.emplace_back(data_type);
        }
        return 0;
    }

    int AscendCLEngine::getInputShape(std::vector<std::vector<int64_t>>& input_shapes)
    {
        for (size_t index = 0; index < m_data_input_num; ++index)
        {
            input_shapes.emplace_back(m_input_infos[index].dims);
        }
        return 0;
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
                    if (0 == buffer_size)
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
                if (0 > output_dims.dims[j])
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
        // check model desc valid
        if (nullptr == m_model_desc)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl model desc is nullptr");
            return false;
        }

        // create input dataset
        aclError ret;
        m_input_dataset = aclmdlCreateDataset();
        if (nullptr == m_input_dataset)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "create input dataset failed");
            return false;
        }

        // init input infos and input dataset
        size_t input_size = aclmdlGetNumInputs(m_model_desc);
        ACL_LOG(ACL_LOG_LEVEL_DEBUG, "model input_size = {}", input_size);
        for (size_t index = 0; index < input_size; ++index)
        {
            aclmdlIODims dims;
            // To get correct dims with static AIPP configured, same result as aclmdlGetInputDims without static AIPP
            if (m_is_dynamic_output)
            {  // There is a bug for aclmdlGetInputDimsV2 when output is dynamic shape.
                ret = aclmdlGetInputDims(m_model_desc, index, &dims);
            }
            else
            {
                ret = aclmdlGetInputDimsV2(m_model_desc, index, &dims);
            }
            if (ACL_ERROR_NONE != ret)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "get input:{} shape failed, ret:{}", index, int(ret));
                return false;
            }

            auto buffer_size = aclmdlGetInputSizeByIndex(m_model_desc, index);
            void *data_mem_buffer = nullptr;
            if (!m_is_dynamic_input && !createDataBuffer(&data_mem_buffer, buffer_size, m_input_dataset))
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "add input:{} data buffer failed, buffer size {}", index, buffer_size);
                return false;
            }
            aclDataType data_type = aclmdlGetInputDataType(m_model_desc, index);
            std::vector<int64_t> shape(dims.dims, dims.dims + dims.dimCount);
            if (!m_is_dynamic_input)
            {
                aclFormat input_format = aclmdlGetInputFormat(m_model_desc, index);
                aclTensorDesc *desc = aclCreateTensorDesc(data_type, dims.dimCount, dims.dims, input_format);
                ret = aclmdlSetDatasetTensorDesc(m_input_dataset, desc, index);
                if (ACL_ERROR_NONE != ret)
                {
                    ACL_LOG(ACL_LOG_LEVEL_ERROR, "aclmdlSetDatasetTensorDesc failed, ret:{}", int(ret));
                    return false;
                }
            }
            std::string input_name = aclmdlGetInputNameByIndex(m_model_desc, index);
            if (input_name.empty())
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "get name of input:{} failed", index);
                return false;
            }
            ACL_LOG(ACL_LOG_LEVEL_DEBUG, "name of input:{} is {}", index, input_name);
            m_input_infos.emplace_back(AclTensorInfo{data_mem_buffer, data_mem_buffer, buffer_size, buffer_size, 
                data_type, shape, input_name});
        }
        ACL_LOG(ACL_LOG_LEVEL_DEBUG, "create model input dataset success");
        return true;
    }

    bool AscendCLEngine::initOutputsBuffer()
    {
        // check model desc valid
        if (nullptr == m_model_desc)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl model desc is nullptr");
            return false;
        }

        // create output dataset
        aclError ret;
        m_output_dataset = aclmdlCreateDataset();
        if (nullptr == m_output_dataset)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "create output dataset failed");
            return false;
        }

        // init output infos and output dataset
        size_t output_size = aclmdlGetNumOutputs(m_model_desc);
        ACL_LOG(ACL_LOG_LEVEL_DEBUG, "model output_size = {}", output_size);
        for (size_t index = 0; index < output_size; ++index)
        {
            aclmdlIODims dims;
            ret = aclmdlGetOutputDims(m_model_desc, index, &dims);
            if (ACL_ERROR_NONE != ret)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "get output:{} shape failed, ret:{}", index, int(ret));
                return false;
            }
            bool is_dynamic_output = false;
            for (size_t dim_idx = 0; dim_idx < dims.dimCount; dim_idx++)
            {
                is_dynamic_output = (dims.dims[dim_idx] < 0) ? true : false;
                break;
            }
            size_t buffer_size = 0;
            if (!is_dynamic_output)
            {
                buffer_size = aclmdlGetOutputSizeByIndex(m_model_desc, index);
            }
            void *data_mem_buffer = nullptr;
            if (!is_dynamic_output && !createDataBuffer(&data_mem_buffer, buffer_size, m_output_dataset))
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "create output data buffer failed, buffer size {}", buffer_size);
                return false;
            }
            aclFormat format = aclmdlGetOutputFormat(m_model_desc, index);
            ACL_LOG(ACL_LOG_LEVEL_DEBUG, "the output:{} format is {}", index, int(format));
            aclDataType data_type = aclmdlGetOutputDataType(m_model_desc, index);
            std::vector<int64_t> shape(dims.dims, dims.dims + dims.dimCount);
            if (is_dynamic_output)
            {
                shape = std::vector<int64_t>({-1});
            }
            std::string output_name = aclmdlGetOutputNameByIndex(m_model_desc, index);
            if (output_name.empty())
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "get name of output {} failed", index);
                return false;
            }
            // 由于ascend_cl输出名称会自动加前缀且用:分开
            auto pos = output_name.rfind(":");
            if (pos != std::string::npos)
                output_name = output_name.substr(pos + 1);
            ACL_LOG(ACL_LOG_LEVEL_DEBUG, "name of om output {} is {} buffer size {}", index, output_name, buffer_size);
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
        for (size_t index = 0; index < aclmdlGetDatasetNumBuffers(m_input_dataset); index++)
        {
            auto dataBuffer = aclmdlGetDatasetBuffer(m_input_dataset, index);
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
        for (size_t index = 0; index < aclmdlGetDatasetNumBuffers(m_output_dataset); index++)
        {
            auto dataBuffer = aclmdlGetDatasetBuffer(m_output_dataset, index);
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
        m_is_run_on_device = (ACL_DEVICE == run_mode);
        ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl model is running in {} mode", int(run_mode));

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

        // create model desc
        m_model_desc = aclmdlCreateDesc();
        if (nullptr == m_model_desc)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "create acl model desc failed");
            return -1;
        }

        // get model desc by model_id
        ret = aclmdlGetDesc(m_model_desc, m_model_id);
        if (ACL_ERROR_NONE != ret)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "get acl model desc failed, ret:{}, msg:{}", int(ret), aclGetRecentErrMsg());
            return -1;
        }

        // init dynamic batch size options
        if (0 != getDynamicBatch(m_dynamic_shape_options.batch_size))
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "init dynamic shape option's batch size fail");
            return -1;
        }

        // init dynamic image size options
        if (0 != getDynamicImage(m_dynamic_shape_options.image_size))
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "init dynamic shape option's image size fail");
            return -1;
        }

        // init dynamic dims options
        if (0 != getDynamicDims(m_dynamic_shape_options.dynamic_dims))
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "init dynamic shape option's dynamic dims fail");
            return -1;
        }

        // check and set dynamic flag
        if (!checkAndSetDynFlag())
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "check and set dynamic flag failed");
            return -1;
        }

        // init input infos and dataset
        if (!initInputsBuffer())
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "create input buffer failed");
            return -1;
        }

        // init output infos and dataset
        if (!initOutputsBuffer())
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "create output buffer failed");
            return -1;
        }

        // dynamic input, input num = input info size
        if (m_is_dynamic_input)
        {
            m_data_input_num = m_input_infos.size();
            return 0;
        }

        // dynamic shape, input num = input info size - 1
        // when use atc tool convert model with dynamic_batch_size/dynamic_image_size/dynamic_dims
        // final om model will add a new input named ACL_DYNAMIC_TENSOR_NAME
        // #define ACL_DYNAMIC_TENSOR_NAME "ascend_mbatch_shape_data"
        m_data_input_num = m_input_infos.size();
        if (isDynamicShape() && 0 < m_data_input_num)
        {
            m_data_input_num -= 1;
        }

        // init input format options
        if (0 != getInputFormat(m_dynamic_shape_options.input_format))
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "init dynamic shape option's input format fail");
            return -1;
        }

        // init input shape options
        if (0 != getInputShape(m_dynamic_shape_options.input_shapes))
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "init dynamic shape option's input shape fail");
            return -1;
        }

        // init dynamic shape process with dynamic shape option
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
            model_datas.emplace_back(file_data);
            data_lens.emplace_back(file_len);
            file_streams.emplace_back(file_stream);
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
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl model has not been loaded");
            return -1;
        }

        // check input tensors size valid
        if (0 == input_tensors_map.size())
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl engine input tensors is empty");
            return -1;
        }

        // construct new shapes
        std::vector<std::vector<int64_t>> new_shapes;
        for (auto index = 0; index < m_data_input_num; index++)
        {
            auto& input = m_input_infos[index];
            std::string input_name = input.name;
            if (input_tensors_map.find(input_name) == input_tensors_map.end())
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl engine input tensors map cannot find tensor: {}", input_name);
                return -1;
            }
            std::vector<int64_t>& input_shape = input_tensors_map[input_name]->shape();
            new_shapes.emplace_back(input_shape);
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
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl engine create input engine tensor {} fail", tensor_name);
                return -1;
            }
            m_input_tensors_map[tensor_name] = temp_tensor;
        }

        return 0;
    }

    int AscendCLEngine::getEngineOutputTensors(std::map<std::string, EngineTensor*>& output_tensors_map)
    {
        // check model valid
        if (!m_status)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl model has not been loaded");
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
        return 0;
    }

    int AscendCLEngine::resizeEngine(std::map<std::string, std::vector<int64_t>>& new_shapes)
    {
        // check model valid
        if (!m_status)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl model has not been loaded");
            return false;
        }

        // set current context
        auto ret = aclrtSetCurrentContext(m_context);
        if (ACL_ERROR_NONE != ret)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl set context failed, ret:{}, msg:{}", int(ret), aclGetRecentErrMsg());
            return -1;
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
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl model has not been loaded");
            return false;
        }

        // set current context
        auto ret = aclrtSetCurrentContext(m_context);
        if (ACL_ERROR_NONE != ret)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl set context failed, ret:{}, msg:{}", int(ret), aclGetRecentErrMsg());
            return -1;
        }

        // get input tensors
        std::vector<EngineTensor*> input_tensors;
        for (auto index = 0; index < m_data_input_num; index++)
        {
            auto& input = m_input_infos[index];
            std::string input_name = input.name;
            if (m_input_tensors_map.find(input_name) == m_input_tensors_map.end())
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl engine input tensors map cannot find tensor: {}", input_name);
                return -1;
            }
            input_tensors.emplace_back(m_input_tensors_map[input_name].get());
        }

        // check and init input tensors
        if (!checkAndInitInput(input_tensors))
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "check or init input tensors failed");
            return false;
        }

        // get output tensors
        std::vector<std::shared_ptr<EngineTensor>> output_tensors;
        for (auto index = 0; index < m_output_infos.size(); index++)
        {
            auto& output = m_output_infos[index];
            std::string output_name = output.name;
            if (m_output_tensors_map.find(output_name) == m_output_tensors_map.end())
            {
                // ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl engine output tensors map cannot find tensor: {}", output_name);
                // return -1;
                continue;
            }
            output_tensors.emplace_back(m_output_tensors_map[output_name]);
        }

        // check and init output tensors
        if (!checkAndInitOutput(output_tensors))
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "check or init output tensors failed");
            return false;
        }

        // model execute
        auto ret = aclmdlExecute(m_model_id, m_input_dataset, m_output_dataset);
        if (ACL_ERROR_NONE != ret)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "execute model failed, ret:{}, msg:{}", int(ret), aclGetRecentErrMsg());
            return -1;
        }

        // reset dynamic output tensors
        if (m_is_dynamic_output)
        {
            output_tensors.clear();
            bool ret = resetDynamicOutputTensor(output_tensors);
            if (!ret)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "reset dyanmic output tensor fail");
                return -1;
            }
        }

        // copy output tensor data
        if (!getOutputs(output_tensors))
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "build output tensors failed");
            return -1;
        }

        // update output tensors
        for (auto index = 0; index < m_output_infos.size(); index++)
        {
            auto& output = m_output_infos[index];
            std::string output_name = output.name;
            m_output_tensors_map[output_name] = output_tensors[index];
        }

        // The device_data is malloced by acl, user need to free the addr
        if (m_is_dynamic_output)
        {
            freeResourceOutput(&m_output_infos);
        }

        return 0;
    }

    int AscendCLEngine::runEngine(std::map<std::string, EngineTensor*>& input_tensors_map, 
        std::map<std::string, EngineTensor*>& output_tensors_map)
    {
        // check model valid
        if (!m_status)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl model has not been loaded");
            return false;
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
        // check model valid
        if (!m_status)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl model has not been loaded");
            return -1;
        }
        input_tensor_infos.clear();

        for (auto index = 0; index < m_input_infos.size(); index++)
        {
            auto input_info = m_input_infos[index];
            // get input tensor name
            std::string tensor_name = input_info.name;
            // get input tensor shape
            std::vector<int64_t> tensor_shape = input_info.dims;
            // get input tensor datatype
            aclDataType tensor_dtype = tensor_info.data_type;

            EngineTensorInfo tensor_info;
            tensor_info.name = tensor_name;
            tensor_info.type = convertAscendCLTypeToTensorType(tensor_dtype);
            tensor_info.shape = tensor_shape;
            input_tensor_infos.emplace_back(tensor_info);
        }
        return 0;
    }

    int AscendCLEngine::getOutputTensorInfos(std::vector<EngineTensorInfo>& output_tensor_infos)
    {
        // check model valid
        if (!m_status)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl model has not been loaded");
            return -1;
        }
        output_tensor_infos.clear();

        for (auto index = 0; index < m_output_infos; index++)
        {
            auto output_info = m_output_infos[index];
            // get output tensor name
            std::string tensor_name = output_info.name;
            // get output tensor shape
            std::vector<int64_t> tensor_shape = output_info.dims;
            // get output tensor datatype
            aclDataType tensor_dtype = output_info.data_type;

            EngineTensorInfo tensor_info;
            tensor_info.name = tensor_name;
            tensor_info.type = convertACLTypeToTensorType(tensor_dtype);
            tensor_info.shape = tensor_shape;
            output_tensor_infos.emplace_back(tensor_info);
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
        // check model desc valid
        if (nullptr == m_model_desc)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl model desc is nullptr");
            return false;
        }

        // reset output info size
        size_t output_size = aclmdlGetNumOutputs(m_model_desc);
        for (size_t index = 0; index < output_size; index++)
        {
            struct aclmdlIODims dims;
            aclError ret = aclmdlGetCurOutputDims(m_model_desc, index, &dims);
            if (ACL_ERROR_NONE != ret)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "get output {} dims error", index);
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
            aclDataType data_type = aclmdlGetOutputDataType(m_model_desc, index);
            m_output_infos[index].dims = shape;
            m_output_infos[index].buffer_size = elem_count * aclDataTypeSize(data_type);
        }
        return true;
    }

    bool AscendCLEngine::resizeDynamicInputShape(const std::vector<std::vector<int64_t>>& new_shapes)
    {
        ACL_LOG(ACL_LOG_LEVEL_DEBUG, "start to resize dynamic input shape");
        // If it is not the first time to resize input shape, the old addr need to be free
        resetInputSize(new_shapes);
        freeResourceInput(m_input_infos);
        if (m_is_dynamic_resize_input)
        {
            m_input_dataset = aclmdlCreateDataset();
            if (nullptr == m_input_dataset)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "create input dataset failed");
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
            if (ACL_ERROR_NONE != ret)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl set dataset tensor desc failed, ret:{}", int(ret));
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
        // check model desc and input dataset valid
        if (m_model_desc == nullptr || m_input_dataset == nullptr)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "model is not inited");
            return false;
        }
        size_t index;
        auto ret = aclmdlGetInputIndexByName(m_model_desc, ACL_DYNAMIC_TENSOR_NAME, &index);
        if (ACL_ERROR_NONE != ret)
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
            if (ACL_ERROR_NONE != ret)
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
            if (ACL_ERROR_NONE != ret)
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
            if (ACL_ERROR_NONE != ret)
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
                    spdlog::fmt_lib::join(new_shape, ", "));
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

    bool AscendCLEngine::checkInputTensors(const std::vector<EngineTensor*>& input_tensors)
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
                    "please check input shape has been modified by DVPP method.", i, 
                    spdlog::fmt_lib::join(info.dims, ", "), spdlog::fmt_lib::join(tensor->shape(), ", "));
                return false;
            }
            if (tensor->getTensorDataType() != convertAscendCLTypeToTensorType(info.data_type))
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "note: input {} data type not match, required {}, but given {}", i, 
                    static_cast<int>(convertAscendCLTypeToTensorType(info.data_type)), 
                    static_cast<int>(tensor->getTensorDataType()));
                return false;
            }
            auto host_data = tensor->host<void>();
            auto host_size = (size_t)tensor->size();
            if (nullptr != host_data)
            {
                if (!m_is_dynamic_input && !m_is_dynamic_shape_range && host_size != info.buffer_size)
                {
                    ACL_LOG(ACL_LOG_LEVEL_ERROR, "input {} data size not match, required size {}, but given count {}", i, 
                        info.buffer_size, host_size);
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

    bool AscendCLEngine::checkOutputTensors(std::vector<std::shared_ptr<EngineTensor>>& outputs)
    {
        // check output tensor size equal to model output info
        if (outputs.size() && outputs.size() != m_output_infos.size())
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "actual tensor count not match, required count {}, but got {}", 
                m_output_infos.size(), outputs.size());
            return false;
        }

        // dynamic output no need check
        if (m_is_dynamic_output)
        {
            ACL_LOG(ACL_LOG_LEVEL_DEBUG, "this model has dynamic output shape");
            return true;
        }

        // check output tensor's shape/dtype/size equal to model output info
        for (size_t index = 0; index < m_output_infos.size(); ++index)
        {
            auto& output_info = m_output_infos[index];
            auto output_shape = output_info.dims;
            auto output_dtype = convertAscendCLTypeToTensorType(output_info.data_type);
            auto output_format = EngineTensor::TENSOR_FORMAT_TYPE_NCHW;
            if (index >= outputs.size())
            {
                std::shared_ptr<EngineTensor> tmp_tensor(EngineTensor::create(output_shape, output_dtype, output_format));
                if (nullptr == tmp_tensor.get())
                {
                    ACL_LOG(ACL_LOG_LEVEL_ERROR, "create engine tensor for output {} fail", index);
                    return -1;
                }
                outputs.emplace_back(tmp_tensor);
            }
            auto& output_tensor = outputs[index];
            if (output_tensor->shape() != output_info.dims)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "note: output {} shape not match, required {}, but given {} "
                    "please check output shape.", index, spdlog::fmt_lib::join(output_info.dims, ", "), 
                    spdlog::fmt_lib::join(output_tensor->shape(), ", "));
            }
            if (output_tensor->getTensorDataType() != output_dtype)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "note: output {} data type not match, required {}, but given {}", 
                    index, int(output_dtype), int(output_tensor->getTensorDataType()));
                return false;
            }
            auto host_data = output_tensor->host<void>();
            auto host_size = (size_t)output_tensor->size();
            if (nullptr != host_data)
            {
                if (host_size != info.buffer_size)
                {
                    ACL_LOG(ACL_LOG_LEVEL_ERROR, "output {} host data size not match, required size {}, but given count {}", 
                        index, info.buffer_size, output_tensor->size());
                    return false;
                }
            }
            else
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "failed to get data from output {}", index);
                return false;
            }
        }
        return true;
    }

    void AscendCLEngine::checkAndInitDynOutputDeviceBuf(const EngineTensor* output, const AclTensorInfo& output_info,
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
        // check inputs valid
        if (!checkInputTensors(inputs))
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "check input tensors failed");
            return false;
        }
        aclError ret;
        // copy inputs tensor data to input dataset
        for (size_t index = 0; index < inputs.size(); ++index)
        {
            auto &info = m_input_infos[index];
            auto input = inputs[index];
            void *input_buffer = nullptr;
            auto input_data = input->host<void>();
            auto input_size = (size_t)input->size();
            if (!m_is_run_on_device)
            {
                ret = aclrtMemcpy(info.device_data, info.buffer_size, input_data, input_size, ACL_MEMCPY_HOST_TO_DEVICE);
                if (ACL_ERROR_NONE != ret)
                {
                    ACL_LOG(ACL_LOG_LEVEL_ERROR, "acl memcpy input {} data to device failed, src input size: {}"
                        ", dst device buffer size: {}", index, size, info.buffer_size);
                    return false;
                }
                input_buffer = info.device_data;
            }
            else
            {
                input_buffer = input_data;
            }
            auto data_buffer = aclmdlGetDatasetBuffer(m_input_dataset, index);
            if (nullptr == data_buffer)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "failed to get dataset buffer of input {}", index);
                return false;
            }
            ret = aclUpdateDataBuffer(data_buffer, input_buffer, info.buffer_size);
            if (ACL_ERROR_NONE != ret)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "failed to update data buffer of input {}, buffer size: {}, input shape: {}", 
                    index, info.buffer_size, spdlog::fmt_lib::join(input->shape(), ", "));
                return false;
            }
        }
        return true;
    }

    bool AscendCLEngine::checkAndInitOutput(std::vector<std::shared_ptr<EngineTensor>>& outputs)
    {
        // check outputs
        if (!checkOutputTensors(outputs))
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "check output tensor failed");
            return false;
        }
        aclError ret;
        // copy outputs
        for (size_t index = 0; index < m_output_infos.size(); ++index)
        {
            auto &info = m_output_infos[index];
            auto output_device_buffer_size = info.buffer_size;
            void *output_device_buffer = nullptr;
            if (m_is_dynamic_output)
            {
                output_device_buffer = nullptr;  // in dynamic output shape, setting nullptr allows acl to alloc memory
                output_device_buffer_size = 0;
                // checkAndInitDynOutputDeviceBuf(output, info, &output_device_buffer, &output_device_buffer_size, i);
            }
            else
            {
                auto output = outputs[index];
                auto host_data = output->host<void>();
                if (m_is_run_on_device)
                {
                    output_device_buffer = host_data;
                }
                else
                {
                    output_device_buffer = info.device_data;
                }
            }

            auto data_buffer = aclmdlGetDatasetBuffer(m_output_dataset, index);
            if (nullptr == data_buffer)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "failed to get dataset buffer of output {}", index);
                return false;
            }
            ret = aclUpdateDataBuffer(data_buffer, output_device_buffer, output_device_buffer_size);
            if (ACL_ERROR_NONE != ret)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "failed to update data buffer of output {}, buffer size: {}, output shape: {}", 
                    index, info.buffer_size, spdlog::fmt_lib::join(output->shape(), ", "));
                return false;
            }
        }
        return true;
    }

    void AscendCLEngine::freeResourceInput(std::vector<AclTensorInfo>& acl_tensor_info)
    {
        for (const auto &item : acl_tensor_info)
        {
            if (nullptr != item.dynamic_acl_tensor_desc)
            {
                aclDestroyTensorDesc(item.dynamic_acl_tensor_desc);
            }
            if (m_is_dynamic_resize_input)
            {
                if (nullptr != item.device_data)
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
                if (nullptr != item.dynamic_acl_data_buffer)
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
        return;
    }

    void AscendCLEngine::freeResourceOutput(std::vector<AclTensorInfo>& acl_tensor_info)
    {
        for (size_t i = 0; i < acl_tensor_info->size(); i++)
        {
            auto &item = (*acl_tensor_info)[i];
            if (item.device_data != nullptr)
            {
                ACL_LOG(ACL_LOG_LEVEL_DEBUG, "freeing device buffer at addr: 0x{:x}", (size_t)item.device_data);
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

    bool AscendCLEngine::resetDynamicOutputTensor(std::vector<std::shared_ptr<EngineTensor>>& outputs)
    {
        for (size_t index = 0; index < m_output_infos.size(); ++index)
        {
            auto& output_info = m_output_infos[index];
            // get actual output tensor info
            aclTensorDesc *tensor_info = aclmdlGetDatasetTensorDesc(m_output_dataset, index);
            size_t output_desc_size = aclGetTensorDescSize(tensor_info);
            if (0 == output_desc_size)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "dynamic output size from acl inference result is 0, please check graph or inputs");
                return false;
            }

            // get dynamic output tensor shape
            size_t dim_nums = aclGetTensorDescNumDims(tensor_info);
            std::vector<int64_t> output_shape;
            for (size_t j = 0; j < dim_nums; ++j)
            {
                int64_t shape_j = aclGetTensorDescDim(tensor_info, j);
                output_shape.emplace_back(shape_j);
            }

            // get dynamic output tensor dtype
            aclDataType acl_dtype = aclGetTensorDescType(tensor_info);
            auto output_dtype = convertAscendCLTypeToTensorType(acl_dtype);

            // get dynamic output tensor format
            aclFormat acl_format = aclGetTensorDescFormat(tensor_info);
            static const std::map<aclFormat, EngineTensor::TensorFormatType> acl_format_map = {
                {ACL_FORMAT_NCHW, EngineTensor::TENSOR_FORMAT_TYPE_NCHW}, 
                {ACL_FORMAT_NHWC, EngineTensor::TENSOR_FORMAT_TYPE_NHWC},
                {ACL_FORMAT_ND, EngineTensor::TENSOR_FORMAT_TYPE_NCHW}};
            auto iter = acl_format_map.find(acl_format);
            if (iter != acl_format_map.end())
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "aclFormat {} not found in map, please double check and add...using default format", 
                    static_cast<int32_t>(format));
                return false;
            }
            auto output_format = acl_format_map[acl_format];

            // create output tensor
            std::shared_ptr<EngineTensor> tmp_tensor(EngineTensor::create(output_shape, output_dtype, output_format));
            if (nullptr == tmp_tensor.get())
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "create engine tensor for output {} fail", index);
                return false;
            }

            // push to output tensor vector
            outputs.emplace_back(tmp_tensor);

            // update acl tensor info
            aclDataBuffer *data_buffer = aclmdlGetDatasetBuffer(m_output_dataset, index);
            void *acl_device_data = aclGetDataBufferAddr(data_buffer);

            output_info.device_data = acl_device_data;
            output_info.cur_device_data = acl_device_data;
            output_info.buffer_size = output_desc_size;
            output_info.malloc_buffer_size = output_desc_size;
        }
        return true;
    }

    bool AscendCLEngine::getOutputs(const std::vector<std::shared_ptr<EngineTensor>>& outputs)
    {
        aclrtMemcpyKind kind = m_is_run_on_device ? ACL_MEMCPY_HOST_TO_HOST : ACL_MEMCPY_DEVICE_TO_HOST;
        for (size_t index = 0; index < m_output_infos.size(); ++index)
        {
            auto& output_tensor = outputs[index];
            auto& output_info = m_output_infos[index];
            if (nullptr == output_info.cur_device_data)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "output {} device addr is nullptr", index);
                return false;
            }
            auto host_data = output_tensor->host<void>();
            auto host_size = (size_t)output_tensor->size();
            if (nullptr != host_data)
            {
                if (host_size != output_info.buffer_size)
                {
                    ACL_LOG(ACL_LOG_LEVEL_ERROR, "specified output host data size {} != execute output data size {}, output_shape: {}", 
                        host_size, output_info.buffer_size, spdlog::fmt_lib::join(output_info.dims, ", "));
                    return false;
                }
                ACL_LOG(ACL_LOG_LEVEL_DEBUG, "copying to host with addr: 0x{:x} with size: {}", (size_t)host_data->addr, 
                    output_info.buffer_size);
                auto ret = aclrtMemcpy(host_data, host_size, output_info.cur_device_data, output_info.buffer_size, kind);
                if (ACL_ERROR_NONE != ret)
                {
                    ACL_LOG(ACL_LOG_LEVEL_ERROR, "memcpy output {} from {} to host failed, memory size {}, ret: {}", 
                        index, (m_is_run_on_device ? "host" : "device"), output_info.buffer_size, int(ret));
                    return false;
                }
            }
            else
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "output {} host addr is nullptr", index);
                return false;
            }
        }
        return true;
    }

} // namespace ACL_ENGINE