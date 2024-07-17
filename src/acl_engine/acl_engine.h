/********************************************
 * @Author: zhaojd-a
 * @Date: 2024-06-13
 * @LastEditTime: 2024-06-13
 * @LastEditors: zhaojd-a
 ********************************************/
#pragma once
#include <iostream>
#include <string>
#include <set>
#include <map>
#include <cstring>
#include <memory>
#include "acl_engine/non_copyable.h"
#include "acl_engine/log.h"
#include "acl_engine/engine_type.h"
#include "acl_engine/engine_tensor.h"
#include "acl_engine/dyn_shape_process.h"
#include "acl/acl.h"

namespace ACL_ENGINE
{

    typedef struct AclTensorInfo
    {
        void*                                           cur_device_data;
        void*                                           device_data;
        size_t                                          buffer_size;
        size_t                                          malloc_buffer_size;
        aclDataType                                     data_type;
        std::vector<int64_t>                            dims;
        std::string                                     name;
        aclTensorDesc*                                  dynamic_acl_tensor_desc = nullptr;
        aclDataBuffer*                                  dynamic_acl_data_buffer = nullptr;
    } AclTensorInfo;

    class AscendCLInitSingleton
    {
    public:
        static AscendCLInitSingleton& Instance()
        {
            static AscendCLInitSingleton acl_instance;
            return acl_instance;
        }

    private:
        AscendCLInitSingleton()
        {
            const char* acl_config_path = "";
            aclError ret = aclInit(acl_config_path);
            if (ACL_SUCCESS != ret)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "aclInit failed, ret:{}", ret);
            }
        };
        ~AscendCLInitSingleton() {};
        AscendCLInitSingleton(const AscendCLInitSingleton &);
        const AscendCLInitSingleton &operator=(const AscendCLInitSingleton &);
    };

    class AscendCLEngine : NonCopyable
    {
    public:
        AscendCLEngine(const EngineConfig& config, const std::vector<std::string>& model_files);
        AscendCLEngine(const EngineConfig& config, const std::vector<const char*>& model_datas, 
            const std::vector<size_t>& data_lens);
        ~AscendCLEngine();

    public:
        bool status() { return m_status; }
        int setEngineInputTensors(std::map<std::string, EngineTensor*>& input_tensors_map);
        int getEngineOutputTensors(std::map<std::string, EngineTensor*>& output_tensors_map);
        int resizeEngine(std::map<std::string, std::vector<int64_t>>& new_shapes);
        int runEngine();
        int runEngine(std::map<std::string, EngineTensor*>& input_tensors_map, 
            std::map<std::string, EngineTensor*>& output_tensors_map);
        void printEngineInfo();
        int getInputTensorInfos(std::vector<EngineTensorInfo>& input_tensor_infos);
        int getOutputTensorInfos(std::vector<EngineTensorInfo>& output_tensor_infos);

    private:
        std::set<uint64_t> getDynamicBatch();
        std::set<std::pair<uint64_t, uint64_t>> getDynamicImage();
        std::pair<aclmdlIODims*, size_t> getDynamicDims();
        const std::vector<ShapeVector> getOutputShape();
        const std::vector<ShapeVector> getInputShape();
        const std::vector<TypeId> getInputDataType();
        const std::vector<TypeId> getOutputDataType();
        int CheckAndSetDynFlag();
        bool initInputsBuffer();
        bool initOutputsBuffer();
        void destroyInputsBuffer();
        void destroyOutputsBuffer();
        bool createDataBuffer(void** data_mem_buffer, size_t buffer_size, aclmdlDataset* dataset);
        bool isDynamicShape();
        bool isDynamicBatchSize();
        bool isDynamicImageSize();
        bool isDynamicDims();
        int checkEngineConfig(const EngineConfig& config);
        int initAclModelFromBuffer(const char* model_data, const size_t& data_len, const EngineConfig& acl_config);
        int loadModelFromFile(const EngineConfig& config, const std::vector<std::string>& model_files);
        int loadModelFromBuffer(const EngineConfig& config, const std::vector<const char*>& model_datas, 
            const std::vector<size_t>& data_lens);

        bool resizeDynamicInputShape(const std::vector<std::vector<int64_t>>& new_shapes);
        bool resizeDynamicInputShapeRange(const std::vector<std::vector<int64_t>>& new_shapes);
        bool resizeDynamicBatchAndImageSize(const std::vector<std::vector<int64_t>>& new_shapes);
        bool resize(const std::vector<std::vector<int64_t>>& new_shapes);

    private:
        bool                                                               m_status = false;
        // ascendcl model
        aclrtContext                                                       m_context = nullptr;
        aclrtStream                                                        m_stream = nullptr;
        aclmdlDesc*                                                        m_model_desc = nullptr;
        aclmdlDataset*                                                     m_input_dataset = nullptr;
        aclmdlDataset*                                                     m_output_dataset = nullptr;
        uint32_t                                                           m_model_id = UINT32_MAX;

        // utils member var
        std::vector<AclTensorInfo>                                         m_input_infos;
        std::vector<AclTensorInfo>                                         m_output_infos;
        // if run one device(AICPU), there is no need to alloc device memory and copy inputs to(/outputs from) device
        bool                                                               m_is_run_on_device = false;
        AclDynamicShapeOptions                                             m_dynamic_shape_options;
        aclmdlIODims*                                                      m_dynamic_dims = nullptr;
        bool                                                               m_is_dynamic_output = false;
        bool                                                               m_is_dynamic_input = false;
        bool                                                               m_is_dynamic_resize_input = false;
        bool                                                               m_is_dynamic_shape_range = false;
        size_t                                                             m_data_input_num = 0;
        DynShapeProcess                                                    m_dyn_shape_proc;
        // acl engine config
        EngineConfig                                                       m_engine_config;

        // acl model inputs/outputs
        std::map<std::string, std::shared_ptr<EngineTensor>>               m_input_tensors_map;
        std::map<std::string, std::shared_ptr<EngineTensor>>               m_output_tensors_map;
    };

} // namespace ACL_ENGINE