// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#pragma once
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_memory.h"
#include "triton/backend/backend_output_responder.h"
#include "triton/backend/backend_model_instance.h"
#include "acl_engine/acl_engine.h"
#include "model_state.h"
#include "acl_utils.h"

using namespace ACL_ENGINE;

namespace triton::backend::acl
{

    class ModelInstanceState : public BackendModelInstance
    {
    public:
        static TRITONSERVER_Error* Create(ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance, ModelInstanceState** state);
        virtual ~ModelInstanceState() = default;
        // Get the state of the model that corresponds to this instance.
        ModelState* StateForModel() const { return model_state_; }
        void ProcessRequests(TRITONBACKEND_Request** requests, const uint32_t request_count);

    private:
        ModelInstanceState(ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance);
        TRITONSERVER_Error* DetermineModelPath(const std::string& model_dir, std::string* model_path, std::string* config_path);
        TRITONSERVER_Error* CreateTensor(const char* input_name, const std::vector<int64_t> shape, TRITONSERVER_DataType triton_dtype, 
            int batchn_byte_size, TRITONSERVER_MemoryType mem_type, int mem_type_id, std::shared_ptr<AclTensor>& tensor,
            void* data = nullptr, bool clone_flag = false);
        TRITONSERVER_Error* CreateStringTensor(const char* input_name, const std::vector<int64_t> shape, TRITONSERVER_DataType triton_dtype, 
            TRITONSERVER_MemoryType mem_type, int mem_type_id, std::shared_ptr<AclTensor>& tensor);
        TRITONSERVER_Error* RunAclModel(std::vector<std::string>& input_names, std::map<std::string, std::shared_ptr<AclTensor>>& input_tensors);
        TRITONSERVER_Error* GetAclModelOutputs(std::vector<std::string>& output_names, std::map<std::string, AclTensor*>& output_tensors);

        // input tensors funcs
        void FillStringData(std::vector<const char*>* string_ptrs, size_t cnt);
        void SetStringInputBuffer(const std::string& input_name, const std::vector<size_t>& expected_byte_sizes,
            const std::vector<size_t>& expected_element_cnts, std::vector<TRITONBACKEND_Response*>* responses, 
            char* input_buffer, std::vector<const char*>* string_ptrs);
        TRITONSERVER_Error* SetStringInputTensor(TRITONBACKEND_Request** requests, const uint32_t request_count,
            std::vector<TRITONBACKEND_Response*>* responses, const char* input_name, 
            std::shared_ptr<BackendMemory>& backend_memory, std::vector<const char*>* string_ptrs, bool* cuda_copy);
        TRITONSERVER_Error* SetInputTensors(size_t total_batch_size, TRITONBACKEND_Request** requests, 
            const uint32_t request_count, std::vector<TRITONBACKEND_Response*>* responses, 
            BackendInputCollector* collector, std::vector<std::string>& input_names, 
            std::map<std::string, std::shared_ptr<AclTensor>>& input_tensors, 
            std::vector<std::shared_ptr<BackendMemory>>& backend_memorys, bool* cuda_copy);

        // output tensors funcs
        bool SetStringBuffer(const std::string& name, const char* content, const size_t* offsets,
            std::vector<int64_t>* batchn_shape, TRITONBACKEND_Request** requests, const uint32_t request_count,
            std::vector<TRITONBACKEND_Response*>* responses, bool state);
        bool SetStringStateBuffer(const std::string& name, const char* content, const size_t* offsets,
            std::vector<int64_t>* batchn_shape, TRITONBACKEND_Request** requests, const uint32_t request_count, 
            std::vector<TRITONBACKEND_Response*>* responses);
        bool SetStringOutputBuffer(const std::string& name, const char* content, const size_t* offsets,
            std::vector<int64_t>* batchn_shape, TRITONBACKEND_Request** requests, const uint32_t request_count,
            std::vector<TRITONBACKEND_Response*>* responses);
        TRITONSERVER_Error* ReadOutputTensor(const std::string& name, std::vector<int64_t>& batchn_shape, 
            TRITONSERVER_DataType& dtype, AclTensor* output_tensor, void** output_buffer, 
            std::vector<std::vector<char>>& string_buffers, std::vector<size_t>& offsets);
        TRITONSERVER_Error* ReadOutputTensors(size_t total_batch_size, TRITONBACKEND_Request** requests, 
            const uint32_t request_count, std::vector<TRITONBACKEND_Response*>* responses);

    private:
        ModelState*                                         model_state_;
        std::shared_ptr<AscendCLEngine>                     acl_engine_;
    };

} // namespace triton::backend::acl
