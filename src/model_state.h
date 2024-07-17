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
#include "triton/backend/backend_model.h"
#include "acl_engine/engine_type.h"

namespace triton::backend::acl
{

    class ModelState : public BackendModel
    {
    public:
        static TRITONSERVER_Error* Create(TRITONBACKEND_Model* triton_model, ModelState** state);
        virtual ~ModelState() = default;
        const std::vector<std::string>& InputNames() const { return input_names_; }
        const std::vector<std::string>& OutputNames() const { return output_names_; }
        const std::vector<TRITONSERVER_DataType>& InputDataTypes() const { return input_data_types_; }
        const std::vector<TRITONSERVER_DataType>& OutputDataTypes() const { return output_data_types_; }
        const std::vector<std::string>& InputFormats() const { return input_formats_; }
        const std::map<std::string, std::pair<int64_t, int64_t>>& ModelOutputs() { return model_outputs_; }
        const ACL_ENGINE::EngineConfig AclEngineConfig() { return acl_config_; }

    private:
        ModelState(TRITONBACKEND_Model* triton_model);
        // Validate that model configuration is supported by this backend
        TRITONSERVER_Error* ValidateModelConfig();
        // Auto-complete the model configuration
        TRITONSERVER_Error* AutoCompleteConfig();
        // parse model config paramerters
        TRITONSERVER_Error* ParseBoolParameter(triton::common::TritonJson::Value& params, const std::string& mkey, bool* value);
        TRITONSERVER_Error* ParseIntParameter(triton::common::TritonJson::Value& params, const std::string& mkey, int* value);
        TRITONSERVER_Error* ParseStrParameter(triton::common::TritonJson::Value& params, const std::string& mkey, std::string& value);
        TRITONSERVER_Error* ParseDoubleParameter(triton::common::TritonJson::Value& params, const std::string& mkey, double* value);
        TRITONSERVER_Error* ParseParameters();

        // model_outputs is a map that contains unique outputs that the model must
        // provide. In the model configuration, the output in the state configuration
        // can have intersection with the outputs section of the model. If an output
        // is specified both in the output section and state section, it indicates
        // that the backend must return the output state to the client too.
        std::map<std::string, std::pair<int64_t, int64_t>>   model_outputs_;

        // model inputs/ouptus tensors info
        std::vector<std::string>                             input_names_;
        std::vector<std::string>                             output_names_;
        std::vector<TRITONSERVER_DataType>                   input_data_types_;
        std::vector<TRITONSERVER_DataType>                   output_data_types_;
        std::vector<std::string>                             input_formats_;
        std::map<std::string, std::vector<int64_t>>          input_dims_;
        std::map<std::string, std::vector<int64_t>>          output_dims_;

        // acl engine config
        ACL_ENGINE::EngineConfig                             acl_config_;
    };

} // namespace triton::backend::acl