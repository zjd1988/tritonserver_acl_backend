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
#include <fstream>
#include "model_state.h"

namespace triton::backend::acl
{

    TRITONSERVER_Error* ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state)
    {
        try
        {
            *state = new ModelState(triton_model);
        }
        catch (const BackendModelException& ex)
        {
            RETURN_ERROR_IF_TRUE(ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL, std::string("unexpected nullptr in BackendModelException"));
            RETURN_IF_ERROR(ex.err_);
        }

        // Auto-complete the configuration if requested...
        bool auto_complete_config = false;
        RETURN_IF_ERROR(TRITONBACKEND_ModelAutoCompleteConfig(triton_model, &auto_complete_config));
        if (auto_complete_config)
        {
            RETURN_IF_ERROR((*state)->AutoCompleteConfig());
            triton::common::TritonJson::WriteBuffer json_buffer;
            (*state)->ModelConfig().Write(&json_buffer);

            TRITONSERVER_Message* message;
            RETURN_IF_ERROR(TRITONSERVER_MessageNewFromSerializedJson(&message, json_buffer.Base(), json_buffer.Size()));
            RETURN_IF_ERROR(TRITONBACKEND_ModelSetConfig(triton_model, 1 /* config_version */, message));
        }


        // Parse the output states in the model configuration
        auto& model_outputs = (*state)->model_outputs_;
        triton::common::TritonJson::Value sequence_batching;
        if ((*state)->ModelConfig().Find("sequence_batching", &sequence_batching))
        {
            triton::common::TritonJson::Value states;
            if (sequence_batching.Find("state", &states))
            {
                for (size_t i = 0; i < states.ArraySize(); i++)
                {
                    triton::common::TritonJson::Value state;
                    RETURN_IF_ERROR(states.IndexAsObject(i, &state));
                    std::string output_state_name;
                    RETURN_IF_ERROR(state.MemberAsString("output_name", &output_state_name));
                    auto it = model_outputs.find(output_state_name);
                    if (it == model_outputs.end())
                    {
                        model_outputs.insert({output_state_name, std::make_pair(-1, i)});
                    }
                    else
                    {
                        it->second.second = i;
                    }
                }
            }
        }

        // Parse the output names in the model configuration
        triton::common::TritonJson::Value outputs;
        RETURN_IF_ERROR((*state)->ModelConfig().MemberAsArray("output", &outputs));
        for (size_t i = 0; i < outputs.ArraySize(); i++)
        {
            triton::common::TritonJson::Value output;
            RETURN_IF_ERROR(outputs.IndexAsObject(i, &output));

            std::string output_name_str;
            RETURN_IF_ERROR(output.MemberAsString("name", &output_name_str));
            auto it = model_outputs.find(output_name_str);
            if (it == model_outputs.end())
            {
                model_outputs.insert({output_name_str, {i, -1}});
            }
            else
            {
                it->second.first = i;
            }
        }

        return nullptr;  // success
    }

    TRITONSERVER_Error* ModelState::ParseBoolParameter(triton::common::TritonJson::Value& params, const std::string& mkey, bool* value)
    {
        std::string value_str;
        RETURN_IF_ERROR(GetParameterValue(params, mkey, &value_str));
        RETURN_IF_ERROR(ParseBoolValue(value_str, value));

        return nullptr;
    }

    TRITONSERVER_Error* ModelState::ParseIntParameter(triton::common::TritonJson::Value& params, const std::string& mkey, int* value)
    {
        std::string value_str;
        RETURN_IF_ERROR(GetParameterValue(params, mkey, &value_str));
        RETURN_IF_ERROR(ParseIntValue(value_str, value));

        return nullptr;
    }

    TRITONSERVER_Error* ModelState::ParseDoubleParameter(triton::common::TritonJson::Value& params, const std::string& mkey, double* value)
    {
        std::string value_str;
        RETURN_IF_ERROR(GetParameterValue(params, mkey, &value_str));
        RETURN_IF_ERROR(ParseDoubleValue(value_str, value));

        return nullptr;
    }

    TRITONSERVER_Error* ModelState::ParseStrParameter(triton::common::TritonJson::Value& params, const std::string& mkey, std::string& value)
    {
        RETURN_IF_ERROR(GetParameterValue(params, mkey, &value));

        return nullptr;
    }

    TRITONSERVER_Error* ModelState::ParseParameters()
    {
        std::string model_name = Name();
        triton::common::TritonJson::Value params;
        bool status = ModelConfig().Find("parameters", &params);
        if (status)
        {
            // num_threads
            int num_threads = 1;
            auto err = ParseIntParameter(params, "num_threads", &num_threads);
            if (err != nullptr)
            {
                if (TRITONSERVER_ERROR_NOT_FOUND != TRITONSERVER_ErrorCode(err))
                    return err;
                else
                    TRITONSERVER_ErrorDelete(err);
            }
            acl_config_.num_threads = num_threads;
            LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("num_threads is ") + 
                std::to_string(num_threads) + " for model '" + Name() + "'").c_str());

            // enable_thread_affinity
            bool enable_thread_affinity = false;
            err = ParseBoolParameter(params, "enable_thread_affinity", &enable_thread_affinity);
            if (err != nullptr)
            {
                if (TRITONSERVER_ERROR_NOT_FOUND != TRITONSERVER_ErrorCode(err))
                    return err;
                else
                    TRITONSERVER_ErrorDelete(err);
            }
            acl_config_.enable_thread_affinity = enable_thread_affinity;
            LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("enable_thread_affinity is ") + 
                (enable_thread_affinity ? "enabled" : "disabled") + " for model '" + Name() + "'").c_str());

            // enable_parallel
            bool enable_parallel = false;
            err = ParseBoolParameter(params, "enable_parallel", &enable_parallel);
            if (err != nullptr)
            {
                if (TRITONSERVER_ERROR_NOT_FOUND != TRITONSERVER_ErrorCode(err))
                    return err;
                else
                    TRITONSERVER_ErrorDelete(err);
            }
            acl_config_.enable_parallel = enable_parallel;
            LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("enable_parallel is ") + 
                (enable_parallel ? "enabled" : "disabled") + " for model '" + Name() + "'").c_str());
            
            // enable_fp16
            bool enable_fp16 = false;
            err = ParseBoolParameter(params, "enable_fp16", &enable_fp16);
            if (err != nullptr)
            {
                if (TRITONSERVER_ERROR_NOT_FOUND != TRITONSERVER_ErrorCode(err))
                    return err;
                else
                    TRITONSERVER_ErrorDelete(err);
            }
            acl_config_.enable_fp16 = enable_fp16;
            LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("enable_fp16 is ") + 
                (enable_fp16 ? "enabled" : "disabled") + " for model '" + Name() + "'").c_str());

            // device_type
            std::string device_type = "ascend";
            err = ParseStrParameter(params, "device_type", device_type);
            if (err != nullptr)
            {
                if (TRITONSERVER_ERROR_NOT_FOUND != TRITONSERVER_ErrorCode(err))
                    return err;
                else
                    TRITONSERVER_ErrorDelete(err);
            }
            acl_config_.device_type = device_type;
            LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("device_type is ") + 
                device_type + " for model '" + Name() + "'").c_str());

            // model_type
            std::string model_type = "kmindir";
            err = ParseStrParameter(params, "model_type", model_type);
            if (err != nullptr)
            {
                if (TRITONSERVER_ERROR_NOT_FOUND != TRITONSERVER_ErrorCode(err))
                    return err;
                else
                    TRITONSERVER_ErrorDelete(err);
            }
            acl_config_.model_type = model_type;
            LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("model_type is ") + 
                model_type + " for model '" + Name() + "'").c_str());

            // device_id
            int device_id = -1;
            err = ParseIntParameter(params, "device_id", &device_id);
            if (err != nullptr)
            {
                if (TRITONSERVER_ERROR_NOT_FOUND != TRITONSERVER_ErrorCode(err))
                    return err;
                else
                    TRITONSERVER_ErrorDelete(err);
            }
            acl_config_.device_id = device_id;
            LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("device_id is ") + 
                std::to_string(device_id) + " for model '" + Name() + "'").c_str());

            // precision_mode
            std::string precision_mode = "preferred_optimal";
            err = ParseStrParameter(params, "precision_mode", precision_mode);
            if (err != nullptr)
            {
                if (TRITONSERVER_ERROR_NOT_FOUND != TRITONSERVER_ErrorCode(err))
                    return err;
                else
                    TRITONSERVER_ErrorDelete(err);
            }
            acl_config_.precision_mode = precision_mode;
            LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("precision_mode is ") + 
                precision_mode + " for model '" + Name() + "'").c_str());

            // frequency
            int frequency = 3;
            err = ParseIntParameter(params, "frequency", &frequency);
            if (err != nullptr)
            {
                if (TRITONSERVER_ERROR_NOT_FOUND != TRITONSERVER_ErrorCode(err))
                    return err;
                else
                    TRITONSERVER_ErrorDelete(err);
            }
            acl_config_.frequency = frequency;
            LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("frequency is ") + 
                std::to_string(frequency) + " for model '" + Name() + "'").c_str());

            // config_file
            std::string config_file = "";
            err = ParseStrParameter(params, "config_file", config_file);
            if (err != nullptr)
            {
                if (TRITONSERVER_ERROR_NOT_FOUND != TRITONSERVER_ErrorCode(err))
                    return err;
                else
                    TRITONSERVER_ErrorDelete(err);
            }
            acl_config_.config_file = config_file;
            LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("config_file is ") + 
                config_file + " for model '" + Name() + "'").c_str());
        }

        return nullptr;
    }

    ModelState::ModelState(TRITONBACKEND_Model* triton_model) : BackendModel(triton_model)
    {
        THROW_IF_BACKEND_MODEL_ERROR(ValidateModelConfig());
        THROW_IF_BACKEND_MODEL_ERROR(ParseParameters());
    }

    TRITONSERVER_Error* ModelState::AutoCompleteConfig()
    {
        // Auto-complete configuration if requests
        // If the model configuration already specifies inputs and outputs
        // then don't perform any auto-completion.
        size_t input_cnt = 0;
        size_t output_cnt = 0;
        {
            triton::common::TritonJson::Value inputs;
            if (ModelConfig().Find("input", &inputs))
            {
                input_cnt = inputs.ArraySize();
            }

            triton::common::TritonJson::Value config_batch_inputs;
            if (ModelConfig().Find("batch_input", &config_batch_inputs))
            {
                input_cnt += config_batch_inputs.ArraySize();
            }

            triton::common::TritonJson::Value outputs;
            if (ModelConfig().Find("output", &outputs))
            {
                output_cnt = outputs.ArraySize();
            }
        }

        if ((input_cnt > 0) && (output_cnt > 0))
        {
            LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("skipping model configuration auto-complete for '") +
                Name() + "': inputs and outputs already specified").c_str());
            if (TRITONSERVER_LogIsEnabled(TRITONSERVER_LOG_VERBOSE))
            {
                triton::common::TritonJson::WriteBuffer buffer;
                RETURN_IF_ERROR(ModelConfig().PrettyWrite(&buffer));
                LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("post auto-complete:\n") + buffer.Contents()).c_str());
            }
        }
        else
        {
            auto err = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL,
                (std::string("should explict set model's input/output in model config file")).c_str());
            return err;
        }

        return nullptr;  // success
    }

    TRITONSERVER_Error* ModelState::ValidateModelConfig()
    {
        common::TritonJson::Value inputs;
        common::TritonJson::Value outputs;
        RETURN_IF_ERROR(ModelConfig().MemberAsArray("input", &inputs));
        RETURN_IF_ERROR(ModelConfig().MemberAsArray("output", &outputs));

        for (size_t i = 0; i < inputs.ArraySize(); ++i)
        {
            common::TritonJson::Value input;
            RETURN_IF_ERROR(inputs.IndexAsObject(i, &input));

            triton::common::TritonJson::Value reshape;
            RETURN_ERROR_IF_TRUE(input.Find("reshape", &reshape), TRITONSERVER_ERROR_UNSUPPORTED,
                                std::string("reshape not supported for input tensor"));

            std::string name;
            RETURN_IF_ERROR(input.MemberAsString("name", &name));
            input_names_.push_back(name);

            std::string data_type;
            RETURN_IF_ERROR(input.MemberAsString("data_type", &data_type));
            input_data_types_.push_back(ModelConfigDataTypeToTritonServerDataType(data_type));

            std::string format;
            RETURN_IF_ERROR(input.MemberAsString("format", &format));
            input_formats_.push_back(format);

            std::vector<int64_t> dims;
            RETURN_IF_ERROR(ParseShape(input, "dims", &dims));
            if (MaxBatchSize())
                dims.insert(dims.begin(), -1);
            input_dims_[name] = dims;
        }

        for (size_t i = 0; i < outputs.ArraySize(); ++i)
        {
            common::TritonJson::Value output;
            RETURN_IF_ERROR(outputs.IndexAsObject(i, &output));

            triton::common::TritonJson::Value reshape;
            RETURN_ERROR_IF_TRUE(output.Find("reshape", &reshape), TRITONSERVER_ERROR_UNSUPPORTED,
                                std::string("reshape not supported for output tensor"));

            std::string name;
            RETURN_IF_ERROR(output.MemberAsString("name", &name));
            output_names_.push_back(name);

            std::string data_type;
            RETURN_IF_ERROR(output.MemberAsString("data_type", &data_type));
            output_data_types_.push_back(ModelConfigDataTypeToTritonServerDataType(data_type));

            std::vector<int64_t> dims;
            RETURN_IF_ERROR(ParseShape(output, "dims", &dims));
            if (MaxBatchSize())
                dims.insert(dims.begin(), -1);
            output_dims_[name] = dims;
        }

        return nullptr;
    }

} // namespace triton::backend::acl