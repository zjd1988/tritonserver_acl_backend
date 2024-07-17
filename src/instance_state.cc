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
#include <set>
#include "model_state.h"
#include "acl_utils.h"
#include "instance_state.h"
#include "acl_engine/engine_tensor_utils.h"

namespace triton::backend::acl
{

    TRITONSERVER_Error* ModelInstanceState::RunAclModel(std::vector<std::string>& input_names, 
        std::map<std::string, std::shared_ptr<AclTensor>>& input_tensors)
    {
        if (nullptr == acl_engine_)
        {
            auto err = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, (std::string("acl engine is nullptr")).c_str());
            return err;
        }

        std::map<std::string, AclTensor*> prepared_tensors;
        for (size_t index = 0; index < input_names.size(); index++)
        {
            std::string tensor_name = input_names[index];
            if (input_tensors.end() == input_tensors.find(tensor_name))
            {
                auto err = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, 
                    (std::string("cannot find tensor named: ") + tensor_name).c_str());
                return err;
            }
            prepared_tensors[tensor_name] = input_tensors[tensor_name].get();
        }

        // set acl engine input tensors
        if (0 != acl_engine_->setEngineInputTensors(prepared_tensors))
        {
            TRITONSERVER_Error* err = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, 
                    (std::string("acl engine set input tensors fail").c_str()));
            return err;
        }

        // acl engine run
        if (0 != acl_engine_->runEngine())
        {
            TRITONSERVER_Error* err = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, 
                (std::string("acl engine run fail").c_str()));
            return err;
        }

        return nullptr;
    }

    TRITONSERVER_Error* ModelInstanceState::GetAclModelOutputs(std::vector<std::string>& output_names, 
        std::map<std::string, AclTensor*>& output_tensors)
    {
        (void)output_names;
        if (0 != acl_engine_->getEngineOutputTensors(output_tensors))
        {
            TRITONSERVER_Error* err = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, 
                (std::string("acl engine get output tensors fail").c_str()));
            return err;
        }
        return nullptr;
    }

    TRITONSERVER_Error* ModelInstanceState::CreateTensor(const char* input_name, const std::vector<int64_t> shape, 
        TRITONSERVER_DataType triton_dtype, int batchn_byte_size, TRITONSERVER_MemoryType mem_type, int mem_type_id, 
        std::shared_ptr<AclTensor>& tensor, void* data, bool clone_flag)
    {
        try
        {
            std::unique_ptr<AclTensor> tensor_tmp;
            AclTensorDataType tensor_dtype = ConvertDataType(triton_dtype);
            std::vector<int64_t> tensor_shape(shape.begin(), shape.end());
            if (TRITONSERVER_MEMORY_GPU == mem_type)
            {
                tensor_tmp.reset(AclTensor::createDevice(tensor_shape, tensor_dtype, 
                    AclTensor::TENSOR_FORMAT_TYPE_NCHW, data, mem_type_id));
            }
            else
            {
                tensor_tmp.reset(AclTensor::create(tensor_shape, tensor_dtype, 
                    AclTensor::TENSOR_FORMAT_TYPE_NCHW, data));
            }
            if (clone_flag)
            {
                tensor_tmp.reset(AclTensor::clone(tensor_tmp.get(), true));
            }
            if (nullptr == tensor_tmp.get() ||
                (nullptr != tensor_tmp.get() && tensor_tmp->size() != batchn_byte_size))
            {
                auto err = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL,
                    (std::string("create tensor ") + input_name +  " fail").c_str());
                return err;
            }
            tensor.reset(tensor_tmp.release());
        }
        catch (const BackendModelInstanceException& ex)
        {
            RETURN_ERROR_IF_TRUE(ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
                std::string("unexpected nullptr in BackendModelInstanceException"));
            RETURN_IF_ERROR(ex.err_);
        }
        return nullptr;  
    }

    TRITONSERVER_Error* ModelInstanceState::CreateStringTensor(const char* input_name, const std::vector<int64_t> shape, 
        TRITONSERVER_DataType triton_dtype, TRITONSERVER_MemoryType mem_type, int mem_type_id, 
        std::shared_ptr<AclTensor>& tensor)
    {
        try
        {
            std::unique_ptr<AclTensor> tensor_tmp;
            AclTensorDataType tensor_dtype = ConvertDataType(triton_dtype);
            if (AclTensor::TENSOR_DATA_TYPE_STRING != tensor_dtype)
            {
                auto err = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, 
                    (std::string("create string tensor ") + input_name +  " failed with wrong dtype " 
                    + std::to_string(int(tensor_dtype))).c_str());
                return err;
            }
            std::vector<int64_t> tensor_shape(shape.begin(), shape.end());
            if (TRITONSERVER_MEMORY_CPU != mem_type && TRITONSERVER_MEMORY_CPU_PINNED != mem_type)
            {
                auto err = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, 
                    (std::string("create string tensor ") + input_name +  " failed with only support cpu memory type").c_str());
                return err;
            }
            tensor_tmp.reset(AclTensor::create(tensor_shape, tensor_dtype, AclTensor::TENSOR_FORMAT_TYPE_NCHW));

            if (nullptr == tensor_tmp.get())
            {
                auto err = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, 
                    (std::string("create string tensor ") + input_name +  " fail").c_str());
                return err;
            }
            tensor.reset(tensor_tmp.release());
        }
        catch (const BackendModelInstanceException& ex)
        {
            RETURN_ERROR_IF_TRUE(ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL, 
                std::string("unexpected nullptr in BackendModelInstanceException"));
            RETURN_IF_ERROR(ex.err_);
        }
        return nullptr;
    }

    TRITONSERVER_Error* ModelInstanceState::Create(ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
        ModelInstanceState** state)
    {
        try
        {
            *state = new ModelInstanceState(model_state, triton_model_instance);
        }
        catch (const BackendModelInstanceException& ex)
        {
            RETURN_ERROR_IF_TRUE(ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL, 
                std::string("unexpected nullptr in BackendModelInstanceException"));
            RETURN_IF_ERROR(ex.err_);
        }

        return nullptr;  // success
    }

    TRITONSERVER_Error* ModelInstanceState::DetermineModelPath(const std::string& model_dir, std::string* model_path, 
        std::string* config_path)
    {
        bool config_exists = true;
        std::string config_file_path = JoinPath({model_dir, "model.txt"});
        RETURN_IF_ERROR(FileExists(config_file_path, &config_exists));
        if (not config_exists)
        {
            LOG_MESSAGE(TRITONSERVER_LOG_WARN, ("cannot find model config " + config_file_path).c_str());
        }
        else
            *config_path = config_file_path;

        // check mindspore lite model file exist
        bool mindir_exists = false;
        bool ms_exists = false;
        std::string mindir_file_path = JoinPath({model_dir, "model.mindir"});
        std::string ms_file_path = JoinPath({model_dir, "model.ms"});
        RETURN_IF_ERROR(FileExists(mindir_file_path, &mindir_exists));
        RETURN_IF_ERROR(FileExists(ms_file_path, &ms_exists));
        if (not mindir_exists && not ms_exists)
        {
            return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_NOT_FOUND, 
                std::string("acl model should be named as 'model.mindir or model.ms'").c_str());
        }
        if (mindir_exists && ms_exists)
        {
            LOG_MESSAGE(TRITONSERVER_LOG_WARN, "model.mindir and model.ms both exists, model.mindir will be used");
        }
        *model_path = mindir_exists ? mindir_file_path : ms_file_path;

        return nullptr;
    }

    ModelInstanceState::ModelInstanceState(ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance)
        : BackendModelInstance(model_state, triton_model_instance), model_state_(model_state)
    {
        // get acl model and config file path
        std::string model_path, config_path;
        int device_id = DeviceId();
        auto model_dir = JoinPath({model_state->RepositoryPath(), std::to_string(model_state->Version())});
        THROW_IF_BACKEND_INSTANCE_ERROR(DetermineModelPath(model_dir, &model_path, &config_path));

        // init engine config
        EngineConfig engine_config = model_state->AclEngineConfig();
        // overwrite device id
        if (-1 == engine_config.device_id)
        {
            LOG_MESSAGE(TRITONSERVER_LOG_INFO, ("overwrite device id to " + std::to_string(device_id)).c_str());
            engine_config.device_id = device_id;
        }
        // overwrite config path
        if ("" == engine_config.config_file && "" != config_path)
        {
            LOG_MESSAGE(TRITONSERVER_LOG_INFO, ("overwrite acl config file to " + config_path).c_str());
            engine_config.config_file = config_path;
        }
        // overwrite model type by model file postfix
        // if model ends with ".mindir" then model_type set kmindir or else set kmindir_lite
        std::string file_postfix = ".mindir";
        if (model_path.rfind(file_postfix) == (model_path.length() - file_postfix.length()))
            engine_config.model_type = "kmindir";
        else
            engine_config.model_type = "kmindir_lite";

        // init acl engine with model files and config info
        std::vector<std::string> model_files = {model_path};
        acl_engine_.reset(new MindSporeLiteEngine(engine_config, model_files));
        if (nullptr == acl_engine_ || false == acl_engine_->status())
        {
            acl_engine_.reset();
            auto err = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, 
                (std::string("Failed to load model from ") + model_path).c_str());
            THROW_IF_BACKEND_INSTANCE_ERROR(err);
        }
        return;
    }

    void ModelInstanceState::FillStringData(std::vector<const char*>* string_ptrs, size_t cnt)
    {
        static const char* empty = "";
        for (size_t c = 0; c < cnt; c++)
        {
            string_ptrs->push_back(empty);
        }
        return;
    }

    void ModelInstanceState::SetStringInputBuffer(const std::string& input_name, const std::vector<size_t>& expected_byte_sizes,
        const std::vector<size_t>& expected_element_cnts, std::vector<TRITONBACKEND_Response*>* responses, char* input_buffer,
        std::vector<const char*>* string_ptrs)
    {
        // offset for each response
        size_t buffer_copy_offset = 0;
        for (size_t idx = 0; idx < expected_byte_sizes.size(); idx++)
        {
            const size_t expected_byte_size = expected_byte_sizes[idx];
            const size_t expected_element_cnt = expected_element_cnts[idx];

            size_t element_cnt = 0;
            if ((*responses)[idx] != nullptr)
            {
                size_t remaining_bytes = expected_byte_size;
                char* data_content = input_buffer + buffer_copy_offset;
                // Continue if the remaining bytes may still contain size info
                while (remaining_bytes >= sizeof(uint32_t))
                {
                    if (element_cnt >= expected_element_cnt)
                    {
                        RESPOND_AND_SET_NULL_IF_ERROR(&((*responses)[idx]), TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG,
                            (std::string("unexpected number of string elements ") + std::to_string(element_cnt + 1) + " for inference input '" +
                            input_name + "', expecting " + std::to_string(expected_element_cnt)).c_str()));
                        break;
                    }

                    const uint32_t len = *(reinterpret_cast<const uint32_t*>(data_content));
                    remaining_bytes -= sizeof(uint32_t);
                    // Make first byte of size info 0, so that if there is string data
                    // in front of it, the data becomes valid C string.
                    *data_content = 0;
                    data_content = data_content + sizeof(uint32_t);
                    if (len > remaining_bytes)
                    {
                        RESPOND_AND_SET_NULL_IF_ERROR(&((*responses)[idx]), TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG,
                            (std::string("incomplete string data for inference input '") + input_name + "', expecting string of length " +
                            std::to_string(len) + " but only " + std::to_string(remaining_bytes) + " bytes available").c_str()));
                        break;
                    }
                    else
                    {
                        string_ptrs->push_back(data_content);
                        element_cnt++;
                        data_content = data_content + len;
                        remaining_bytes -= len;
                    }
                }
            }

            FillStringData(string_ptrs, expected_element_cnt - element_cnt);
            buffer_copy_offset += expected_byte_size;
        }
        return;
    }

    TRITONSERVER_Error* ModelInstanceState::SetStringInputTensor(TRITONBACKEND_Request** requests, const uint32_t request_count,
        std::vector<TRITONBACKEND_Response*>* responses, const char* input_name, std::shared_ptr<BackendMemory>& backend_memory, 
        std::vector<const char*>* string_ptrs, bool* cuda_copy)
    {
        size_t total_byte_size = 0;
        std::vector<size_t> expected_byte_sizes;
        std::vector<size_t> expected_element_cnts;
        expected_byte_sizes.reserve(request_count);
        expected_element_cnts.reserve(request_count);
        for (size_t ridx = 0; ridx < request_count; ++ridx)
        {
            TRITONBACKEND_Input* in;
            RESPOND_AND_SET_NULL_IF_ERROR(&((*responses)[ridx]), TRITONBACKEND_RequestInput(requests[ridx], input_name, &in));

            const int64_t* input_shape;
            uint32_t input_dims_count;
            uint64_t input_byte_size;
            RETURN_IF_ERROR(TRITONBACKEND_InputProperties(in, nullptr, nullptr, &input_shape, &input_dims_count, &input_byte_size, nullptr));

            // Skip input in this request if error response has already been sent.
            if ((*responses)[ridx] == nullptr)
            {
                expected_byte_sizes.push_back(0);
                expected_element_cnts.push_back(0);
            }
            else
            {
                expected_element_cnts.push_back(GetElementCount(input_shape, input_dims_count));
                expected_byte_sizes.push_back(input_byte_size);
            }
            total_byte_size += expected_byte_sizes.back();
        }

        // For string input, the copy to contiguous buffer is needed because acl
        // expects elements to be C strings thus we need to modify input buffer.
        // Reserve one more byte at the end of input_buffer to ensure last
        // element of String data can become valid C string.
        BackendMemory* input_memory;
        RETURN_IF_ERROR(BackendMemory::Create(model_state_->TritonMemoryManager(), 
            {BackendMemory::AllocationType::CPU_PINNED_POOL, BackendMemory::AllocationType::CPU}, 
            0 /* memory_type_id */, total_byte_size + 1, &input_memory));
        backend_memory.reset(input_memory);

        const TRITONSERVER_MemoryType mem_type = input_memory->MemoryType();
        char* input_buffer = input_memory->MemoryPtr();
        size_t buffer_offset = 0;
        for (size_t ridx = 0; ridx < request_count; ++ridx)
        {
            TRITONBACKEND_Input* in;
            TRITONSERVER_Error* err = TRITONBACKEND_RequestInput(requests[ridx], input_name, &in);
            if ((err == nullptr) && ((*responses)[ridx] != nullptr))
            {
                uint32_t input_buffer_count;
                RETURN_IF_ERROR(TRITONBACKEND_InputPropertiesForHostPolicy(
                    in, HostPolicyName().c_str(), nullptr, nullptr, nullptr, nullptr, nullptr, &input_buffer_count));

                size_t input_offset = 0;
                for (size_t idx = 0; idx < input_buffer_count; ++idx)
                {
                    const void* src_buffer;
                    size_t src_byte_size;
                    TRITONSERVER_MemoryType src_memory_type;
                    int64_t src_memory_type_id;
                    err = TRITONBACKEND_InputBufferForHostPolicy(in, HostPolicyName().c_str(), idx, &src_buffer, 
                        &src_byte_size, &src_memory_type, &src_memory_type_id);
                    if (err == nullptr)
                    {
                        if ((input_offset + src_byte_size) > expected_byte_sizes[ridx])
                        {
                            err = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, (std::string("buffer size for input '") + 
                                input_name + "' exceeds batch byte size " + std::to_string(expected_byte_sizes[ridx])).c_str());
                        }
                        else
                        {
                            bool cuda_used = false;
                            err = CopyBuffer(input_name, src_memory_type, src_memory_type_id, mem_type, 0, src_byte_size, 
                                src_buffer, input_buffer + buffer_offset + input_offset, CudaStream(), &cuda_used);
                            *cuda_copy |= cuda_used;
                        }
                    }

                    if (err == nullptr)
                    {
                        input_offset += src_byte_size;
                    }
                    else
                    {
                        break;
                    }
                }
            }

            if (err != nullptr)
            {
                if ((*responses)[ridx] != nullptr)
                {
                    RESPOND_AND_SET_NULL_IF_ERROR(&((*responses)[ridx]), err);
                }
                TRITONSERVER_ErrorDelete(err);
            }

            buffer_offset += expected_byte_sizes[ridx];
        }

        #ifdef TRITON_ENABLE_GPU
        // Synchronize to ensure the buffer is ready to be modified
        if (*cuda_copy)
        {
            cudaStreamSynchronize(CudaStream());
            *cuda_copy = false;
        }
        #endif  // TRITON_ENABLE_GPU

        // Modify input buffer and set string expected by acl
        SetStringInputBuffer(input_name, expected_byte_sizes, expected_element_cnts, responses, input_buffer, string_ptrs);
        input_buffer[total_byte_size] = 0;
        return nullptr;
    }

    TRITONSERVER_Error* ModelInstanceState::SetInputTensors(size_t total_batch_size, TRITONBACKEND_Request** requests,
        const uint32_t request_count, std::vector<TRITONBACKEND_Response*>* responses, 
        BackendInputCollector* collector, std::vector<std::string>& input_names, 
        std::map<std::string, std::shared_ptr<AclTensor>>& input_tensors, 
        std::vector<std::shared_ptr<BackendMemory>>& backend_memorys, bool* cuda_copy)
    {
        const int max_batch_size = model_state_->MaxBatchSize();

        // All requests must have equally-sized input tensors so use any
        // request as the representative for the input tensors.
        uint32_t input_count;
        RETURN_IF_ERROR(TRITONBACKEND_RequestInputCount(requests[0], &input_count));

        for (uint32_t input_idx = 0; input_idx < input_count; input_idx++)
        {
            TRITONBACKEND_Input* input;
            RETURN_IF_ERROR(TRITONBACKEND_RequestInputByIndex(requests[0], input_idx, &input));

            const char* input_name;
            TRITONSERVER_DataType input_datatype;
            const int64_t* input_shape;
            uint32_t input_dims_count;
            RETURN_IF_ERROR(TRITONBACKEND_InputProperties(input, &input_name, &input_datatype, 
                &input_shape, &input_dims_count, nullptr, nullptr));

            input_names.emplace_back(input_name);
            std::shared_ptr<AclTensor> input_tensor;
            std::vector<int64_t> batchn_shape;
            // For a ragged input tensor, the tensor shape should be
            // the flatten shape of the whole batch
            if (StateForModel()->IsInputRagged(input_name))
            {
                batchn_shape = std::vector<int64_t>{0};
                for (size_t idx = 0; idx < request_count; idx++)
                {
                    TRITONBACKEND_Input* input;
                    RESPOND_AND_SET_NULL_IF_ERROR(&((*responses)[idx]), 
                        TRITONBACKEND_RequestInput(requests[idx], input_name, &input));
                    const int64_t* input_shape;
                    uint32_t input_dims_count;
                    RESPOND_AND_SET_NULL_IF_ERROR(&((*responses)[idx]), 
                        TRITONBACKEND_InputProperties(input, nullptr, nullptr, &input_shape, 
                        &input_dims_count, nullptr, nullptr));

                    batchn_shape[0] += GetElementCount(input_shape, input_dims_count);
                }
            }
            // The shape for the entire input batch, [total_batch_size, ...]
            else 
            {
                batchn_shape = std::vector<int64_t>(input_shape, input_shape + input_dims_count);
                if (max_batch_size != 0)
                {
                    batchn_shape[0] = total_batch_size;
                }
            }

            if (input_datatype != TRITONSERVER_TYPE_BYTES)
            {
                // The input must be in contiguous CPU memory. Use appropriate
                // allocator info to bind inputs to the right device. .i.e bind inputs
                // to GPU if they are being provided on GPU.
                const char* input_buffer;
                size_t batchn_byte_size;
                TRITONSERVER_MemoryType memory_type;
                int64_t memory_type_id;
                std::vector<std::pair<TRITONSERVER_MemoryType, int64_t>> allowed_input_types;
                if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU)
                {
                    allowed_input_types = {{TRITONSERVER_MEMORY_GPU, DeviceId()},
                                           {TRITONSERVER_MEMORY_CPU_PINNED, 0},
                                           {TRITONSERVER_MEMORY_CPU, 0}};
                }
                else
                {
                    allowed_input_types = {{TRITONSERVER_MEMORY_CPU_PINNED, 0}, 
                                           {TRITONSERVER_MEMORY_CPU, 0}};
                }

                RETURN_IF_ERROR(collector->ProcessTensor(input_name, nullptr, 0, allowed_input_types, &input_buffer,
                    &batchn_byte_size, &memory_type, &memory_type_id));

                // Create acl Tensor
                RETURN_IF_ERROR(CreateTensor(input_name, batchn_shape, input_datatype, batchn_byte_size, 
                    memory_type, memory_type_id, input_tensor, (void*)input_buffer));
            }
            else
            {
                // For BYTES input, we need to convert the serialized string
                // representation into what is required for ORT. ORT expects a
                // vector of char*, one for each element. For each tensor we get
                // a copy of the data in a contiguous CPU buffer and then
                // in-place modify that from the Triton
                // <int32_len><bytes><int32_len><bytes>... serialization into a
                // <bytes><null-terminator><bytes><null-terminator>... serialization
                // and then initialize 'string_ptrs' to point to each <bytes>.
                std::vector<const char*> string_ptrs;
                std::shared_ptr<BackendMemory> backend_memory;
                RETURN_IF_ERROR(SetStringInputTensor(requests, request_count, responses, input_name, 
                    backend_memory, &string_ptrs, cuda_copy));
                backend_memorys.push_back(backend_memory);

                TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
                int64_t memory_type_id = -1;
                RETURN_IF_ERROR(CreateStringTensor(input_name, batchn_shape, input_datatype, memory_type, 
                    memory_type_id, input_tensor));

                RETURN_ERROR_IF_TRUE(0 != TensorUtils::setStringTensorContent(input_tensor.get(), string_ptrs.data(), string_ptrs.size()), 
                    TRITONSERVER_ERROR_INTERNAL, std::string("set string tensor ") + input_name + std::string(" content fail"));
            }
            input_tensors[input_name] = input_tensor;
        }

        // Process batch input if any
        for (const auto& batch_input : StateForModel()->BatchInputs())
        {
            std::vector<int64_t> shape;
            collector->BatchInputShape(batch_input, &shape);
            for (const auto& input_name : batch_input.TargetNames())
            {
                input_names.emplace_back(input_name);

                const char* dst_buffer;
                size_t dst_buffer_byte_size;
                TRITONSERVER_MemoryType dst_memory_type;
                int64_t dst_memory_type_id;

                // Batch inputs are always created on CPU
                RESPOND_ALL_AND_SET_NULL_IF_ERROR((*responses), responses->size(),
                    collector->ProcessBatchInput(batch_input, nullptr, 0, {{TRITONSERVER_MEMORY_CPU, 0}},
                    &dst_buffer, &dst_buffer_byte_size, &dst_memory_type, &dst_memory_type_id));

                // Create acl Tensor
                std::shared_ptr<AclTensor> input_tensor;
                RETURN_IF_ERROR(CreateTensor(input_name.c_str(), shape, batch_input.DataType(), 
                    dst_buffer_byte_size, dst_memory_type, dst_memory_type_id, input_tensor, (void*)dst_buffer));
                input_tensors[input_name] = input_tensor;

            }
        }

        // Finalize...
        *cuda_copy |= collector->Finalize();
        return nullptr;
    }

    TRITONSERVER_Error* ModelInstanceState::ReadOutputTensor(const std::string& name, std::vector<int64_t>& batchn_shape, 
        TRITONSERVER_DataType& dtype, AclTensor* output_tensor, void** output_buffer, 
        std::vector<std::vector<char>>& string_buffers, std::vector<size_t>& offsets)
    {
        // Get output type and shape
        auto type = output_tensor->getTensorDataType();
        auto shape = output_tensor->shape();
        dtype = ConvertDataType(type);
        if (TRITONSERVER_TYPE_INVALID == dtype)
        {
            TRITONSERVER_Error* err = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, (std::string("output tensor:") + 
                name + " data type:" + std::to_string(int(type)) + " not supported").c_str());
            return err;
        }
        batchn_shape.insert(batchn_shape.end(), shape.begin(), shape.end());
        if (type == AclTensor::TENSOR_DATA_TYPE_STRING)
        {
            const size_t element_count = GetElementCount(batchn_shape);
            size_t total_length = 0;
            RETURN_ERROR_IF_TRUE(0 != TensorUtils::getStringTensorByteSize(output_tensor, total_length), 
                TRITONSERVER_ERROR_INTERNAL, std::string("get string tensor size fail"));
            string_buffers.emplace_back(std::vector<char>(total_length));
            auto content = string_buffers.back().data();
            offsets.reserve(element_count + 1);
            RETURN_ERROR_IF_TRUE(0 != TensorUtils::getStringTensorContent(output_tensor, content, total_length, 
                offsets.data(), element_count), TRITONSERVER_ERROR_INTERNAL, 
                std::string("get string tensor content fail"));
            // Mark "passed end byte offset"
            offsets[element_count] = total_length;
        }
        else
        {
            // Fixed size data type...
            void* device_ptr = (void*)output_tensor->devicePtr();
            void* host_ptr = output_tensor->host<void>();
            *output_buffer = (nullptr != device_ptr) ? device_ptr : host_ptr;
        }
        return nullptr;  // success
    }

    bool ModelInstanceState::SetStringBuffer(const std::string& name, const char* content, const size_t* offsets,
        std::vector<int64_t>* batchn_shape, TRITONBACKEND_Request** requests, const uint32_t request_count,
        std::vector<TRITONBACKEND_Response*>* responses, bool state)
    {
        size_t element_idx = 0;
        bool cuda_copy = false;
        for (size_t ridx = 0; ridx < request_count; ++ridx)
        {
            const auto& request = requests[ridx];
            auto& response = (*responses)[ridx];

            // batchn_shape holds the shape of the entire tensor batch. When
            // batching is enabled override the first batch dimension with each
            // requests batch size (reusing for efficiency).
            if (model_state_->MaxBatchSize() > 0)
            {
                TRITONBACKEND_Input* input;
                TRITONBACKEND_RequestInputByIndex(request, 0 /* index */, &input);
                const int64_t* shape;
                TRITONBACKEND_InputProperties(input, nullptr, nullptr, &shape, nullptr, nullptr, nullptr);
                (*batchn_shape)[0] = shape[0];
            }

            const size_t expected_element_cnt = GetElementCount(*batchn_shape);

            // If 'request' requested this output then copy it from
            // 'content'. If it did not request this output then just skip it
            // in the 'content'.
            bool need_output = false;
            if (!state)
            {
                if (response != nullptr)
                {
                    uint32_t output_count;
                    RESPOND_AND_SET_NULL_IF_ERROR(&response, TRITONBACKEND_RequestOutputCount(request, &output_count));
                    if (response != nullptr)
                    {
                        for (uint32_t output_idx = 0; output_idx < output_count; output_idx++)
                        {
                            const char* req_output_name;
                            RESPOND_AND_SET_NULL_IF_ERROR(&response, TRITONBACKEND_RequestOutputName(
                                request, output_idx, &req_output_name));
                            if ((response != nullptr) && (req_output_name == name))
                            {
                                need_output = true;
                                break;
                            }
                        }
                    }
                }
            }
            else
            {
                // need_output must be always set to true for state tensors.
                need_output = true;
            }

            if (need_output)
            {
                TRITONSERVER_Error* err;
                TRITONBACKEND_Output* response_output;
                TRITONBACKEND_State* response_state;
                if (!state)
                {
                    err = TRITONBACKEND_ResponseOutput(response, &response_output, name.c_str(), TRITONSERVER_TYPE_BYTES,
                        batchn_shape->data(), batchn_shape->size());
                }
                else
                {
                    err = TRITONBACKEND_StateNew(&response_state, request, name.c_str(), TRITONSERVER_TYPE_BYTES,
                        batchn_shape->data(), batchn_shape->size());
                }
                if (err == nullptr)
                {
                    // Calculate expected byte size in advance using string offsets
                    const size_t data_byte_size = offsets[element_idx + expected_element_cnt] - offsets[element_idx];
                    const size_t expected_byte_size = data_byte_size + sizeof(uint32_t) * expected_element_cnt;

                    TRITONSERVER_MemoryType actual_memory_type = TRITONSERVER_MEMORY_CPU_PINNED;
                    int64_t actual_memory_type_id = 0;
                    void* buffer;
                    if (!state)
                    {
                        err = TRITONBACKEND_OutputBuffer(response_output, &buffer, expected_byte_size, &actual_memory_type,
                            &actual_memory_type_id);
                    }
                    else
                    {
                        err = TRITONBACKEND_StateBuffer(response_state, &buffer, expected_byte_size, &actual_memory_type,
                            &actual_memory_type_id);
                    }
                    if (err == nullptr)
                    {
                        bool cuda_used = false;
                        size_t copied_byte_size = 0;
                        for (size_t e = 0; e < expected_element_cnt; ++e)
                        {
                            const uint32_t len = offsets[element_idx + e + 1] - offsets[element_idx + e];
                            // Prepend size of the string
                            err = CopyBuffer(name, TRITONSERVER_MEMORY_CPU /* src_memory_type */,
                                0 /* src_memory_type_id */, actual_memory_type, actual_memory_type_id, sizeof(uint32_t),
                                static_cast<const void*>(&len), static_cast<char*>(buffer) + copied_byte_size, stream_, &cuda_used);
                            if (err != nullptr)
                            {
                                break;
                            }

                            cuda_copy |= cuda_used;
                            copied_byte_size += sizeof(uint32_t);

                            // Copy raw string content
                            err = CopyBuffer(name, TRITONSERVER_MEMORY_CPU /* src_memory_type */,
                                0 /* src_memory_type_id */, actual_memory_type,
                                actual_memory_type_id, len, content + offsets[element_idx + e],
                                static_cast<char*>(buffer) + copied_byte_size, stream_, &cuda_used);
                            if (err != nullptr)
                            {
                                break;
                            }

                            cuda_copy |= cuda_used;
                            copied_byte_size += len;
                        }
                    }
                }

                RESPOND_AND_SET_NULL_IF_ERROR(&response, err);
                if (state)
                {
                    RESPOND_AND_SET_NULL_IF_ERROR(&response, TRITONBACKEND_StateUpdate(response_state));
                }
            }

            element_idx += expected_element_cnt;
        }

        return cuda_copy;
    }

    bool ModelInstanceState::SetStringStateBuffer(const std::string& name, const char* content, const size_t* offsets,
        std::vector<int64_t>* batchn_shape, TRITONBACKEND_Request** requests, const uint32_t request_count, 
        std::vector<TRITONBACKEND_Response*>* responses)
    {
        return SetStringBuffer(name, content, offsets, batchn_shape, requests, request_count, responses, true /* state */);
    }

    bool ModelInstanceState::SetStringOutputBuffer(const std::string& name, const char* content, const size_t* offsets,
        std::vector<int64_t>* batchn_shape, TRITONBACKEND_Request** requests, const uint32_t request_count,
        std::vector<TRITONBACKEND_Response*>* responses)
    {
        return SetStringBuffer(name, content, offsets, batchn_shape, requests, request_count, responses, false /* state */);
    }

    TRITONSERVER_Error* ModelInstanceState::ReadOutputTensors(size_t total_batch_size, TRITONBACKEND_Request** requests,
        const uint32_t request_count, std::vector<TRITONBACKEND_Response*>* responses)
    {
        BackendOutputResponder responder(requests, request_count, responses, model_state_->TritonMemoryManager(),
            model_state_->MaxBatchSize() > 0, model_state_->EnablePinnedInput(), CudaStream());

        // Use to hold string output contents
        bool cuda_copy = false;
        auto& model_outputs = StateForModel()->ModelOutputs();
        auto model_outout_names = StateForModel()->OutputNames();

        std::map<std::string, AclTensor*> output_tensors;
        RETURN_IF_ERROR(GetAclModelOutputs(model_outout_names, output_tensors));
        if (output_tensors.size() != model_outputs.size())
        {
            RETURN_IF_ERROR(TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL,
                ("Retrieved output count is not equal to expected count.")));
        }

        std::vector<std::vector<char>> string_buffers;
        auto model_outputs_it = model_outputs.begin();
        for (size_t idx = 0; idx < model_outputs.size(); idx++, model_outputs_it++)
        {
            AclTensor* output_tensor = nullptr;
            const std::string& name = model_outputs_it->first;
            auto& output_tensor_pair = model_outputs_it->second;
            if (output_tensors.end() == output_tensors.find(name))
            {
                RETURN_IF_ERROR(TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL,
                    (std::string("output tensor '") + name + "' is not found").c_str()));
            }
            output_tensor = output_tensors[name];
            void* device_ptr = (void*)output_tensor->devicePtr();
            void* host_ptr = output_tensor->host<void>();
            if (nullptr == host_ptr && nullptr == device_ptr)
            {
                RETURN_IF_ERROR(TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL,
                    (std::string("output tensor '") + name + "' host/device data both is nullptr").c_str()));
            }

            const auto memory_type = (nullptr == device_ptr) ? TRITONSERVER_MEMORY_CPU : TRITONSERVER_MEMORY_GPU;
            const auto memory_id = (nullptr == device_ptr) ? 0 : output_tensor->deviceId();
            const BatchOutput* batch_output = StateForModel()->FindBatchOutput(name);
            if (batch_output == nullptr)
            {
                std::vector<int64_t> batchn_shape;
                TRITONSERVER_DataType dtype;
                void* output_buffer;
                std::vector<std::vector<char>> string_buffers;
                std::vector<size_t> offsets;
                RETURN_IF_ERROR(ReadOutputTensor(name, batchn_shape, dtype, output_tensor, &output_buffer, string_buffers, offsets));

                if (output_tensor_pair.first != -1)
                {
                    if (dtype == TRITONSERVER_TYPE_BYTES)
                    {
                        auto content = string_buffers.back().data();
                        cuda_copy |= SetStringOutputBuffer(name, content, offsets.data(), &batchn_shape, requests,
                            request_count, responses);
                    }
                    else
                    {
                        responder.ProcessTensor(name, dtype, batchn_shape, reinterpret_cast<char*>(output_buffer),
                            memory_type, memory_id);
                    }
                }

                if (output_tensor_pair.second != -1)
                {
                    std::vector<TRITONBACKEND_State*> states;
                    if (dtype == TRITONSERVER_TYPE_BYTES)
                    {
                        auto content = string_buffers.back().data();
                        cuda_copy |= SetStringStateBuffer(name, content, offsets.data(), &batchn_shape, requests,
                            request_count, responses);
                    }
                    else
                    {
                        states = responder.ProcessStateTensor(name, dtype, batchn_shape, reinterpret_cast<char*>(output_buffer),
                            memory_type, memory_id);
                    }

                    // Update the states
                    for (auto& state : states)
                    {
                        RETURN_IF_ERROR(TRITONBACKEND_StateUpdate(state));
                    }
                }
            }
            else
            {
                char* output_buffer = (memory_type == TRITONSERVER_MEMORY_CPU) ? (char*)host_ptr : (char*)device_ptr;
                responder.ProcessBatchOutput(name, *batch_output, output_buffer, memory_type, memory_id);
            }
        }

        // Finalize and wait for any pending buffer copies.
        cuda_copy |= responder.Finalize();

        #ifdef TRITON_ENABLE_GPU
        if (cuda_copy)
        {
            cudaStreamSynchronize(stream_);
        }
        #endif  // TRITON_ENABLE_GPU
        return nullptr;
    }

    void ModelInstanceState::ProcessRequests(TRITONBACKEND_Request** requests, const uint32_t request_count)
    {
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("TRITONBACKEND_ModelExecute: Running ") + 
            Name() + " with " + std::to_string(request_count) + " requests begin").c_str());

        uint64_t exec_start_ns = 0;
        SET_TIMESTAMP(exec_start_ns);

        const int max_batch_size = model_state_->MaxBatchSize();

        // For each request collect the total batch size for this inference
        // execution. The batch-size, number of inputs, and size of each
        // input has already been checked so don't need to do that here.
        size_t total_batch_size = 0;
        for (size_t i = 0; i < request_count; i++)
        {
            // If we get a nullptr request then something is badly wrong. Fail
            // and release all requests.
            if (requests[i] == nullptr)
            {
                RequestsRespondWithError(requests, request_count, TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL,
                        std::string("null request given to acl Runtime backend for '" + Name() + "'").c_str()));
                return;
            }

            if (max_batch_size > 0)
            {
                // Retrieve the batch size from one of the inputs, if the model
                // supports batching, the first dimension size is batch size
                TRITONBACKEND_Input* input;
                TRITONSERVER_Error* err = TRITONBACKEND_RequestInputByIndex(requests[i], 0 /* index */, &input);
                if (err == nullptr)
                {
                    const int64_t* shape;
                    err = TRITONBACKEND_InputProperties(input, nullptr, nullptr, &shape, nullptr, nullptr, nullptr);
                    total_batch_size += shape[0];
                }
                if (err != nullptr)
                {
                    RequestsRespondWithError(requests, request_count, err);
                    return;
                }
            }
            else
            {
                total_batch_size += 1;
            }
        }

        // If there are no valid payloads then no need to run the inference.
        if (total_batch_size == 0)
        {
            return;
        }

        // Make sure the maximum batch size is not exceeded. The
        // total_batch_size must be 1 for models that don't support batching
        // (i.e. max_batch_size == 0). If max_batch_size is exceeded then
        // scheduler has done something badly wrong so fail and release all
        // requests.
        if ((total_batch_size != 1) && (total_batch_size > (size_t)max_batch_size))
        {
            RequestsRespondWithError(requests, request_count, TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL,
                std::string("batch size " + std::to_string(total_batch_size) + " for '" + 
                Name() + "', max allowed is " + std::to_string(max_batch_size)).c_str()));
            return;
        }

        // At this point we are committed to running inference with all
        // 'requests'. Create a response for each request. During input
        // processing if there is an error with any request that error will
        // be sent immediately with the corresponding response (and the
        // response unique_ptr will then be nullptr). The request object
        // itself will not be released until after all inferencing is done
        // (below) as we may need to access the request object when
        // determine how to process outputs (for example, even if we don't
        // need the outputs for a request that has an error, we do need to
        // know the size of those outputs associated with the request so we
        // can skip them in the output tensors).
        std::vector<TRITONBACKEND_Response*> responses;
        responses.reserve(request_count);
        bool all_response_failed = false;

        for (size_t i = 0; i < request_count; i++)
        {
            TRITONBACKEND_Response* response;
            auto err = TRITONBACKEND_ResponseNew(&response, requests[i]);
            if (err == nullptr)
            {
                responses.emplace_back(response);
            }
            else
            {
                responses.emplace_back(nullptr);
                LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "Fail to create response");
                TRITONSERVER_ErrorDelete(err);
            }
        }

        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("TRITONBACKEND_ModelExecute: Running ") + 
            Name() + " with " + std::to_string(request_count) + " requests SetInputTensors").c_str());

        std::vector<std::shared_ptr<BackendMemory>> backend_memorys;
        std::map<std::string, std::shared_ptr<AclTensor>> input_tensors;
        std::vector<std::string> input_names;
        bool cuda_copy = false;
        BackendInputCollector collector(requests, request_count, &responses, model_state_->TritonMemoryManager(), 
            model_state_->EnablePinnedInput(), CudaStream(), nullptr, nullptr, 0, HostPolicyName().c_str());
        RESPOND_ALL_AND_SET_TRUE_IF_ERROR(responses, request_count, all_response_failed, 
            SetInputTensors(total_batch_size, requests, request_count, &responses, &collector, 
            input_names, input_tensors, backend_memorys, &cuda_copy));

        if (input_names.size() != input_tensors.size())
        {
            RequestsRespondWithError(requests, request_count, TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, 
                std::string(Name() + " SetInputTensors get input names number is " + std::to_string(input_names.size()) + 
                ", but input tensors number is " + std::to_string(input_tensors.size())).c_str()));
            return;
        }

        // Wait for any in-flight input tensor copies to complete.
        #ifdef TRITON_ENABLE_GPU
        if (cuda_copy)
        {
            cudaStreamSynchronize(CudaStream());
        }
        #endif

        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("TRITONBACKEND_ModelExecute: Running ") + 
            Name() + " with " + std::to_string(request_count) + " requests RunAclModel").c_str());

        uint64_t compute_start_ns = 0;
        SET_TIMESTAMP(compute_start_ns);

        if (!all_response_failed)
        {
            RESPOND_ALL_AND_SET_TRUE_IF_ERROR(responses, request_count, all_response_failed, 
                RunAclModel(input_names, input_tensors));
        }

        uint64_t compute_end_ns = 0;
        SET_TIMESTAMP(compute_end_ns);

        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("TRITONBACKEND_ModelExecute: Running ") + 
            Name() + " with " + std::to_string(request_count) + " requests ReadOutputTensors").c_str());

        if (!all_response_failed)
        {
            RESPOND_ALL_AND_SET_TRUE_IF_ERROR(responses, request_count, all_response_failed, 
                ReadOutputTensors(total_batch_size, requests, request_count, &responses));
        }

        uint64_t exec_end_ns = 0;
        SET_TIMESTAMP(exec_end_ns);

        // Send all the responses that haven't already been sent because of
        // an earlier error. Note that the responses are not set to nullptr
        // here as we need that indication below to determine if the request
        // we successful or not.
        for (auto& response : responses)
        {
            if (response != nullptr)
            {
                LOG_IF_ERROR(TRITONBACKEND_ResponseSend(response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
                    "failed to send acl backend response");
            }
        }

        // Report statistics for each request.
        for (uint32_t r = 0; r < request_count; ++r)
        {
            auto& request = requests[r];
            LOG_IF_ERROR(TRITONBACKEND_ModelInstanceReportStatistics(TritonModelInstance(), request, 
                (responses[r] != nullptr) /* success */, exec_start_ns, compute_start_ns, compute_end_ns, 
                exec_end_ns), "failed reporting request statistics");

            LOG_IF_ERROR(TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
                "failed releasing request");
        }

        if (!all_response_failed)
        {
            // Report the entire batch statistics.
            LOG_IF_ERROR(TRITONBACKEND_ModelInstanceReportBatchStatistics(TritonModelInstance(), total_batch_size, 
                exec_start_ns, compute_start_ns, compute_end_ns, exec_end_ns), 
                "failed reporting batch request statistics");
        }

        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("TRITONBACKEND_ModelExecute: Running ") + 
            Name() + " with " + std::to_string(request_count) + " requests end").c_str());
        return;
    }

} // namespace triton::backend::acl