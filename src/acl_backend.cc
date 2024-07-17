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
#include "triton/backend/backend_common.h"
#include "triton/core/tritonbackend.h"
#include "instance_state.h"
#include "model_state.h"
#include "acl_utils.h"
#include "acl_engine/log.h"

namespace triton::backend::acl
{
    extern "C" 
    {

        TRITONSERVER_Error* TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
        {
            const char* cname;
            RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
            std::string name(cname);

            LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("TRITONBACKEND_Initialize: ") + name).c_str());
            // Check the backend API version that Triton supports vs. what this
            // backend was compiled against. Make sure that the Triton major
            // version is the same and the minor version is >= what this backend
            // uses.
            uint32_t api_version_major, api_version_minor;
            RETURN_IF_ERROR(TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));

            LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("Triton TRITONBACKEND API version: ") +
                std::to_string(api_version_major) + "." + std::to_string(api_version_minor)).c_str());

            LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("'") + name + "' TRITONBACKEND API version: " +
                std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." + std::to_string(TRITONBACKEND_API_VERSION_MINOR)).c_str());

            if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
                (api_version_minor < TRITONBACKEND_API_VERSION_MINOR))
            {
                return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_UNSUPPORTED,
                    (std::string("Triton TRITONBACKEND API version: ") + std::to_string(api_version_major) + "." +
                    std::to_string(api_version_minor) + " does not support '" + name + "' TRITONBACKEND API version: " +
                    std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." + std::to_string(TRITONBACKEND_API_VERSION_MINOR)).c_str());
            }

            // The backend configuration may contain information needed by the
            // backend, such a command-line arguments.
            TRITONSERVER_Message* backend_config_message;
            RETURN_IF_ERROR(TRITONBACKEND_BackendConfig(backend, &backend_config_message));

            const char* buffer;
            size_t byte_size;
            RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(backend_config_message, &buffer, &byte_size));
            LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("backend configuration:\n") + buffer).c_str());

            triton::common::TritonJson::Value backend_config;
            if (byte_size != 0)
            {
                RETURN_IF_ERROR(backend_config.Parse(buffer, byte_size));
            }

            // add cmdline parse for acl backend
            std::string backend_log_file = "./triton-acl.log";
            int backend_log_level = ACL_LOG_LEVEL_INFO;
            triton::common::TritonJson::Value cmdline;
            if (backend_config.Find("cmdline", &cmdline))
            {
                triton::common::TritonJson::Value log_file_value;
                std::string log_file_value_str;
                if (cmdline.Find("log_file", &log_file_value))
                {
                    LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("parse log_file from backend configuration")).c_str());
                    RETURN_IF_ERROR(log_file_value.AsString(&log_file_value_str));
                    backend_log_file = log_file_value_str;
                }

                // defaule log level is INFO
                triton::common::TritonJson::Value log_level_value;
                std::string log_level_value_str;
                if (cmdline.Find("log_level", &log_level_value))
                {
                    LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("parse log_level from backend configuration")).c_str());
                    try
                    {
                        RETURN_IF_ERROR(log_level_value.AsString(&log_level_value_str));
                        auto log_level_value_int = std::stol(log_level_value_str);
                        backend_log_level = log_level_value_int;
                    }
                    catch (const std::invalid_argument& ia)
                    {
                        return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, ia.what());
                    }
                }
            }

            // init backend logger
            ACL_ENGINE::AclLog::Instance().initAclLog(backend_log_file, backend_log_level);

            return nullptr;  // success
        }

        // Triton calls TRITONBACKEND_Finalize when a backend is no longer
        // needed.
        //
        TRITONSERVER_Error* TRITONBACKEND_Finalize(TRITONBACKEND_Backend* backend)
        {
            return nullptr;  // success
        }

        TRITONSERVER_Error* TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
        {
            const char* cname;
            RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &cname));
            std::string name(cname);

            uint64_t version;
            RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &version));

            LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("TRITONBACKEND_ModelInitialize: ") + name + 
                " (version " + std::to_string(version) + ")").c_str());

            // Create a ModelState object and associate it with the
            // TRITONBACKEND_Model.
            ModelState* model_state = nullptr;
            RETURN_IF_ERROR(ModelState::Create(model, &model_state));
            RETURN_IF_ERROR(TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

            return nullptr;  // success
        }

        TRITONSERVER_Error* TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
        {
            void* vstate;
            RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
            ModelState* model_state = reinterpret_cast<ModelState*>(vstate);

            LOG_MESSAGE(TRITONSERVER_LOG_INFO, "TRITONBACKEND_ModelFinalize: delete model state");
            delete model_state;

            return nullptr;  // success
        }

        // Triton calls TRITONBACKEND_ModelInstanceInitialize when a model
        // instance is created to allow the backend to initialize any state
        // associated with the instance.
        //
        TRITONSERVER_Error* TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
        {
            const char* cname;
            RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceName(instance, &cname));
            std::string name(cname);

            int32_t device_id;
            RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id));
            TRITONSERVER_InstanceGroupKind kind;
            RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceKind(instance, &kind));

            LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("TRITONBACKEND_ModelInstanceInitialize: ") + name + " (" +
                TRITONSERVER_InstanceGroupKindString(kind) + " device " + std::to_string(device_id) + ")").c_str());

            // Get the model state associated with this instance's model
            TRITONBACKEND_Model* model;
            RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

            void* vmodelstate;
            RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
            ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);

            // With each instance we create a ModelInstanceState object and
            // associate it with the TRITONBACKEND_ModelInstance.
            ModelInstanceState* instance_state;
            RETURN_IF_ERROR(ModelInstanceState::Create(model_state, instance, &instance_state));
            RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(instance, reinterpret_cast<void*>(instance_state)));

            return nullptr;
        }

        // Triton calls TRITONBACKEND_ModelInstanceFinalize when a model
        // instance is no longer needed. The backend should cleanup any state
        // associated with the model instance.
        //
        TRITONSERVER_Error* TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
        {
            void* vstate;
            RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
            ModelInstanceState* instance_state = reinterpret_cast<ModelInstanceState*>(vstate);

            LOG_MESSAGE(TRITONSERVER_LOG_INFO, "TRITONBACKEND_ModelInstanceFinalize: delete instance state");
            delete instance_state;

            return nullptr;
        }

        // When Triton calls TRITONBACKEND_ModelInstanceExecute it is required
        // that a backend create a response for each request in the batch. A
        // response may be the output tensors required for that request or may
        // be an error that is returned in the response.
        //
        TRITONSERVER_Error* TRITONBACKEND_ModelInstanceExecute(TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests, const uint32_t request_count)
        {
            // Triton will not call this function simultaneously for the same
            // 'instance'. But since this backend could be used by multiple
            // instances from multiple models the implementation needs to handle
            // multiple calls to this function at the same time (with different
            // 'instance' objects). Suggested practice for this is to use only
            // function-local and model-instance-specific state (obtained from
            // 'instance'), which is what we do here.
            ModelInstanceState* instance_state;
            RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, reinterpret_cast<void**>(&instance_state)));
            ModelState* model_state = instance_state->StateForModel();

            LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("model ") + model_state->Name() + ", instance " +
                instance_state->Name() + ", executing " + std::to_string(request_count) + " requests").c_str());

            // At this point we accept ownership of 'requests', which means that
            // even if something goes wrong we must still return success from
            // this function. If something does go wrong in processing a
            // particular request then we send an error response just for the
            // specific request.
            instance_state->ProcessRequests(requests, request_count);

            return nullptr;  // success
        }

    }  // extern "C"
}  // namespace triton::backend::acl