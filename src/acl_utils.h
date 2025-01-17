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
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include "triton/backend/backend_common.h"
#include "triton/core/tritonserver.h"
#include "acl_engine/engine_tensor.h"
using AclTensor = ACL_ENGINE::EngineTensor;
using AclTensorDataType = ACL_ENGINE::EngineTensor::TensorDataType;

namespace triton::backend::acl
{

    AclTensorDataType ConvertDataType(TRITONSERVER_DataType dtype);
    AclTensorDataType ConvertDataType(const std::string& dtype);
    TRITONSERVER_DataType ConvertDataType(AclTensorDataType dtype);

    TRITONSERVER_Error* SetLibraryDirectory();
    TRITONSERVER_Error* ResetLibraryDirectory();
    TRITONSERVER_Error* OpenLibraryHandle(const std::string& path, void** handle);
    TRITONSERVER_Error* CloseLibraryHandle(void* handle);

} // namespace triton::backend::acl
