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
#include <algorithm>
#include <functional>
#include <iterator>
#include <numeric>
#include <sstream>
#ifdef _WIN32
// suppress the min and max definitions in Windef.h.
#define NOMINMAX
#include <Windows.h>
#else
#include <dlfcn.h>
#endif
#include "acl_utils.h"

namespace triton::backend::acl
{

    AclTensorDataType ConvertDataType(TRITONSERVER_DataType dtype)
    {
        switch (dtype)
        {
            case TRITONSERVER_TYPE_INVALID:
                return AclTensor::TENSOR_DATA_TYPE_MAX;
            case TRITONSERVER_TYPE_UINT8:
                return AclTensor::TENSOR_DATA_TYPE_UINT8;
            case TRITONSERVER_TYPE_UINT16:
                return AclTensor::TENSOR_DATA_TYPE_UINT16;
            case TRITONSERVER_TYPE_UINT32:
                return AclTensor::TENSOR_DATA_TYPE_UINT32;
            case TRITONSERVER_TYPE_UINT64:
                return AclTensor::TENSOR_DATA_TYPE_UINT64;
            case TRITONSERVER_TYPE_INT8:
                return AclTensor::TENSOR_DATA_TYPE_INT8;
            case TRITONSERVER_TYPE_INT16:
                return AclTensor::TENSOR_DATA_TYPE_INT16;
            case TRITONSERVER_TYPE_INT32:
                return AclTensor::TENSOR_DATA_TYPE_INT32;
            case TRITONSERVER_TYPE_INT64:
                return AclTensor::TENSOR_DATA_TYPE_INT64;
            case TRITONSERVER_TYPE_FP16:
                return AclTensor::TENSOR_DATA_TYPE_FLOAT16;
            case TRITONSERVER_TYPE_FP32:
                return AclTensor::TENSOR_DATA_TYPE_FLOAT32;
            default:
                break;
        }
        return AclTensor::TENSOR_DATA_TYPE_MAX;
    }

    AclTensorDataType ConvertDataType(const std::string& dtype)
    {
        if (dtype == "TYPE_INVALID")
            return AclTensor::TENSOR_DATA_TYPE_MAX;
        else if (dtype == "TYPE_UINT8")
            return AclTensor::TENSOR_DATA_TYPE_UINT8;
        else if (dtype == "TYPE_UINT16")
            return AclTensor::TENSOR_DATA_TYPE_UINT16;
        else if (dtype == "TYPE_UINT32")
            return AclTensor::TENSOR_DATA_TYPE_UINT32;
        else if (dtype == "TYPE_UINT64")
            return AclTensor::TENSOR_DATA_TYPE_UINT64;
        else if (dtype == "TYPE_INT8")
            return AclTensor::TENSOR_DATA_TYPE_INT8;
        else if (dtype == "TYPE_INT16")
            return AclTensor::TENSOR_DATA_TYPE_INT16;
        else if (dtype == "TYPE_INT32")
            return AclTensor::TENSOR_DATA_TYPE_INT32;
        else if (dtype == "TYPE_INT64")
            return AclTensor::TENSOR_DATA_TYPE_INT64;
        else if (dtype == "TYPE_FP16")
            return AclTensor::TENSOR_DATA_TYPE_FLOAT16;
        else if (dtype == "TYPE_FP32")
            return AclTensor::TENSOR_DATA_TYPE_FLOAT32;
        else
            return AclTensor::TENSOR_DATA_TYPE_MAX;
    }

    TRITONSERVER_DataType ConvertDataType(AclTensorDataType dtype)
    {
        switch (dtype)
        {
            case AclTensor::TENSOR_DATA_TYPE_MAX:
                return TRITONSERVER_TYPE_INVALID;
            case AclTensor::TENSOR_DATA_TYPE_UINT8:
                return TRITONSERVER_TYPE_UINT8;
            case AclTensor::TENSOR_DATA_TYPE_UINT16:
                return TRITONSERVER_TYPE_UINT16;
            case AclTensor::TENSOR_DATA_TYPE_UINT32:
                return TRITONSERVER_TYPE_UINT32;
            case AclTensor::TENSOR_DATA_TYPE_UINT64:
                return TRITONSERVER_TYPE_UINT64;
            case AclTensor::TENSOR_DATA_TYPE_INT8:
                return TRITONSERVER_TYPE_INT8;
            case AclTensor::TENSOR_DATA_TYPE_INT16:
                return TRITONSERVER_TYPE_INT16;
            case AclTensor::TENSOR_DATA_TYPE_INT32:
                return TRITONSERVER_TYPE_INT32;
            case AclTensor::TENSOR_DATA_TYPE_INT64:
                return TRITONSERVER_TYPE_INT64;
            case AclTensor::TENSOR_DATA_TYPE_FLOAT32:
                return TRITONSERVER_TYPE_FP32;
            case AclTensor::TENSOR_DATA_TYPE_FLOAT16:
                return TRITONSERVER_TYPE_FP16;
            case AclTensor::TENSOR_DATA_TYPE_STRING:
                return TRITONSERVER_TYPE_BYTES;
            default:
                break;
        }
        return TRITONSERVER_TYPE_INVALID;
    }

    std::string DirName(const std::string& path)
    {
        if (path.empty())
            return path;

        size_t last = path.size() - 1;
        while ((last > 0) && (path[last] == '/'))
            last -= 1;

        if (path[last] == '/')
            return std::string("/");

        const size_t idx = path.find_last_of("/", last);
        if (idx == std::string::npos)
            return std::string(".");

        if (idx == 0)
            return std::string("/");

        return path.substr(0, idx);
    }

    TRITONSERVER_Error* SetLibraryDirectory(const std::string& path)
    {
    #ifdef _WIN32
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("SetLibraryDirectory: path = ") + path).c_str());
        if (!SetDllDirectory(path.c_str()))
        {
            LPSTR err_buffer = nullptr;
            size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                FORMAT_MESSAGE_IGNORE_INSERTS, NULL, GetLastError(), MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                (LPSTR)&err_buffer, 0, NULL);
            std::string errstr(err_buffer, size);
            LocalFree(err_buffer);

            return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_NOT_FOUND,
                ("unable to set dll path " + path + ": " + errstr).c_str());
        }
    #endif

        return nullptr;  // success
    }

    TRITONSERVER_Error* ResetLibraryDirectory()
    {
    #ifdef _WIN32
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, "ResetLibraryDirectory");
        if (!SetDllDirectory(NULL))
        {
            LPSTR err_buffer = nullptr;
            size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                    FORMAT_MESSAGE_IGNORE_INSERTS, NULL, GetLastError(), MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                (LPSTR)&err_buffer, 0, NULL);
            std::string errstr(err_buffer, size);
            LocalFree(err_buffer);

            return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_NOT_FOUND,
                ("unable to reset dll path: " + errstr).c_str());
        }
    #endif

        return nullptr;  // success
    }

    TRITONSERVER_Error* OpenLibraryHandle(const std::string& path, void** handle)
    {
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("OpenLibraryHandle: ") + path).c_str());

    #ifdef _WIN32
        // Need to put shared library directory on the DLL path so that any
        // dependencies of the shared library are found
        const std::string library_dir = DirName(path);
        RETURN_IF_ERROR(SetLibraryDirectory(library_dir));

        // HMODULE is typedef of void*
        // https://docs.microsoft.com/en-us/windows/win32/winprog/windows-data-types
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, (std::string("OpenLibraryHandle: path = ") + path).c_str());
        *handle = LoadLibrary(path.c_str());

        // Remove the dll path added above... do this unconditionally before
        // check for failure in dll load.
        RETURN_IF_ERROR(ResetLibraryDirectory());

        if (*handle == nullptr)
        {
            LPSTR err_buffer = nullptr;
            size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                FORMAT_MESSAGE_IGNORE_INSERTS, NULL, GetLastError(), MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                (LPSTR)&err_buffer, 0, NULL);
            std::string errstr(err_buffer, size);
            LocalFree(err_buffer);

            return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_NOT_FOUND,
                ("unable to load shared library: " + errstr).c_str());
        }
    #else
        *handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (*handle == nullptr)
        {
            return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_NOT_FOUND,
                ("unable to load shared library: " + std::string(dlerror())).c_str());
        }
    #endif

        return nullptr;  // success
    }

    TRITONSERVER_Error* CloseLibraryHandle(void* handle)
    {
        if (handle != nullptr)
        {
    #ifdef _WIN32
            if (FreeLibrary((HMODULE)handle) == 0) 
            {
                LPSTR err_buffer = nullptr;
                size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                    FORMAT_MESSAGE_IGNORE_INSERTS, NULL, GetLastError(), MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                    (LPSTR)&err_buffer, 0, NULL);
                std::string errstr(err_buffer, size);
                LocalFree(err_buffer);
                return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, 
                    ("unable to unload shared library: " + errstr).c_str());
            }
    #else
            if (dlclose(handle) != 0)
            {
                return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL,
                    ("unable to unload shared library: " + std::string(dlerror())).c_str());
            }
    #endif
        }

        return nullptr;  // success
    }

} // namespace triton::backend::acl