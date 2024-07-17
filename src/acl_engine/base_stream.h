/********************************************
 * @Author: zhaojd-a
 * @Date: 2024-06-13
 * @LastEditTime: 2024-06-13
 * @LastEditors: zhaojd-a
 ********************************************/
#pragma once
#include <stdint.h>
#include "acl_engine/non_copyable.h"

namespace ACL_ENGINE
{

    class BaseInputStream : public NonCopyable
    {
    public:
        virtual size_t read(char *buf, size_t len) = 0;
        virtual ~BaseInputStream() {}
        virtual const char* getWorkPath() { return ""; }
    };

    class BaseOutputStream : public NonCopyable 
    {
    public:
        virtual size_t write(const char *buf, size_t len) = 0;
        virtual ~BaseOutputStream() {}
        virtual const char* getWorkPath() { return ""; }
    };

} // namespace ACL_ENGINE