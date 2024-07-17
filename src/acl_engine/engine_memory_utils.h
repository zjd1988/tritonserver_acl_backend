/********************************************
 * @Author: zhaojd-a
 * @Date: 2024-06-13
 * @LastEditTime: 2024-06-13
 * @LastEditors: zhaojd-a
 ********************************************/
#pragma once
#include <stdio.h>
#define ENGINE_MEMORY_ALIGN_DEFAULT 64

namespace ACL_ENGINE
{

    void* memoryAllocAlign(size_t size, size_t align);
    void* memoryCallocAlign(size_t size, size_t align);
    void  memoryFreeAlign(void* mem);

} //namespace ACL_ENGINE