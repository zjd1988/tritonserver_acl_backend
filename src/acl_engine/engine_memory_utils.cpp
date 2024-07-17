/********************************************
 * @Author: zhaojd-a
 * @Date: 2024-06-13
 * @LastEditTime: 2024-06-13
 * @LastEditors: zhaojd-a
 ********************************************/
#include <stdint.h>
#include <stdlib.h>
#include "acl_engine/engine_memory_utils.h"
#include "acl_engine/log.h"

namespace ACL_ENGINE
{

    static inline void **alignPointer(void **ptr, size_t alignment)
    {
        return (void **)((intptr_t)((unsigned char *)ptr + alignment - 1) & -alignment);
    }

    void *memoryAllocAlign(size_t size, size_t alignment)
    {
        if (size <= 0)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "malloc size {} is not valid", size);
            return nullptr;
        }

    #ifdef ENGINE_DEBUG_MEMORY
        return malloc(size);
    #else
        void **origin = (void **)malloc(size + sizeof(void *) + alignment);
        if (!origin)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "malloc ptr is nullptr");
            return nullptr;
        }

        void **aligned = alignPointer(origin + 1, alignment);
        aligned[-1]    = origin;
        return aligned;
    #endif
    }

    void *memoryCallocAlign(size_t size, size_t alignment) 
    {
        if (size <= 0)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "malloc size {} is not valid", size);
            return nullptr;
        }
    #ifdef ENGINE_DEBUG_MEMORY
        return calloc(size, 1);
    #else
        void **origin = (void **)calloc(size + sizeof(void *) + alignment, 1);
        if (!origin)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "calloc ptr is nullptr");
            return nullptr;
        }
        void **aligned = alignPointer(origin + 1, alignment);
        aligned[-1]    = origin;
        return aligned;
    #endif
    }

    void memoryFreeAlign(void *aligned) 
    {
    #ifdef ENGINE_DEBUG_MEMORY
        free(aligned);
    #else
        if (aligned) {
            void *origin = ((void **)aligned)[-1];
            free(origin);
        }
    #endif
    }

} // namespace ACL_ENGINE