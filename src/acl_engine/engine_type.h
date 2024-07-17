/********************************************
 * @Author: zhaojd-a
 * @Date: 2024-06-13
 * @LastEditTime: 2024-06-13
 * @LastEditors: zhaojd-a
 ********************************************/
#pragma once
#include <iostream>
#include <string>

namespace ACL_ENGINE
{

    typedef struct EngineConfig
    {
        int                                       device_id = -1;                              // ascend core id
        std::string                               config_file = "";                            // model config file
    } EngineConfig;

} // namespace ACL_ENGINE