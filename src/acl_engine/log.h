/********************************************
 * @Author: zhaojd-a
 * @Date: 2024-06-13
 * @LastEditTime: 2024-06-13
 * @LastEditors: zhaojd-a
 ********************************************/
#pragma once
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#include <iostream>
#include <memory>
#include "spdlog/spdlog.h"
#include "mslite_engine/non_copyable.h"

#define ACL_LOG_LEVEL_TRACE    SPDLOG_LEVEL_TRACE
#define ACL_LOG_LEVEL_DEBUG    SPDLOG_LEVEL_DEBUG
#define ACL_LOG_LEVEL_INFO     SPDLOG_LEVEL_INFO
#define ACL_LOG_LEVEL_WARN     SPDLOG_LEVEL_WARN
#define ACL_LOG_LEVEL_ERROR    SPDLOG_LEVEL_ERROR
#define ACL_LOG_LEVEL_FATAL    SPDLOG_LEVEL_CRITICAL
#define ACL_LOG_LEVEL_OFF      SPDLOG_LEVEL_OFF

#define ACL_LOG_TRACE(...)       SPDLOG_TRACE(__VA_ARGS__)
#define ACL_LOG_DEBUG(...)       SPDLOG_DEBUG(__VA_ARGS__)
#define ACL_LOG_INFO(...)        SPDLOG_INFO(__VA_ARGS__)
#define ACL_LOG_WARN(...)        SPDLOG_WARN(__VA_ARGS__)
#define ACL_LOG_ERROR(...)       SPDLOG_ERROR(__VA_ARGS__)
#define ACL_LOG_CRITICAL(...)    SPDLOG_CRITICAL(__VA_ARGS__)

#define ACL_LOG_IMPL(level, ...)                                       \
do {                                                                   \
    switch(level)                                                      \
    {                                                                  \
        case ACL_LOG_LEVEL_TRACE:                                      \
            ACL_LOG_TRACE(__VA_ARGS__);                                \
            break;                                                     \
        case ACL_LOG_LEVEL_DEBUG:                                      \
            ACL_LOG_DEBUG(__VA_ARGS__);                                \
            break;                                                     \
        case ACL_LOG_LEVEL_INFO:                                       \
            ACL_LOG_INFO(__VA_ARGS__);                                 \
            break;                                                     \
        case ACL_LOG_LEVEL_WARN:                                       \
            ACL_LOG_WARN(__VA_ARGS__);                                 \
            break;                                                     \
        case ACL_LOG_LEVEL_ERROR:                                      \
            ACL_LOG_ERROR(__VA_ARGS__);                                \
            break;                                                     \
        case ACL_LOG_LEVEL_FATAL:                                      \
            ACL_LOG_CRITICAL(__VA_ARGS__);                             \
            break;                                                     \
        case ACL_LOG_LEVEL_OFF:                                        \
            break;                                                     \
        default:                                                       \
            std::string err_str = "unspported log level ...";          \
            ACL_LOG_CRITICAL(err_str, __VA_ARGS__);                    \
    }                                                                  \
} while(0)

#define ACL_LOG(level, ...)  ACL_LOG_IMPL(level, ##__VA_ARGS__)

namespace ACL_ENGINE
{

    class AclLog : public NonCopyable
    {
    public:
        static AclLog& Instance();
        void initAclLog(std::string file_name, int log_level = spdlog::level::trace);
        void stopAclLog();
        void setLevel(int level = spdlog::level::trace);
        spdlog::logger* getLogger()
        {
            return m_logger.get();
        }

    private:
        AclLog();
        ~AclLog() = default;

    private:
        std::shared_ptr<spdlog::logger> m_logger;
    };

} // namespace ACL_ENGINE