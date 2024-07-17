/********************************************
 * @Author: zhaojd-a
 * @Date: 2024-06-13
 * @LastEditTime: 2024-06-13
 * @LastEditors: zhaojd-a
 ********************************************/
#include "acl_engine/log.h"
#include "ghc/filesystem.hpp"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/rotating_file_sink.h"
namespace fs = ghc::filesystem;

#define LOG_FILE_SIZE 50 * 1024 * 1024
#define LOG_FILE_NUM 3

namespace ACL_ENGINE 
{

    AclLog::AclLog()
    {
        // std::string log_file_name = "./log.txt";
        // int log_level = ACL_LOG_LEVEL_INFO;
        // initAclLog(log_file_name, log_level);
    }

    AclLog& AclLog::Instance()
    {
        static AclLog log;
        return log;
    }

    void AclLog::initAclLog(std::string file_name, int log_level)
    {
        // set log level
        if (log_level != ACL_LOG_LEVEL_TRACE && log_level != ACL_LOG_LEVEL_DEBUG && 
            log_level != ACL_LOG_LEVEL_INFO  && log_level != ACL_LOG_LEVEL_WARN  && 
            log_level != ACL_LOG_LEVEL_ERROR && log_level != ACL_LOG_LEVEL_FATAL && 
            log_level != ACL_LOG_LEVEL_OFF)
            log_level = ACL_LOG_LEVEL_INFO;

        // set log rotate
        int log_file_size = LOG_FILE_SIZE;
        if (log_file_size <= 0)
            log_file_size = 50 * 1024 * 1024; // 50MB
        int num_log_files = LOG_FILE_NUM;
            num_log_files = 3;
        if (file_name.empty())
            file_name = "gldai_algo.log";

        // get absoulte path of log file
        fs::path log_path{file_name};
        fs::path log_full_path = fs::absolute(log_path);
        file_name = log_full_path.string();

        std::string logger_name = "acl_engine";
        std::cout << "log level: " << log_level << std::endl;
        std::cout << "logger name: " << logger_name << std::endl;
        std::cout << "log file path: " << file_name << std::endl;

        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(file_name, log_file_size, num_log_files);
        m_logger.reset();
        m_logger = std::unique_ptr<spdlog::logger>(new spdlog::logger(logger_name, {console_sink, file_sink}));
        spdlog::register_logger(m_logger);
        // set log pattern
        m_logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e %z] [%n] [%^---%L---%$] [thread %t] [%s %! %#] %v");
        spdlog::set_default_logger(spdlog::get(logger_name));
        // spdlog::flush_on(spdlog::level::info);
        spdlog::flush_on(spdlog::level::debug);
        spdlog::set_level(static_cast<spdlog::level::level_enum>(log_level));
    }

    void AclLog::stopAclLog()
    {
        spdlog::shutdown();
        m_logger.reset();
    }

    // AclLog::~AclLog()
    // {
    //     if (nullptr != m_logger.get())
    //     {
    //         stopAclLog();
    //     }
    // }

    void AclLog::setLevel(int level)
    {
        spdlog::set_level(static_cast<spdlog::level::level_enum>(level));
    }

} // namespace ACL_ENGINE