/********************************************
 * @Author: zhaojd-a
 * @Date: 2024-06-13
 * @LastEditTime: 2024-06-13
 * @LastEditors: zhaojd-a
 ********************************************/
#pragma once
#include <iostream>
#include <string>
#include <fstream>
#include <memory>
#include "mio/mio.h"
#include "acl_engine/base_stream.h"

#define LARGE_FILE_THRESHOLD  512*1024*1024

namespace ACL_ENGINE
{

    class FileInputStream : public BaseInputStream
    {
    public:
        FileInputStream(const std::string file);
        virtual ~FileInputStream();
        size_t read(char* buf, size_t len) override;
        const char* getFileData();
        std::string getFile() { return m_file; }
        std::ifstream& getStream() { return m_in; }
        bool isOpen();
        virtual const char* getWorkPath() override { return m_workpath.c_str(); }
        size_t getFileSize() { return m_size; }

    private:
        size_t                      m_size = 0;
        std::ifstream               m_in;
        std::string                 m_file;
        std::string                 m_workpath;
        // file data for call getFileData when use fstream
        std::shared_ptr<char>       m_file_data;
        // large file size
        mio::shared_mmap_source     m_map_file;
    };

    class FileOutputStream : public BaseOutputStream 
    {
    public:
        FileOutputStream(const std::string file);
        virtual ~FileOutputStream();
        size_t write(const char* buf, size_t len) override;
        std::string getFile() { return m_file; }
        std::ofstream& getStream() { return m_out; }
        bool isOpen();
        virtual const char* getWorkPath() override { return m_workpath.c_str(); }

    private:
        std::ofstream               m_out;
        std::string                 m_file;
        std::string                 m_workpath;
    };

    int loadFileData(const std::string& file_name, std::shared_ptr<char>& load_data, size_t& load_len);
    int saveFileData(const std::string& file_name, const std::shared_ptr<char>& save_data, const size_t save_len);

} // namespace ACL_ENGINE