/********************************************
 * @Author: zhaojd-a
 * @Date: 2024-06-13
 * @LastEditTime: 2024-06-13
 * @LastEditors: zhaojd-a
 ********************************************/
#include "acl_engine/file_stream.h"
#include "acl_engine/log.h"
#include "ghc/filesystem.hpp"
namespace fs = ghc::filesystem;

namespace ACL_ENGINE
{

    // file input stream
    FileInputStream::FileInputStream(const std::string file)
    {
        // init member var
        m_size = 0;
        m_file = file;
        m_workpath = "";
        m_in.open(file.c_str(), std::ios_base::binary);
        if(!m_in.is_open())
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "open file {} failed.", file);
            return;
        }

        // get file size
        m_in.seekg(0, std::ios::end);
        m_size = m_in.tellg();
        m_in.seekg(0, std::ios::beg);
        if (0 >= m_size)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "file {} size:{} is <= 0, please check", file, m_size);
            return;
        }

        // get file work path
        fs::path fs_path{m_file};
        fs::path fs_work_path = fs::absolute(fs_path).remove_filename();
        m_workpath = fs_work_path.string();

        // check file size, if larger than
        if (LARGE_FILE_THRESHOLD < m_size)
        {
            ACL_LOG(ACL_LOG_LEVEL_INFO, "file {} size:{} is larger than {}, try to use mmap to open", 
                file, m_size, LARGE_FILE_THRESHOLD);
            m_in.close();
            std::error_code err;
            m_map_file = mio::make_mmap_source(file, err);
            if (err)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "use mmap open file {} fail with {}", file, err.message());
            }
            size_t m_map_file_size = m_map_file.size();
            if (m_map_file_size != m_size)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "file {} mmap size {} not equal to file size {}", 
                    file, m_map_file_size, m_size);
            }
        }
    }

    FileInputStream::~FileInputStream()
    {
        if (m_in.is_open())
            m_in.close();
        if (m_map_file.is_open())
            m_map_file.unmap();
    }

    size_t FileInputStream::read(char* buf, size_t len)
    {
        if (!m_in.is_open())
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "read api only support file stream not for mmap, please check");
            return 0;
        }
        ssize_t nread = 0;
        if (nullptr == buf)
        {
            return nread;
        }
        if(m_in.is_open())
        {
            m_in.read(buf, len);
            if(!m_in.bad()) 
            {
                nread = m_in.gcount();
            }
        }
        if(nread > 0)
        {
            return (size_t)nread;
        }
        else
        {
            return 0;
        }
    }

    bool FileInputStream::isOpen()
    {
        return m_in.is_open() || m_map_file.is_open();
    }

    const char* FileInputStream::getFileData()
    {
        if (m_in.is_open())
        {
            if (nullptr == m_file_data.get())
            {
                m_file_data.reset(new char[m_size], std::default_delete<char []>());
                if (m_size != read(m_file_data.get(), m_size))
                {
                    ACL_LOG(ACL_LOG_LEVEL_ERROR, "read data from file {} fail", m_file);
                    return nullptr;
                }
            }
            return (const char*)m_file_data.get();
        }
        else if(m_map_file.is_open())
        {
            return m_map_file.data();
        }
        else
            return nullptr;
    }

    FileOutputStream::FileOutputStream(const std::string file)
    {
        // init member var
        m_file = file;
        m_workpath = "";
        // get file work path
        fs::path fs_path{m_file};
        fs::path fs_work_path = fs::absolute(fs_path).remove_filename();
        m_workpath = fs_work_path.string();

        m_out.open(file.c_str(), std::ios_base::binary);
        if(!m_out.is_open())
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "open file {} failed.", file);
        }
    }

    FileOutputStream::~FileOutputStream()
    {
        if (m_out.is_open())
            m_out.close();
    }

    bool FileOutputStream::isOpen()
    {
        return m_out.is_open();
    }

    // file output stream
    size_t FileOutputStream::write(const char* buf, size_t len)
    {
        if (!m_out.is_open())
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "file {} is not open, please check", m_file);
            return 0;
        }
        size_t nwrite = 0;
        if(m_out.is_open()) 
        {
            m_out.write(buf, len);
            if(!m_out.bad())
            {
                nwrite = len;
            }
        }
        return nwrite;
    }

    int loadFileData(const std::string& file_name, std::shared_ptr<char>& load_data, size_t& load_len)
    {
        load_data.reset();
        auto file_stream = std::shared_ptr<FileInputStream>(new FileInputStream(file_name));
        if (nullptr == file_stream.get() || !file_stream->isOpen())
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "open file {} to read fail", file_name);
            return -1;
        }
        load_len = file_stream->getFileSize();
        if (load_len > 0)
        {
            load_data.reset(new char[load_len], std::default_delete<char []>());
            size_t read_len = file_stream->read(load_data.get(), load_len);
            if (read_len != load_len)
            {
                ACL_LOG(ACL_LOG_LEVEL_ERROR, "file contain {} bytes actually read {} bytes from file {}, please check", 
                    load_len, read_len, file_name);
                return -1;
            }
        }
        else
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "read file {} fail", file_name);
            return -1;
        }
        return 0;
    }

    int saveFileData(const std::string& file_name, const std::shared_ptr<char>& save_data, const size_t save_len)
    {
        if (nullptr == save_data.get())
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "data to wirte is nullptr, please check", file_name);
            return -1;
        }
        if (0 >= save_len)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "data to wirte {} bytes, please check", save_len);
            return -1;
        }
        auto file_stream = std::shared_ptr<FileOutputStream>(new FileOutputStream(file_name));
        if (nullptr == file_stream.get() || !file_stream->isOpen())
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "open file {} to wirte fail", file_name);
            return -1;
        }
        size_t write_len = file_stream->write(save_data.get(), save_len);
        if (write_len != save_len)
        {
            ACL_LOG(ACL_LOG_LEVEL_ERROR, "want to write {} bytes actually wirte {} bytes to file {}, please check",
                save_len, write_len, file_name);
            return false;
        }
        return 0;
    }

}  // namespace ACL_ENGINE