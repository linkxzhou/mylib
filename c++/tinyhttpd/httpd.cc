#include "httpd.h"
#include <algorithm>
#include <string>
#include <sys/stat.h>

const std::string HTDOCS_PATH = "htdocs";

static void* __accept_request(void *arg)
{
    // Convert argument to HttpdSocketPtr
    HttpdSocketPtr socket = static_cast<HttpdSocketPtr>(arg);
    LOG("[New connection]");
    struct stat st;
    socket->parseMethod();
    if (socket->isGET()) {
        LOG("[GET request received]");
    } else if (socket->isPOST()) {
        LOG("[POST request received]");
    } else {
        LOG("[Unsupported HTTP method] %s", socket->getMethod().c_str());
        socket->error501();
        return nullptr;
    }
    // Parse URL and headers
    socket->parseUrl();
    socket->parseHeader();
    LOG("[Request parsing done] url:%s", socket->getUrl().c_str());

    // Use std::string for safer path manipulation
    std::string path = socket->getUrl();
    if (path.empty() || path == "/") {
        path = HTDOCS_PATH + "/index.html";
        LOG("[Default homepage] %s", path.c_str());
    } else {
        path = HTDOCS_PATH + path;
        LOG("[Requested path] %s", path.c_str());
    }
    LOG("[2]path:%s, m_url_:%s", path.c_str(), socket->getUrl().c_str());

    // Stat the file
    if (stat(path.c_str(), &st) == -1) {
        LOG("[Path not found] %s", path.c_str());
        socket->discardBody();
        socket->error404();
        socket->close();
        socket->getHttpd()->freeObject(socket);
        return nullptr;
    }

    // If directory, append /index.html
    if (S_ISDIR(st.st_mode)) {
        if (path.back() != '/') path += "/";
        path += "index.html";
        LOG("[Directory auto-completed] %s", path.c_str());
    }
    LOG("cgi:%d, path:%s", socket->cgi(), path.c_str());
    if (!socket->cgi()) {
        LOG("[File served] %s", path.c_str());
        socket->serveFile(path.c_str());
    } else {
        LOG("[CGI execution] %s", path.c_str());
        socket->executeCGI(path.c_str());
    }
    socket->close();
    socket->getHttpd()->freeObject(socket);
    LOG("[Connection closed]");
    return nullptr;
}

void Httpd::startup(u_short port)
{
    struct sockaddr_in name;

    m_socket_fd_ = socket(PF_INET, SOCK_STREAM, 0);
    if (m_socket_fd_ == -1)
    {
        ERROR_DIE("socket");
    }
    
    memset(&name, 0, sizeof(name));
    name.sin_family = AF_INET;
    name.sin_port = htons(port);
    name.sin_addr.s_addr = htonl(INADDR_ANY);
    if (::bind(m_socket_fd_, (struct sockaddr *)&name, sizeof(name)) < 0)
    {
        ERROR_DIE("bind");
    }
    
    if (port == 0)
    {
        socklen_t namelen = sizeof(name);
        if (getsockname(m_socket_fd_, (struct sockaddr *)&name, &namelen) == -1)
        {
            ERROR_DIE("getsockname");
        }
        port = ntohs(name.sin_port);
    }

    if (listen(m_socket_fd_, 5) < 0)
    {
        ERROR_DIE("listen");
    }

    // 执行循环处理
    loop();
}

void Httpd::loop()
{
    pthread_t pthread;

    while (true)
    {
        struct sockaddr_in client_name;
        socklen_t client_name_len = sizeof(client_name);

        int fd = accept(m_socket_fd_,
                        (struct sockaddr *)&client_name,
                        &client_name_len);
        if (fd == -1)
        {
            ERROR_DIE("accept");
        }

        HttpdSocketPtr s = newObject();
        s->setClientFd(fd);
        s->setClientName(client_name);
        s->setHttpd(this);

        if (pthread_create(&pthread , NULL, __accept_request, s) != 0)
        {
            ERROR("pthread_create");
        }
    }

    ::close(m_socket_fd_);
}

void HttpdSocket::serveFile(const char *path)
{
    FILE *resource = NULL;
    discardBody();

    resource = ::fopen(path, "r");
    LOG("path:%s", path);
    if (resource == NULL)
    {
        error404();
    }
    else
    {
        static string s = string("HTTP/1.0 200 OK\r\n") + 
            SERVER_STRING + 
            "Content-Type: text/html\r\n" +
            "\r\n";
        ::send(m_client_fd_, s.c_str(), strlen(s.c_str()), 0);

        while (!::feof(resource))
        {
            memset(m_buffer_, 0, sizeof(m_buffer_));
            ::fgets(m_buffer_, sizeof(m_buffer_), resource);
            LOG("buffer:%s", m_buffer_);
            ::send(m_client_fd_, m_buffer_, strlen(m_buffer_), 0);
        }
    }

    ::fclose(resource);
}

void HttpdSocket::executeCGI(const char* path)
{
    // 管道
    int cgi_output[2], cgi_input[2];
    pid_t pid;
    int status;
    char c;

    static string s = string("HTTP/1.0 200 OK\r\n") + SERVER_STRING;
    ::send(m_client_fd_, s.c_str(), strlen(s.c_str()), 0);

    if (pipe(cgi_output) < 0) 
    {
        error500();
        return ;
    }

    if (pipe(cgi_input) < 0) 
    {
        error500();
        return ;
    }

    if ((pid = fork()) < 0) 
    {
        error500();
        return ;
    }

    int content_len = getContentLength();
    // 运行cgi脚本
    if (pid == 0)
    {
        char meth_env[255], query_env[255], length_env[255];

        ::dup2(cgi_output[1], 1);  // stdout -> pipe
        ::dup2(cgi_output[1], 2);  // stderr -> pipe
        ::dup2(cgi_input[0], 0);

        ::close(cgi_output[0]);
        ::close(cgi_input[1]);

        snprintf(meth_env, sizeof(meth_env) - 1, "REQUEST_METHOD=%s", m_method_.c_str());
        putenv(meth_env);
        if (isGET())
        {
            size_t pos = m_query_.find('?');
            if (pos != std::string::npos) {
                m_query_ = m_query_.substr(0, pos);
            }
            snprintf(query_env, sizeof(query_env) - 1, "QUERY_STRING=%s", m_query_.c_str());
        }

        snprintf(length_env, sizeof(length_env) - 1, "CONTENT_LENGTH=%d", content_len);
        putenv(length_env);

        execl(path, path, NULL);
        int err = errno;
        if (err != 0) {
            printf("[ERROR] CGI exec error: %s\n", strerror(err));
        }
        fflush(stdout);
        exit(1);
    } 
    else
    {
        ::close(cgi_output[1]);
        ::close(cgi_input[0]);
        
        if (isPOST())
        {
            for (int i = 0; i < content_len; i++) 
            {
                ::recv(m_client_fd_, &c, 1, 0);
                ::write(cgi_input[1], &c, 1);
            }
        }
        LOG("pid:%d, content_len:%d", pid, content_len);
        // 读取CGI输出到字符串
        std::string cgi_result;
        char buf[1024];
        ssize_t n;
        while ((n = ::read(cgi_output[0], buf, sizeof(buf))) > 0) {
            cgi_result.append(buf, n);
        }
        ::close(cgi_output[0]);
        ::close(cgi_input[1]);

        ::waitpid(pid, &status, 0);
        LOG("status:%d", status);
        // 检查是否有Content-Type头
        if (cgi_result.find("Content-Type:") == std::string::npos) {
            std::string header = "Content-Type: text/plain\r\n\r\n";
            ::send(m_client_fd_, header.c_str(), header.size(), 0);
        }
        ::send(m_client_fd_, cgi_result.c_str(), cgi_result.size(), 0);
    }
}

// 解析方法
void HttpdSocket::parseMethod()
{
    // 获取数据
    if (m_buffer_xi_ > m_buffer_len_ || m_buffer_len_ == 0)
    {
        getLine();
        m_buffer_xi_ = 0;

        LOG("body:%s, m_buffer_len_:%d", sanitize_newlines(m_buffer_).c_str(), m_buffer_len_);
    }
    
    // 跳过空格
    while (isspace(m_buffer_[m_buffer_xi_]) && (m_buffer_xi_ < sizeof(m_buffer_))) m_buffer_xi_++;

    size_t xi = 0;
    while (!isspace(m_buffer_[m_buffer_xi_]) && 
        (xi < sizeof(m_method_) - 1) && 
        (m_buffer_xi_ < sizeof(m_buffer_)))
    {
        m_method_[xi] = m_buffer_[m_buffer_xi_];
        xi++; 
        m_buffer_xi_++;
    }

    m_method_[m_buffer_xi_] = '\0';

    LOG("method:%s, m_buffer_xi_:%d, m_buffer_len_:%d", m_method_.c_str(), m_buffer_xi_, m_buffer_len_);
}

void HttpdSocket::parseUrl()
{
    // 获取数据
    if (m_buffer_xi_ >= m_buffer_len_ || m_buffer_len_ == 0)
    {
        getLine();
        m_buffer_xi_ = 0;

        LOG("body:%s", sanitize_newlines(m_buffer_).c_str());
    }

    while (isspace(m_buffer_[m_buffer_xi_]) && (m_buffer_xi_ < sizeof(m_buffer_))) m_buffer_xi_++;

    size_t xi = 0;
    while (!isspace(m_buffer_[m_buffer_xi_]) && (xi < sizeof(m_url_) - 1) && (m_buffer_xi_ < sizeof(m_buffer_)))
    {
        m_url_[xi] = m_buffer_[m_buffer_xi_];
        xi++; 
        m_buffer_xi_++;
    }

    m_url_[xi] = '\0';

    if (strcasecmp(m_method_.c_str(), "GET") == 0)
    {
        m_query_ = m_url_;
        size_t pos = m_query_.find('?');
        if (pos != std::string::npos) {
            m_query_ = m_query_.substr(0, pos);
        }
    }

    LOG("m_url_:%s, m_buffer_xi_:%d", m_url_, m_buffer_xi_);
    // 跳过解析协议
    m_buffer_xi_ = m_buffer_len_;
}

void HttpdSocket::parseHeader()
{
    while (true)
    {
        // 获取数据
        if (m_buffer_xi_ >= m_buffer_len_ || m_buffer_len_ == 0)
        {
            getLine();
            m_buffer_xi_ = 0;

            LOG("body:%s, m_buffer_len_:%d", sanitize_newlines(m_buffer_).c_str(), m_buffer_len_);
        }

        while (isspace(m_buffer_[m_buffer_xi_]) && (m_buffer_xi_ < sizeof(m_buffer_))) m_buffer_xi_++;

        if (m_buffer_len_ <= 0)
        {
            break;
        }

        if (strcmp("\r", m_buffer_) == 0 ||
            strcmp("\n", m_buffer_) == 0)
        {
            break; // header的解析终止
        }

        LOG("header:%s", sanitize_newlines(m_buffer_).c_str());
        string s1, s2;
        m_buffer_xi_ = split(m_buffer_, s1, s2);
        if (m_buffer_xi_ > 1)
        {
            transform(s1.begin(), s1.end(), s1.begin(), ::tolower);
            m_header_[s1] = s2;
        }

        LOG("[Header parsing] m_buffer_xi: %d, m_buffer_len: %d, header: %s", 
            m_buffer_xi_, m_buffer_len_, sanitize_newlines(m_buffer_).c_str());
    }

    LOG("parseHeader end");
}

void HttpdSocket::error501()
{
    static string s = string("HTTP/1.0 501 Method Not Implemented\r\n") + 
        SERVER_STRING + 
        "Content-Type: text/html\r\n" +
        "\r\n" + 
        "<HTML><HEAD><TITLE>Method Not Implemented\r\n" +
        "</TITLE></HEAD>\r\n" +
        "<BODY>HTTP request method not supported.\r\n" +
        "</BODY></HTML>\r\n";

    ::send(m_client_fd_, s.c_str(), strlen(s.c_str()), 0);
}

void HttpdSocket::error500()
{
    static string s = string("HTTP/1.0 500 Internal Server Error\r\n") + 
        "Content-Type: text/html\r\n" +
        "\r\n" + 
        "Error prohibited CGI execution.\r\n";

    ::send(m_client_fd_, s.c_str(), strlen(s.c_str()), 0);
}

void HttpdSocket::error400()
{
    static string s = string("HTTP/1.0 400 BAD REQUEST\r\n") + 
        "Content-type: text/html\r\n" +
        "\r\n" + 
        "Your browser sent a bad request, " +
        "such as a POST without a Content-Length.\r\n";

    ::send(m_client_fd_, s.c_str(), strlen(s.c_str()), 0);
}

void HttpdSocket::error404()
{
    static string s = string("HTTP/1.0 404 NOT FOUND\r\n") + 
        SERVER_STRING + 
        "Content-type: text/html\r\n" +
        "\r\n" + 
        "<HTML><TITLE>Not Found</TITLE>\r\n" +
        "<BODY>The server could not fulfill\r\n" +
        "your request because the resource specified\r\n" +
        "is unavailable or nonexistent.\r\n" +
        "</BODY></HTML>\r\n";
    ::send(m_client_fd_, s.c_str(), strlen(s.c_str()), 0);
}

int main(int argc, char **argv)
{
    int port = 8081;
    if (argc > 1) {
        try {
            port = std::stoi(argv[1]);
        } catch (const std::exception &e) {
            LOG("Invalid port argument, using default port %d", port);
        }
    }
    Httpd httpd;
    LOG("startup port:%d", port);
    httpd.startup(port);
    return 0;
}