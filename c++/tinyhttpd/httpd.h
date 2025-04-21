#pragma once

// Standard C++ headers
#include <cstddef>
#include <array>
#include <string>
#include <queue>
#include <map>
#include <sstream>

// System headers
#include <sys/socket.h>
#include <unistd.h>
#include <strings.h>
#include <netinet/in.h>

using namespace std;

#define SERVER_STRING "Server: httpd++/1.0.0\r\n"
#define ERROR_DIE(msg)  do              \
    {                                   \
        perror("[ERROR_DIE]"#msg);      \
        exit(1);                        \
    } while(0)

#define ERROR(msg)      do              \
    {                                   \
        perror("[ERROR]"#msg);          \
    } while(0)

#ifdef DEBUG
#define LOG(fmt, ...)   fprintf(stdout, "[%s:%u]" fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#else
#define LOG(fmt, ...)
#endif

constexpr std::size_t MAX_BUF_SIZE = 1024;

inline std::string sanitize_newlines(const std::string& input) {
    std::ostringstream oss;
    for (char c : input) {
        if (c == '\r') oss << "\\r";
        else if (c == '\n') oss << "\\n";
        else oss << c;
    }
    return oss.str();
}

class Httpd;

/**
 * Handles individual HTTP connections and request parsing.
 */
class HttpdSocket {
public:
    HttpdSocket() = default;
    HttpdSocket(int fd, const struct sockaddr_in &s) :
        m_client_fd_(fd), m_client_name_(s), m_buffer_xi_(0),
        m_query_(), m_buffer_len_(0), m_httpd_(nullptr) {}
    virtual ~HttpdSocket() { reset(); }
    HttpdSocket(const HttpdSocket&) = delete;
    HttpdSocket& operator=(const HttpdSocket&) = delete;
    HttpdSocket(HttpdSocket&&) = delete;
    HttpdSocket& operator=(HttpdSocket&&) = delete;

    /**
     * Sets the client file descriptor.
     * @param fd The client file descriptor.
     */
    void setClientFd(int fd) { m_client_fd_ = fd; }

    /**
     * Sets the client name.
     * @param client The client name.
     */
    void setClientName(const struct sockaddr_in &client) { m_client_name_ = client; }

    /**
     * Sets the Httpd instance.
     * @param h The Httpd instance.
     */
    void setHttpd(Httpd *h) { m_httpd_ = h; }

    /**
     * Gets the Httpd instance.
     * @return The Httpd instance.
     */
    Httpd* getHttpd() const { return m_httpd_; }

    /**
     * Closes the client connection.
     */
    void close() {
        if (m_client_fd_ > 0) {
            ::close(m_client_fd_);
        }
    }

    /**
     * Resets the socket state.
     */
    void reset() {
        m_client_fd_ = 0;
        m_buffer_xi_ = 0;
        m_query_.clear();
        m_buffer_len_ = 0;
        m_httpd_ = nullptr;
        m_method_.clear();
        m_header_.clear();

        memset(m_buffer_, 0, sizeof(m_buffer_));
        memset(m_url_, 0, sizeof(m_url_));
    }

    /**
     * Parses the HTTP method.
     */
    void parseMethod();

    /**
     * Parses the URL.
     */
    void parseUrl();

    /**
     * Parses the HTTP headers.
     */
    void parseHeader();

    /**
     * Gets the content length.
     * @return The content length.
     */
    int getContentLength() const {
        auto iter = m_header_.find("content-length");
        if (iter != m_header_.end()) {
            return atoi((iter->second).c_str());
        }
        return 0;
    }

    /**
     * Checks if the request is a POST request.
     * @return True if the request is a POST request, false otherwise.
     */
    bool isPOST() const {
        return strcasecmp(m_method_.c_str(), "POST") == 0;
    }

    /**
     * Checks if the request is a GET request.
     * @return True if the request is a GET request, false otherwise.
     */
    bool isGET() const {
        return strcasecmp(m_method_.c_str(), "GET") == 0;
    }

    /**
     * Gets the URL.
     * @return The URL.
     */
    const std::string getUrl() const { return std::string(m_url_); }

    /**
     * Gets the HTTP method.
     * @return The HTTP method string.
     */
    const std::string& getMethod() const { return m_method_; }

    /**
     * Checks if the request is a CGI request.
     * @return True if the request is a CGI request, false otherwise.
     */
    virtual bool cgi() const {
        return std::string(m_url_).find(".cgi") != std::string::npos;
    }

    /**
     * Sends a 501 error response.
     */
    virtual void error501();

    /**
     * Sends a 500 error response.
     */
    virtual void error500();

    /**
     * Sends a 404 error response.
     */
    virtual void error404();

    /**
     * Sends a 400 error response.
     */
    virtual void error400();

    /**
     * Serves a file.
     * @param path The file path.
     */
    virtual void serveFile(const char *path);

    /**
     * Executes a CGI script.
     * @param path The script path.
     */
    virtual void executeCGI(const char *path);

    /**
     * Discards the request body.
     * @return The number of bytes discarded.
     */
    int discardBody() {
        int len = 0, read_len = 0;
        while ((len > 0) && m_buffer_[0] != '\n') {
            len = getLine();
            read_len += len;
        }
        m_buffer_xi_ = 0;
        return read_len;
    }

    /**
     * Splits a string into key-value pairs.
     * @param str The input string.
     * @param key The key.
     * @param value The value.
     * @return The index of the next character to process.
     */
    int split(const char* str, std::string &key, std::string &value) const {
        int xi = 0;
        while (str[xi] != '\0' && isspace(str[xi])) xi++;
        // 先处理key
        while (str[xi] != '\0' && str[xi] != ':' && str[xi] != '\n') {
            key += static_cast<char>(tolower(str[xi]));
            xi++;
        }
        if (str[xi] == ':' || str[xi] == '\n') {
            xi++;
        }
        while (str[xi] != '\0' && isspace(str[xi])) xi++;
        while (str[xi] != '\0' && str[xi] != '\n') {
            value += static_cast<char>(tolower(str[xi]));
            xi++;
        }
        return xi + 1;
    }

    /**
     * Reads a line from the client.
     * @return The number of bytes read.
     */
    int getLine() {
        int i = 0, n = 0;
        char c = '\0';
        while ((i < static_cast<int>(MAX_BUF_SIZE) - 1) && (c != '\n')) {
            n = ::recv(m_client_fd_, &c, 1, 0);
            if (n > 0) {
                if (c == '\r') {
                    n = ::recv(m_client_fd_, &c, 1, MSG_PEEK);
                    if ((n > 0) && (c == '\n')) {
                        ::recv(m_client_fd_, &c, 1, 0);
                    } else {
                        c = '\n';
                    }
                }
                m_buffer_[i] = c;
                i++;
            } else {
                c = '\n';
            }
        }
        m_buffer_[i] = '\0';
        m_buffer_len_ = i;
        LOG("read line:%s", sanitize_newlines(std::string(m_buffer_)).c_str());
        return i;
    }

private:
    int m_client_fd_ = 0;
    struct sockaddr_in m_client_name_{};
    std::string m_query_;
    std::size_t m_buffer_xi_ = 0, m_buffer_len_ = 0;
    std::string m_method_;
    std::map<std::string, std::string> m_header_;
    Httpd *m_httpd_ = nullptr;

    char m_buffer_[MAX_BUF_SIZE] = {};
    char m_url_[MAX_BUF_SIZE] = {};
};

using HttpdSocketPtr = HttpdSocket*;

/**
 * Manages socket server and object pool for HttpdSocket.
 */
class Httpd {
public:
    Httpd() = default;
    ~Httpd() {
        // 清空m_queue_
        while (!m_queue_.empty()) {
            HttpdSocketPtr o = m_queue_.front();
            m_queue_.pop();
            delete o;
        }
    }
    Httpd(const Httpd&) = delete;
    Httpd& operator=(const Httpd&) = delete;
    Httpd(Httpd&&) = delete;
    Httpd& operator=(Httpd&&) = delete;

    /**
     * Starts the server.
     * @param port The port number.
     */
    void startup(u_short port);

    /**
     * Runs the server loop.
     */
    void loop();

    /**
     * Creates a new HttpdSocket instance.
     * @return The new HttpdSocket instance.
     */
    HttpdSocketPtr newObject() {
        if (m_queue_.empty()) {
            m_queue_.push(new HttpdSocket());
        }
        HttpdSocketPtr o = m_queue_.front();
        m_queue_.pop();
        return o;
    }

    /**
     * Frees an HttpdSocket instance.
     * @param o The HttpdSocket instance to free.
     */
    void freeObject(HttpdSocketPtr o) {
        if (o != nullptr) {
            o->reset();
            m_queue_.push(o);
        }
    }

private:
    int m_socket_fd_ = 0;
    std::queue<HttpdSocketPtr> m_queue_;
};