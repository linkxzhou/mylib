#include <boost/asio.hpp>
#include <boost/bind/bind.hpp>
#include <iostream>
#include <memory>
#include <cstring>
#include <cstdio>

using boost::asio::ip::tcp;

class session : public std::enable_shared_from_this<session> {
public:
    session(boost::asio::io_context& io_context)
        : socket_(io_context) {}
    
    tcp::socket& socket() {
        return socket_;
    }
    
    void start() {
        socket_.async_read_some(boost::asio::buffer(data_, max_length),
            boost::bind(&session::handle_read, shared_from_this(),
                boost::asio::placeholders::error,
                boost::asio::placeholders::bytes_transferred));
    }
    
private:
    void send_http_response(const char* body) {
        char response[2048];
        int content_length = strlen(body);
        
        snprintf(response, sizeof(response),
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/plain\r\n"
            "Content-Length: %d\r\n"
            "Connection: keep-alive\r\n"
            "Server: boost-asio-http/1.0\r\n"
            "\r\n"
            "%s",
            content_length, body);
        
        response_data_ = std::string(response);
        boost::asio::async_write(socket_,
            boost::asio::buffer(response_data_),
            boost::bind(&session::handle_write, shared_from_this(),
                boost::asio::placeholders::error));
    }
    
    void handle_http_request(const char* request) {
        char method[16], path[256], version[16];
        
        if (sscanf(request, "%15s %255s %15s", method, path, version) == 3) {
            char response_body[1024];
            snprintf(response_body, sizeof(response_body),
                "Boost.Asio HTTP Server Response\n"
                "Method: %s\n"
                "Path: %s\n"
                "Version: %s\n"
                "\n"
                "Request received successfully!",
                method, path, version);
            
            send_http_response(response_body);
        } else {
            const char* bad_request = 
                "HTTP/1.1 400 Bad Request\r\n"
                "Content-Type: text/plain\r\n"
                "Content-Length: 11\r\n"
                "Connection: close\r\n"
                "\r\n"
                "Bad Request";
            response_data_ = std::string(bad_request);
            boost::asio::async_write(socket_,
                boost::asio::buffer(response_data_),
                boost::bind(&session::handle_write, shared_from_this(),
                    boost::asio::placeholders::error));
        }
    }
    
    void handle_read(const boost::system::error_code& error, size_t bytes_transferred) {
        if (!error) {
            data_[bytes_transferred] = '\0';
            
            if (strstr(data_, "\r\n\r\n") != NULL) {
                std::cout << "Received HTTP request:\n" << data_ << std::endl;
                handle_http_request(data_);
            } else {
                // Continue reading if we don't have a complete HTTP request
                socket_.async_read_some(boost::asio::buffer(data_, max_length),
                    boost::bind(&session::handle_read, shared_from_this(),
                        boost::asio::placeholders::error,
                        boost::asio::placeholders::bytes_transferred));
            }
        }
    }
    
    void handle_write(const boost::system::error_code& error) {
        if (!error) {
            socket_.async_read_some(boost::asio::buffer(data_, max_length),
                boost::bind(&session::handle_read, shared_from_this(),
                    boost::asio::placeholders::error,
                    boost::asio::placeholders::bytes_transferred));
        }
    }
    
    tcp::socket socket_;
    enum { max_length = 4096 };
    char data_[max_length];
    std::string response_data_;
};

class server {
public:
    server(boost::asio::io_context& io_context, short port)
        : io_context_(io_context),
          acceptor_(io_context, tcp::endpoint(tcp::v4(), port)) {
        start_accept();
    }
    
private:
    void start_accept() {
        auto new_session = std::make_shared<session>(io_context_);
        acceptor_.async_accept(new_session->socket(),
            boost::bind(&server::handle_accept, this, new_session,
                boost::asio::placeholders::error));
    }
    
    void handle_accept(std::shared_ptr<session> new_session,
                      const boost::system::error_code& error) {
        if (!error) {
            new_session->start();
        }
        start_accept();
    }
    
    boost::asio::io_context& io_context_;
    tcp::acceptor acceptor_;
};

int main() {
    try {
        boost::asio::io_context io_context;
        server s(io_context, 8080);
        
        std::cout << "Boost.Asio HTTP server listening on port 8080" << std::endl;
        io_context.run();
    }
    catch (std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
    }

    return 0;
}