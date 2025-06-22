#include <seastar/core/app-template.hh>
#include <seastar/core/reactor.hh>
#include <seastar/core/future.hh>
#include <seastar/net/api.hh>
#include <seastar/core/sleep.hh>
#include <seastar/core/sstring.hh>
#include <iostream>
#include <sstream>
#include <string>

using namespace seastar;
using namespace net;
using namespace std::chrono_literals;

class http_server {
private:
    server_socket _listener;
    
public:
    future<> listen(socket_address addr) {
        listen_options lo;
        lo.reuse_address = true;
        _listener = seastar::listen(addr, lo);
        
        std::cout << "seastar HTTP/1.1 server listening on port 8080" << std::endl;
        
        // Accept connections in a loop
        return keep_doing([this] {
            return _listener.accept().then(
                [this] (accept_result ar) {
                    auto conn = std::move(ar.connection);
                    auto addr = std::move(ar.remote_address);
                    
                    // Handle connection asynchronously
                    (void)handle_connection(std::move(conn), addr).handle_exception(
                        [addr] (std::exception_ptr ep) {
                            std::cerr << "Connection error from " << addr << ": " << ep << std::endl;
                        });
                });
        });
    }
    
private:
    sstring parse_http_request(const sstring& request) {
        std::istringstream iss(request.c_str());
        std::string method, path, version;
        iss >> method >> path >> version;
        
        // Extract headers and body
        std::string line;
        std::string headers;
        std::string body;
        bool in_body = false;
        
        while (std::getline(iss, line)) {
            if (line == "\r" || line.empty()) {
                in_body = true;
                continue;
            }
            if (in_body) {
                body += line + "\n";
            } else {
                headers += line + "\n";
            }
        }
        
        return sstring("Method: " + method + "\nPath: " + path + "\nVersion: " + version + 
                      "\nHeaders:\n" + headers + "\nBody:\n" + body);
    }
    
    sstring create_http_response(const sstring& content) {
        std::ostringstream response;
        response << "HTTP/1.1 200 OK\r\n";
        response << "Content-Type: text/plain\r\n";
        response << "Content-Length: " << content.size() << "\r\n";
        response << "Connection: keep-alive\r\n";
        response << "Server: Seastar-HTTP-Server/1.0\r\n";
        response << "\r\n";
        response << content.c_str();
        return sstring(response.str());
    }
    
    sstring create_bad_request_response() {
        return sstring("HTTP/1.1 400 Bad Request\r\n"
                      "Content-Type: text/plain\r\n"
                      "Content-Length: 11\r\n"
                      "Connection: close\r\n"
                      "\r\n"
                      "Bad Request");
    }
    
    bool is_complete_http_request(const sstring& request) {
        // Check if we have a complete HTTP request
        size_t header_end = request.find("\r\n\r\n");
        if (header_end == sstring::npos) {
            return false;
        }
        
        // Check for Content-Length header
        size_t content_length_pos = request.find("Content-Length:");
        if (content_length_pos != sstring::npos) {
            size_t value_start = request.find(":", content_length_pos) + 1;
            size_t value_end = request.find("\r\n", value_start);
            std::string length_str = request.substr(value_start, value_end - value_start).c_str();
            
            // Remove leading/trailing whitespace
            length_str.erase(0, length_str.find_first_not_of(" \t"));
            length_str.erase(length_str.find_last_not_of(" \t") + 1);
            
            int content_length = std::stoi(length_str);
            size_t body_start = header_end + 4;
            return request.size() >= body_start + content_length;
        }
        
        return true; // No body expected
    }
    
    future<> handle_connection(connected_socket conn, socket_address addr) {
        auto in = conn.input();
        auto out = conn.output();
        
        return do_with(std::move(in), std::move(out), sstring(),
            [this, addr] (auto& in, auto& out, auto& buffer) {
                return repeat([this, &in, &out, &buffer, addr] {
                    return in.read().then([this, &out, &buffer, addr] (temporary_buffer<char> buf) {
                        if (buf.empty()) {
                            return make_ready_future<stop_iteration>(stop_iteration::yes);
                        }
                        
                        // Append to buffer
                        buffer += sstring(buf.get(), buf.size());
                        
                        // Check if we have a complete HTTP request
                        if (!is_complete_http_request(buffer)) {
                            return make_ready_future<stop_iteration>(stop_iteration::no);
                        }
                        
                        sstring response;
                        if (buffer.find("HTTP/") != sstring::npos) {
                            // Parse HTTP request and create response
                            sstring parsed_request = parse_http_request(buffer);
                            sstring response_content = sstring("Echo HTTP Request:\n") + parsed_request;
                            response = create_http_response(response_content);
                        } else {
                            // Send HTTP 400 Bad Request for invalid requests
                            response = create_bad_request_response();
                        }
                        
                        // Clear buffer for next request
                        buffer = sstring();
                        
                        // Send response
                        return out.write(response.c_str(), response.size()).then([&out] {
                            return out.flush();
                        }).then([] {
                            return make_ready_future<stop_iteration>(stop_iteration::no);
                        });
                    });
                });
            });
    }
};

int main(int argc, char** argv) {
    app_template app;
    
    return app.run(argc, argv, [&] {
        auto server = new http_server();
        
        return server->listen(socket_address{ipv4_addr{"127.0.0.1", 8080}}).handle_exception(
            [] (std::exception_ptr ep) {
                std::cerr << "Server failed: " << ep << std::endl;
                return make_exception_future<>(ep);
            });
    });
}