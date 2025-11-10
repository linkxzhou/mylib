#include "protocol_handler.h"
#include <sstream>
#include <algorithm>
#include <fstream>
#include <openssl/sha.h>
#include <openssl/evp.h>
#include <openssl/bio.h>
#include <openssl/buffer.h>
#include <cstring>

// HttpRequest 实现
std::string HttpRequest::get_header(const std::string& name) const {
    auto it = headers.find(name);
    return (it != headers.end()) ? it->second : "";
}

std::string HttpRequest::get_param(const std::string& name) const {
    auto it = params.find(name);
    return (it != params.end()) ? it->second : "";
}

bool HttpRequest::has_header(const std::string& name) const {
    return headers.find(name) != headers.end();
}

bool HttpRequest::has_param(const std::string& name) const {
    return params.find(name) != params.end();
}

// HttpResponse 实现
void HttpResponse::set_header(const std::string& name, const std::string& value) {
    headers[name] = value;
}

void HttpResponse::set_content_type(const std::string& type) {
    set_header("Content-Type", type);
}

void HttpResponse::set_content_length(size_t length) {
    set_header("Content-Length", std::to_string(length));
}

std::string HttpResponse::to_string() const {
    std::ostringstream oss;
    
    // 状态行
    oss << version << " " << static_cast<int>(status) << " ";
    
    // 状态文本
    switch (status) {
        case HttpStatus::OK: oss << "OK"; break;
        case HttpStatus::CREATED: oss << "Created"; break;
        case HttpStatus::NO_CONTENT: oss << "No Content"; break;
        case HttpStatus::BAD_REQUEST: oss << "Bad Request"; break;
        case HttpStatus::UNAUTHORIZED: oss << "Unauthorized"; break;
        case HttpStatus::FORBIDDEN: oss << "Forbidden"; break;
        case HttpStatus::NOT_FOUND: oss << "Not Found"; break;
        case HttpStatus::METHOD_NOT_ALLOWED: oss << "Method Not Allowed"; break;
        case HttpStatus::INTERNAL_SERVER_ERROR: oss << "Internal Server Error"; break;
        case HttpStatus::NOT_IMPLEMENTED: oss << "Not Implemented"; break;
        case HttpStatus::SERVICE_UNAVAILABLE: oss << "Service Unavailable"; break;
        default: oss << "Unknown"; break;
    }
    oss << "\r\n";
    
    // 头部
    for (const auto& header : headers) {
        oss << header.first << ": " << header.second << "\r\n";
    }
    
    // 空行
    oss << "\r\n";
    
    // 主体
    oss << body;
    
    return oss.str();
}

// WebSocketFrame 实现
std::vector<uint8_t> WebSocketFrame::to_bytes() const {
    std::vector<uint8_t> result;
    
    // 第一个字节：FIN + RSV + Opcode
    uint8_t first_byte = (fin ? 0x80 : 0x00) | static_cast<uint8_t>(opcode);
    result.push_back(first_byte);
    
    // 第二个字节：MASK + Payload length
    if (payload_length < 126) {
        uint8_t second_byte = (masked ? 0x80 : 0x00) | static_cast<uint8_t>(payload_length);
        result.push_back(second_byte);
    } else if (payload_length < 65536) {
        uint8_t second_byte = (masked ? 0x80 : 0x00) | 126;
        result.push_back(second_byte);
        result.push_back((payload_length >> 8) & 0xFF);
        result.push_back(payload_length & 0xFF);
    } else {
        uint8_t second_byte = (masked ? 0x80 : 0x00) | 127;
        result.push_back(second_byte);
        for (int i = 7; i >= 0; --i) {
            result.push_back((payload_length >> (i * 8)) & 0xFF);
        }
    }
    
    // Mask key
    if (masked) {
        result.push_back((mask_key >> 24) & 0xFF);
        result.push_back((mask_key >> 16) & 0xFF);
        result.push_back((mask_key >> 8) & 0xFF);
        result.push_back(mask_key & 0xFF);
    }
    
    // Payload
    if (masked) {
        for (size_t i = 0; i < payload.size(); ++i) {
            uint8_t mask_byte = (mask_key >> ((3 - (i % 4)) * 8)) & 0xFF;
            result.push_back(payload[i] ^ mask_byte);
        }
    } else {
        result.insert(result.end(), payload.begin(), payload.end());
    }
    
    return result;
}

WebSocketFrame WebSocketFrame::from_bytes(const uint8_t* data, size_t length) {
    WebSocketFrame frame;
    
    if (length < 2) {
        throw std::runtime_error("Invalid WebSocket frame: too short");
    }
    
    // 解析第一个字节
    frame.fin = (data[0] & 0x80) != 0;
    frame.opcode = static_cast<WebSocketOpCode>(data[0] & 0x0F);
    
    // 解析第二个字节
    frame.masked = (data[1] & 0x80) != 0;
    uint8_t payload_len = data[1] & 0x7F;
    
    size_t header_size = 2;
    
    // 解析payload长度
    if (payload_len < 126) {
        frame.payload_length = payload_len;
    } else if (payload_len == 126) {
        if (length < 4) throw std::runtime_error("Invalid WebSocket frame: incomplete length");
        frame.payload_length = (data[2] << 8) | data[3];
        header_size += 2;
    } else {
        if (length < 10) throw std::runtime_error("Invalid WebSocket frame: incomplete length");
        frame.payload_length = 0;
        for (int i = 0; i < 8; ++i) {
            frame.payload_length = (frame.payload_length << 8) | data[2 + i];
        }
        header_size += 8;
    }
    
    // 解析mask key
    if (frame.masked) {
        if (length < header_size + 4) throw std::runtime_error("Invalid WebSocket frame: incomplete mask");
        frame.mask_key = (data[header_size] << 24) | (data[header_size + 1] << 16) |
                        (data[header_size + 2] << 8) | data[header_size + 3];
        header_size += 4;
    }
    
    // 解析payload
    if (length < header_size + frame.payload_length) {
        throw std::runtime_error("Invalid WebSocket frame: incomplete payload");
    }
    
    frame.payload.resize(frame.payload_length);
    if (frame.masked) {
        for (size_t i = 0; i < frame.payload_length; ++i) {
            uint8_t mask_byte = (frame.mask_key >> ((3 - (i % 4)) * 8)) & 0xFF;
            frame.payload[i] = data[header_size + i] ^ mask_byte;
        }
    } else {
        std::memcpy(frame.payload.data(), data + header_size, frame.payload_length);
    }
    
    return frame;
}

// HttpProtocolHandler 实现
HttpProtocolHandler::HttpProtocolHandler() {
    // 设置默认处理器
    default_handler_ = [](const HttpRequest& req, HttpResponse& resp, std::shared_ptr<Connection> conn) {
        resp.status = HttpStatus::NOT_FOUND;
        resp.set_content_type("text/html");
        resp.body = "<html><body><h1>404 Not Found</h1></body></html>";
        resp.set_content_length(resp.body.length());
    };
}

bool HttpProtocolHandler::can_handle(const char* data, size_t length) {
    if (length < 4) return false;
    
    // 检查是否以HTTP方法开头
    std::string start(data, std::min(length, size_t(16)));
    return start.find("GET ") == 0 || start.find("POST ") == 0 ||
           start.find("PUT ") == 0 || start.find("DELETE ") == 0 ||
           start.find("HEAD ") == 0 || start.find("OPTIONS ") == 0 ||
           start.find("PATCH ") == 0;
}

void HttpProtocolHandler::handle_data(std::shared_ptr<Connection> conn, const char* data, size_t length) {
    int conn_id = conn->get_fd();
    
    // 累积请求数据
    partial_requests_[conn_id].append(data, length);
    
    // 检查是否收到完整的HTTP请求
    std::string& request_data = partial_requests_[conn_id];
    size_t header_end = request_data.find("\r\n\r\n");
    
    if (header_end == std::string::npos) {
        return;  // 还没收到完整的头部
    }
    
    // 检查Content-Length
    size_t content_length = 0;
    size_t content_pos = request_data.find("Content-Length:");
    if (content_pos != std::string::npos) {
        size_t value_start = request_data.find(":", content_pos) + 1;
        size_t value_end = request_data.find("\r\n", value_start);
        std::string length_str = request_data.substr(value_start, value_end - value_start);
        // 去除空格
        length_str.erase(0, length_str.find_first_not_of(" \t"));
        length_str.erase(length_str.find_last_not_of(" \t") + 1);
        content_length = std::stoul(length_str);
    }
    
    size_t total_expected = header_end + 4 + content_length;
    if (request_data.length() < total_expected) {
        return;  // 还没收到完整的请求体
    }
    
    // 解析并处理请求
    try {
        HttpRequest request = parse_request(request_data);
        HttpResponse response;
        
        // 执行中间件
        bool continue_processing = true;
        for (const auto& middleware : middlewares_) {
            if (!middleware(request, response)) {
                continue_processing = false;
                break;
            }
        }
        
        if (continue_processing) {
            // 路由请求
            if (!route_request(request, response)) {
                // 使用默认处理器
                default_handler_(request, response, conn);
            }
        }
        
        // 发送响应
        send_response(conn, response);
        
    } catch (const std::exception& e) {
        // 发送500错误
        HttpResponse error_response(HttpStatus::INTERNAL_SERVER_ERROR);
        error_response.set_content_type("text/html");
        error_response.body = "<html><body><h1>500 Internal Server Error</h1></body></html>";
        error_response.set_content_length(error_response.body.length());
        send_response(conn, error_response);
    }
    
    // 清理请求数据
    partial_requests_.erase(conn_id);
}

void HttpProtocolHandler::add_route(HttpMethod method, const std::string& pattern, RouteHandler handler) {
    Route route;
    route.method = method;
    route.original_pattern = pattern;
    
    // 将路径模式转换为正则表达式
    std::string regex_pattern = pattern;
    // 简单的参数替换：/user/:id -> /user/([^/]+)
    size_t pos = 0;
    while ((pos = regex_pattern.find(":", pos)) != std::string::npos) {
        size_t end = regex_pattern.find("/", pos);
        if (end == std::string::npos) end = regex_pattern.length();
        regex_pattern.replace(pos, end - pos, "([^/]+)");
        pos += 7;  // 长度 "([^/]+)"
    }
    
    route.pattern = std::regex(regex_pattern);
    route.handler = std::move(handler);
    
    routes_.push_back(std::move(route));
}

void HttpProtocolHandler::get(const std::string& pattern, RouteHandler handler) {
    add_route(HttpMethod::GET, pattern, std::move(handler));
}

void HttpProtocolHandler::post(const std::string& pattern, RouteHandler handler) {
    add_route(HttpMethod::POST, pattern, std::move(handler));
}

void HttpProtocolHandler::put(const std::string& pattern, RouteHandler handler) {
    add_route(HttpMethod::PUT, pattern, std::move(handler));
}

void HttpProtocolHandler::delete_(const std::string& pattern, RouteHandler handler) {
    add_route(HttpMethod::DELETE, pattern, std::move(handler));
}

void HttpProtocolHandler::add_middleware(Middleware middleware) {
    middlewares_.push_back(std::move(middleware));
}

void HttpProtocolHandler::serve_static(const std::string& url_prefix, const std::string& file_path) {
    add_route(HttpMethod::GET, url_prefix + "/(.*)", [file_path](const HttpRequest& req, HttpResponse& resp) {
        // 简单的静态文件服务实现
        std::string file_name = req.path.substr(req.path.find_last_of('/') + 1);
        std::string full_path = file_path + "/" + file_name;
        
        std::ifstream file(full_path, std::ios::binary);
        if (file.is_open()) {
            std::string content((std::istreambuf_iterator<char>(file)),
                              std::istreambuf_iterator<char>());
            resp.body = content;
            resp.set_content_length(content.length());
            
            // 设置Content-Type
            if (file_name.ends_with(".html")) {
                resp.set_content_type("text/html");
            } else if (file_name.ends_with(".css")) {
                resp.set_content_type("text/css");
            } else if (file_name.ends_with(".js")) {
                resp.set_content_type("application/javascript");
            } else {
                resp.set_content_type("application/octet-stream");
            }
        } else {
            resp.status = HttpStatus::NOT_FOUND;
            resp.set_content_type("text/html");
            resp.body = "<html><body><h1>404 File Not Found</h1></body></html>";
            resp.set_content_length(resp.body.length());
        }
    });
}

HttpRequest HttpProtocolHandler::parse_request(const std::string& raw_request) {
    HttpRequest request;
    std::istringstream iss(raw_request);
    std::string line;
    
    // 解析请求行
    if (std::getline(iss, line)) {
        // 移除\r
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        
        std::istringstream line_stream(line);
        std::string method_str, url, version;
        line_stream >> method_str >> url >> version;
        
        request.method = parse_method(method_str);
        request.version = version;
        
        // 解析URL和查询字符串
        size_t query_pos = url.find('?');
        if (query_pos != std::string::npos) {
            request.path = url.substr(0, query_pos);
            request.query_string = url.substr(query_pos + 1);
            
            // 解析查询参数
            std::istringstream query_stream(request.query_string);
            std::string param;
            while (std::getline(query_stream, param, '&')) {
                size_t eq_pos = param.find('=');
                if (eq_pos != std::string::npos) {
                    std::string key = param.substr(0, eq_pos);
                    std::string value = param.substr(eq_pos + 1);
                    request.params[key] = value;
                }
            }
        } else {
            request.path = url;
        }
    }
    
    // 解析头部
    while (std::getline(iss, line) && line != "\r") {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        
        size_t colon_pos = line.find(':');
        if (colon_pos != std::string::npos) {
            std::string name = line.substr(0, colon_pos);
            std::string value = line.substr(colon_pos + 1);
            
            // 去除空格
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);
            
            request.headers[name] = value;
        }
    }
    
    // 解析请求体
    std::string body_line;
    while (std::getline(iss, body_line)) {
        request.body += body_line;
        if (iss.peek() != EOF) {
            request.body += "\n";
        }
    }
    
    return request;
}

bool HttpProtocolHandler::route_request(const HttpRequest& request, HttpResponse& response) {
    for (const auto& route : routes_) {
        if (route.method == request.method) {
            std::smatch matches;
            if (std::regex_match(request.path, matches, route.pattern)) {
                // 提取路径参数
                HttpRequest modified_request = request;
                for (size_t i = 1; i < matches.size(); ++i) {
                    modified_request.params["param" + std::to_string(i)] = matches[i].str();
                }
                
                route.handler(modified_request, response);
                return true;
            }
        }
    }
    return false;
}

void HttpProtocolHandler::send_response(std::shared_ptr<Connection> conn, const HttpResponse& response) {
    std::string response_str = response.to_string();
    conn->write_data(response_str.c_str(), response_str.length());
}

HttpMethod HttpProtocolHandler::parse_method(const std::string& method_str) {
    if (method_str == "GET") return HttpMethod::GET;
    if (method_str == "POST") return HttpMethod::POST;
    if (method_str == "PUT") return HttpMethod::PUT;
    if (method_str == "DELETE") return HttpMethod::DELETE;
    if (method_str == "HEAD") return HttpMethod::HEAD;
    if (method_str == "OPTIONS") return HttpMethod::OPTIONS;
    if (method_str == "PATCH") return HttpMethod::PATCH;
    return HttpMethod::UNKNOWN;
}

// WebSocketProtocolHandler 实现
WebSocketProtocolHandler::WebSocketProtocolHandler() {
}

bool WebSocketProtocolHandler::can_handle(const char* data, size_t length) {
    std::string request(data, std::min(length, size_t(1024)));
    return request.find("Upgrade: websocket") != std::string::npos ||
           request.find("Sec-WebSocket-Key:") != std::string::npos;
}

void WebSocketProtocolHandler::handle_data(std::shared_ptr<Connection> conn, const char* data, size_t length) {
    int conn_id = conn->get_fd();
    
    // 检查是否已完成握手
    if (handshake_completed_.find(conn_id) == handshake_completed_.end() ||
        !handshake_completed_[conn_id]) {
        
        // 执行WebSocket握手
        std::string request(data, length);
        if (perform_handshake(conn, request)) {
            handshake_completed_[conn_id] = true;
            connections_.push_back(conn);
        }
        return;
    }
    
    // 处理WebSocket帧
    auto& buffer = frame_buffers_[conn_id];
    buffer.insert(buffer.end(), data, data + length);
    
    // 尝试解析帧
    while (buffer.size() >= 2) {
        try {
            WebSocketFrame frame = WebSocketFrame::from_bytes(buffer.data(), buffer.size());
            
            // 计算帧的总大小
            size_t frame_size = 2;  // 基本头部
            if (frame.payload_length >= 126) {
                frame_size += (frame.payload_length >= 65536) ? 8 : 2;
            }
            if (frame.masked) {
                frame_size += 4;
            }
            frame_size += frame.payload_length;
            
            if (buffer.size() >= frame_size) {
                handle_frame(conn, frame);
                buffer.erase(buffer.begin(), buffer.begin() + frame_size);
            } else {
                break;  // 帧不完整
            }
        } catch (const std::exception& e) {
            // 帧解析错误，关闭连接
            close_connection(conn, 1002, "Protocol error");
            break;
        }
    }
}

void WebSocketProtocolHandler::on_connection_established(std::shared_ptr<Connection> conn) {
    // WebSocket连接需要握手，这里不做处理
}

void WebSocketProtocolHandler::send_text(std::shared_ptr<Connection> conn, const std::string& message) {
    WebSocketFrame frame;
    frame.fin = true;
    frame.opcode = WebSocketOpCode::TEXT;
    frame.masked = false;
    frame.payload_length = message.length();
    frame.payload.assign(message.begin(), message.end());
    
    send_frame(conn, frame);
}

void WebSocketProtocolHandler::send_binary(std::shared_ptr<Connection> conn, const std::vector<uint8_t>& data) {
    WebSocketFrame frame;
    frame.fin = true;
    frame.opcode = WebSocketOpCode::BINARY;
    frame.masked = false;
    frame.payload_length = data.size();
    frame.payload = data;
    
    send_frame(conn, frame);
}

void WebSocketProtocolHandler::send_ping(std::shared_ptr<Connection> conn, const std::string& data) {
    WebSocketFrame frame;
    frame.fin = true;
    frame.opcode = WebSocketOpCode::PING;
    frame.masked = false;
    frame.payload_length = data.length();
    frame.payload.assign(data.begin(), data.end());
    
    send_frame(conn, frame);
}

void WebSocketProtocolHandler::send_pong(std::shared_ptr<Connection> conn, const std::string& data) {
    WebSocketFrame frame;
    frame.fin = true;
    frame.opcode = WebSocketOpCode::PONG;
    frame.masked = false;
    frame.payload_length = data.length();
    frame.payload.assign(data.begin(), data.end());
    
    send_frame(conn, frame);
}

void WebSocketProtocolHandler::close_connection(std::shared_ptr<Connection> conn, uint16_t code, const std::string& reason) {
    WebSocketFrame frame;
    frame.fin = true;
    frame.opcode = WebSocketOpCode::CLOSE;
    frame.masked = false;
    
    // 构造关闭帧的payload
    std::vector<uint8_t> payload;
    payload.push_back((code >> 8) & 0xFF);
    payload.push_back(code & 0xFF);
    payload.insert(payload.end(), reason.begin(), reason.end());
    
    frame.payload_length = payload.size();
    frame.payload = payload;
    
    send_frame(conn, frame);
    
    // 从连接列表中移除
    connections_.erase(
        std::remove(connections_.begin(), connections_.end(), conn),
        connections_.end());
}

void WebSocketProtocolHandler::broadcast_text(const std::string& message) {
    for (auto& conn : connections_) {
        if (!conn->is_closed()) {
            send_text(conn, message);
        }
    }
}

void WebSocketProtocolHandler::broadcast_binary(const std::vector<uint8_t>& data) {
    for (auto& conn : connections_) {
        if (!conn->is_closed()) {
            send_binary(conn, data);
        }
    }
}

bool WebSocketProtocolHandler::perform_handshake(std::shared_ptr<Connection> conn, const std::string& request) {
    // 解析WebSocket握手请求
    size_t key_pos = request.find("Sec-WebSocket-Key:");
    if (key_pos == std::string::npos) {
        return false;
    }
    
    size_t key_start = request.find(":", key_pos) + 1;
    size_t key_end = request.find("\r\n", key_start);
    std::string key = request.substr(key_start, key_end - key_start);
    
    // 去除空格
    key.erase(0, key.find_first_not_of(" \t"));
    key.erase(key.find_last_not_of(" \t") + 1);
    
    // 生成Accept key
    std::string accept_key = generate_accept_key(key);
    
    // 构造响应
    std::ostringstream response;
    response << "HTTP/1.1 101 Switching Protocols\r\n";
    response << "Upgrade: websocket\r\n";
    response << "Connection: Upgrade\r\n";
    response << "Sec-WebSocket-Accept: " << accept_key << "\r\n";
    response << "\r\n";
    
    std::string response_str = response.str();
    conn->write_data(response_str.c_str(), response_str.length());
    
    return true;
}

void WebSocketProtocolHandler::handle_frame(std::shared_ptr<Connection> conn, const WebSocketFrame& frame) {
    switch (frame.opcode) {
        case WebSocketOpCode::TEXT:
            if (message_handler_) {
                std::string message(frame.payload.begin(), frame.payload.end());
                message_handler_(conn, message);
            }
            break;
            
        case WebSocketOpCode::BINARY:
            if (binary_handler_) {
                binary_handler_(conn, frame.payload);
            }
            break;
            
        case WebSocketOpCode::CLOSE:
            {
                uint16_t code = 1000;
                std::string reason;
                
                if (frame.payload.size() >= 2) {
                    code = (frame.payload[0] << 8) | frame.payload[1];
                    if (frame.payload.size() > 2) {
                        reason = std::string(frame.payload.begin() + 2, frame.payload.end());
                    }
                }
                
                if (close_handler_) {
                    close_handler_(conn, code, reason);
                }
                
                close_connection(conn, code, reason);
            }
            break;
            
        case WebSocketOpCode::PING:
            // 自动回复PONG
            {
                std::string data(frame.payload.begin(), frame.payload.end());
                send_pong(conn, data);
            }
            break;
            
        case WebSocketOpCode::PONG:
            // PONG帧通常不需要特殊处理
            break;
            
        default:
            // 未知操作码，关闭连接
            close_connection(conn, 1002, "Unknown opcode");
            break;
    }
}

void WebSocketProtocolHandler::send_frame(std::shared_ptr<Connection> conn, const WebSocketFrame& frame) {
    std::vector<uint8_t> frame_data = frame.to_bytes();
    conn->write_data(reinterpret_cast<const char*>(frame_data.data()), frame_data.size());
}

std::string WebSocketProtocolHandler::generate_accept_key(const std::string& key) {
    // WebSocket规范要求的GUID
    const std::string websocket_guid = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    std::string combined = key + websocket_guid;
    
    // 计算SHA-1哈希
    unsigned char hash[SHA_DIGEST_LENGTH];
    SHA1(reinterpret_cast<const unsigned char*>(combined.c_str()), combined.length(), hash);
    
    // Base64编码
    BIO* bio = BIO_new(BIO_s_mem());
    BIO* b64 = BIO_new(BIO_f_base64());
    BIO_set_flags(b64, BIO_FLAGS_BASE64_NO_NL);
    bio = BIO_push(b64, bio);
    
    BIO_write(bio, hash, SHA_DIGEST_LENGTH);
    BIO_flush(bio);
    
    BUF_MEM* buffer_ptr;
    BIO_get_mem_ptr(bio, &buffer_ptr);
    
    std::string result(buffer_ptr->data, buffer_ptr->length);
    BIO_free_all(bio);
    
    return result;
}

// ProtocolManager 实现
ProtocolManager::ProtocolManager() {
}

ProtocolManager::~ProtocolManager() {
}

void ProtocolManager::register_handler(std::unique_ptr<ProtocolHandler> handler) {
    handlers_.push_back(std::move(handler));
}

void ProtocolManager::handle_connection_data(std::shared_ptr<Connection> conn, const char* data, size_t length) {
    int conn_id = conn->get_fd();
    
    // 检查是否已经确定了协议
    auto it = connection_protocols_.find(conn_id);
    if (it != connection_protocols_.end()) {
        it->second->handle_data(conn, data, length);
        return;
    }
    
    // 检测协议
    ProtocolHandler* handler = detect_protocol(data, length);
    if (handler) {
        connection_protocols_[conn_id] = handler;
        handler->handle_data(conn, data, length);
    }
}

void ProtocolManager::on_connection_established(std::shared_ptr<Connection> conn) {
    for (auto& handler : handlers_) {
        handler->on_connection_established(conn);
    }
}

void ProtocolManager::on_connection_closed(std::shared_ptr<Connection> conn) {
    int conn_id = conn->get_fd();
    
    auto it = connection_protocols_.find(conn_id);
    if (it != connection_protocols_.end()) {
        it->second->on_connection_closed(conn);
        connection_protocols_.erase(it);
    }
    
    for (auto& handler : handlers_) {
        handler->on_connection_closed(conn);
    }
}

ProtocolHandler* ProtocolManager::detect_protocol(const char* data, size_t length) {
    for (auto& handler : handlers_) {
        if (handler->can_handle(data, length)) {
            return handler.get();
        }
    }
    return nullptr;
}

// ProtocolFactory 实现
std::unique_ptr<HttpProtocolHandler> ProtocolFactory::create_http_handler() {
    return std::make_unique<HttpProtocolHandler>();
}

std::unique_ptr<WebSocketProtocolHandler> ProtocolFactory::create_websocket_handler() {
    return std::make_unique<WebSocketProtocolHandler>();
}

std::unique_ptr<ProtocolManager> ProtocolFactory::create_manager_with_defaults() {
    auto manager = std::make_unique<ProtocolManager>();
    
    // 注册默认协议处理器
    manager->register_handler(create_websocket_handler());
    manager->register_handler(create_http_handler());
    
    return manager;
}