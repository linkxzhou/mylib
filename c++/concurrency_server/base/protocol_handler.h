#ifndef PROTOCOL_HANDLER_H
#define PROTOCOL_HANDLER_H

#include "connection_manager.h"
#include <string>
#include <memory>
#include <functional>
#include <unordered_map>
#include <vector>
#include <regex>

// 协议类型枚举
enum class ProtocolType {
    HTTP,
    WEBSOCKET,
    CUSTOM,
    RAW
};

// HTTP方法枚举
enum class HttpMethod {
    GET,
    POST,
    PUT,
    DELETE,
    HEAD,
    OPTIONS,
    PATCH,
    UNKNOWN
};

// HTTP状态码
enum class HttpStatus {
    OK = 200,
    CREATED = 201,
    NO_CONTENT = 204,
    BAD_REQUEST = 400,
    UNAUTHORIZED = 401,
    FORBIDDEN = 403,
    NOT_FOUND = 404,
    METHOD_NOT_ALLOWED = 405,
    INTERNAL_SERVER_ERROR = 500,
    NOT_IMPLEMENTED = 501,
    SERVICE_UNAVAILABLE = 503
};

// HTTP请求结构
struct HttpRequest {
    HttpMethod method;
    std::string path;
    std::string query_string;
    std::unordered_map<std::string, std::string> headers;
    std::unordered_map<std::string, std::string> params;
    std::string body;
    std::string version;  // HTTP/1.1, HTTP/2.0等
    
    // 辅助方法
    std::string get_header(const std::string& name) const;
    std::string get_param(const std::string& name) const;
    bool has_header(const std::string& name) const;
    bool has_param(const std::string& name) const;
};

// HTTP响应结构
struct HttpResponse {
    HttpStatus status;
    std::unordered_map<std::string, std::string> headers;
    std::string body;
    std::string version;
    
    HttpResponse(HttpStatus st = HttpStatus::OK) : status(st), version("HTTP/1.1") {}
    
    // 辅助方法
    void set_header(const std::string& name, const std::string& value);
    void set_content_type(const std::string& type);
    void set_content_length(size_t length);
    std::string to_string() const;
};

// WebSocket帧类型
enum class WebSocketOpCode {
    CONTINUATION = 0x0,
    TEXT = 0x1,
    BINARY = 0x2,
    CLOSE = 0x8,
    PING = 0x9,
    PONG = 0xA
};

// WebSocket帧结构
struct WebSocketFrame {
    bool fin;
    WebSocketOpCode opcode;
    bool masked;
    uint64_t payload_length;
    uint32_t mask_key;
    std::vector<uint8_t> payload;
    
    std::vector<uint8_t> to_bytes() const;
    static WebSocketFrame from_bytes(const uint8_t* data, size_t length);
};

// 协议处理器基类
class ProtocolHandler {
public:
    virtual ~ProtocolHandler() = default;
    
    // 协议检测
    virtual bool can_handle(const char* data, size_t length) = 0;
    
    // 数据处理
    virtual void handle_data(std::shared_ptr<Connection> conn, const char* data, size_t length) = 0;
    
    // 连接建立
    virtual void on_connection_established(std::shared_ptr<Connection> conn) {}
    
    // 连接关闭
    virtual void on_connection_closed(std::shared_ptr<Connection> conn) {}
    
    // 获取协议类型
    virtual ProtocolType get_protocol_type() const = 0;
};

// HTTP协议处理器
class HttpProtocolHandler : public ProtocolHandler {
public:
    using RequestHandler = std::function<void(const HttpRequest&, HttpResponse&, std::shared_ptr<Connection>)>;
    using RouteHandler = std::function<void(const HttpRequest&, HttpResponse&)>;
    
    HttpProtocolHandler();
    ~HttpProtocolHandler() override = default;
    
    bool can_handle(const char* data, size_t length) override;
    void handle_data(std::shared_ptr<Connection> conn, const char* data, size_t length) override;
    ProtocolType get_protocol_type() const override { return ProtocolType::HTTP; }
    
    // 路由注册
    void add_route(HttpMethod method, const std::string& pattern, RouteHandler handler);
    void get(const std::string& pattern, RouteHandler handler);
    void post(const std::string& pattern, RouteHandler handler);
    void put(const std::string& pattern, RouteHandler handler);
    void delete_(const std::string& pattern, RouteHandler handler);
    
    // 中间件支持
    using Middleware = std::function<bool(const HttpRequest&, HttpResponse&)>;
    void add_middleware(Middleware middleware);
    
    // 静态文件服务
    void serve_static(const std::string& url_prefix, const std::string& file_path);
    
    // 设置默认处理器
    void set_default_handler(RequestHandler handler) { default_handler_ = std::move(handler); }
    
private:
    struct Route {
        HttpMethod method;
        std::regex pattern;
        RouteHandler handler;
        std::string original_pattern;
    };
    
    std::vector<Route> routes_;
    std::vector<Middleware> middlewares_;
    RequestHandler default_handler_;
    std::unordered_map<int, std::string> partial_requests_;  // 连接ID -> 部分请求数据
    
    // 内部方法
    HttpRequest parse_request(const std::string& raw_request);
    bool route_request(const HttpRequest& request, HttpResponse& response);
    void send_response(std::shared_ptr<Connection> conn, const HttpResponse& response);
    std::string get_status_text(HttpStatus status);
    HttpMethod parse_method(const std::string& method_str);
};

// WebSocket协议处理器
class WebSocketProtocolHandler : public ProtocolHandler {
public:
    using MessageHandler = std::function<void(std::shared_ptr<Connection>, const std::string&)>;
    using BinaryHandler = std::function<void(std::shared_ptr<Connection>, const std::vector<uint8_t>&)>;
    using CloseHandler = std::function<void(std::shared_ptr<Connection>, uint16_t code, const std::string& reason)>;
    
    WebSocketProtocolHandler();
    ~WebSocketProtocolHandler() override = default;
    
    bool can_handle(const char* data, size_t length) override;
    void handle_data(std::shared_ptr<Connection> conn, const char* data, size_t length) override;
    void on_connection_established(std::shared_ptr<Connection> conn) override;
    ProtocolType get_protocol_type() const override { return ProtocolType::WEBSOCKET; }
    
    // 事件处理器设置
    void set_message_handler(MessageHandler handler) { message_handler_ = std::move(handler); }
    void set_binary_handler(BinaryHandler handler) { binary_handler_ = std::move(handler); }
    void set_close_handler(CloseHandler handler) { close_handler_ = std::move(handler); }
    
    // 发送消息
    void send_text(std::shared_ptr<Connection> conn, const std::string& message);
    void send_binary(std::shared_ptr<Connection> conn, const std::vector<uint8_t>& data);
    void send_ping(std::shared_ptr<Connection> conn, const std::string& data = "");
    void send_pong(std::shared_ptr<Connection> conn, const std::string& data = "");
    void close_connection(std::shared_ptr<Connection> conn, uint16_t code = 1000, const std::string& reason = "");
    
    // 广播消息
    void broadcast_text(const std::string& message);
    void broadcast_binary(const std::vector<uint8_t>& data);
    
private:
    MessageHandler message_handler_;
    BinaryHandler binary_handler_;
    CloseHandler close_handler_;
    
    std::unordered_map<int, std::vector<uint8_t>> frame_buffers_;  // 连接ID -> 帧缓冲区
    std::unordered_map<int, bool> handshake_completed_;  // 连接ID -> 握手状态
    std::vector<std::shared_ptr<Connection>> connections_;  // WebSocket连接列表
    
    // 内部方法
    bool perform_handshake(std::shared_ptr<Connection> conn, const std::string& request);
    void handle_frame(std::shared_ptr<Connection> conn, const WebSocketFrame& frame);
    void send_frame(std::shared_ptr<Connection> conn, const WebSocketFrame& frame);
    std::string generate_accept_key(const std::string& key);
};

// 协议管理器 - 自动检测和分发协议
class ProtocolManager {
public:
    ProtocolManager();
    ~ProtocolManager();
    
    // 注册协议处理器
    void register_handler(std::unique_ptr<ProtocolHandler> handler);
    
    // 处理连接数据
    void handle_connection_data(std::shared_ptr<Connection> conn, const char* data, size_t length);
    
    // 连接事件
    void on_connection_established(std::shared_ptr<Connection> conn);
    void on_connection_closed(std::shared_ptr<Connection> conn);
    
    // 获取特定类型的处理器
    template<typename T>
    T* get_handler() {
        for (auto& handler : handlers_) {
            if (auto* typed_handler = dynamic_cast<T*>(handler.get())) {
                return typed_handler;
            }
        }
        return nullptr;
    }
    
private:
    std::vector<std::unique_ptr<ProtocolHandler>> handlers_;
    std::unordered_map<int, ProtocolHandler*> connection_protocols_;  // 连接ID -> 协议处理器
    
    // 协议检测
    ProtocolHandler* detect_protocol(const char* data, size_t length);
};

// 协议工厂
class ProtocolFactory {
public:
    static std::unique_ptr<HttpProtocolHandler> create_http_handler();
    static std::unique_ptr<WebSocketProtocolHandler> create_websocket_handler();
    static std::unique_ptr<ProtocolManager> create_manager_with_defaults();
};

#endif // PROTOCOL_HANDLER_H