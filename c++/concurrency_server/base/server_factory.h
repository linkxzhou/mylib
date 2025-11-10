#ifndef SERVER_FACTORY_H
#define SERVER_FACTORY_H

#include "enhanced_event_loop.h"
#include "protocol_handler.h"
#include "connection_manager.h"
#include "event_dispatcher.h"
#include <memory>
#include <functional>
#include <string>
#include <chrono>
#include <thread>

// 服务器配置结构
struct ServerConfig {
    // 网络配置
    std::string host = "0.0.0.0";
    int port = 8080;
    size_t max_connections = 10000;
    std::chrono::seconds keepalive_timeout{300};
    
    // 事件循环配置
    EventDispatcherFactory::Type dispatcher_type = EventDispatcherFactory::Type::AUTO;
    size_t worker_threads = std::thread::hardware_concurrency();
    bool use_multi_thread = false;
    
    // 协议配置
    bool enable_http = true;
    bool enable_websocket = true;
    bool enable_custom_protocol = false;
    
    // 性能配置
    size_t connection_pool_size = 100;
    bool enable_statistics = true;
    std::chrono::seconds stats_report_interval{60};
    
    // 安全配置
    bool enable_rate_limiting = false;
    size_t max_requests_per_second = 1000;
    size_t max_request_size = 1024 * 1024;  // 1MB
    
    // 日志配置
    bool enable_logging = true;
    std::string log_level = "INFO";
    std::string log_file = "";
};

// 统一的服务器接口
class UnifiedServer {
public:
    using RequestHandler = std::function<void(const HttpRequest&, HttpResponse&)>;
    using WebSocketMessageHandler = std::function<void(std::shared_ptr<Connection>, const std::string&)>;
    using ConnectionHandler = std::function<void(std::shared_ptr<Connection>)>;
    using ErrorHandler = std::function<void(const std::string&)>;
    
    explicit UnifiedServer(const ServerConfig& config);
    ~UnifiedServer();
    
    // 服务器控制
    bool start();
    void stop();
    bool is_running() const;
    
    // HTTP路由注册
    void get(const std::string& path, RequestHandler handler);
    void post(const std::string& path, RequestHandler handler);
    void put(const std::string& path, RequestHandler handler);
    void delete_(const std::string& path, RequestHandler handler);
    
    // 静态文件服务
    void serve_static(const std::string& url_prefix, const std::string& file_path);
    
    // WebSocket处理
    void on_websocket_message(WebSocketMessageHandler handler);
    void broadcast_websocket(const std::string& message);
    
    // 连接事件处理
    void on_connection_open(ConnectionHandler handler);
    void on_connection_close(ConnectionHandler handler);
    void on_error(ErrorHandler handler);
    
    // 中间件支持
    using Middleware = std::function<bool(const HttpRequest&, HttpResponse&)>;
    void use_middleware(Middleware middleware);
    
    // 统计信息
    struct ServerStats {
        uint64_t total_connections = 0;
        uint64_t active_connections = 0;
        uint64_t total_requests = 0;
        uint64_t total_bytes_read = 0;
        uint64_t total_bytes_written = 0;
        std::chrono::steady_clock::time_point start_time;
        double requests_per_second = 0.0;
    };
    
    ServerStats get_statistics() const;
    void reset_statistics();
    
    // 配置访问
    const ServerConfig& get_config() const { return config_; }
    void update_config(const ServerConfig& config);
    
private:
    ServerConfig config_;
    std::unique_ptr<EnhancedEventLoop> event_loop_;
    std::unique_ptr<MultiThreadEventLoop> multi_thread_loop_;
    std::unique_ptr<ProtocolManager> protocol_manager_;
    
    // 处理器
    HttpProtocolHandler* http_handler_;
    WebSocketProtocolHandler* websocket_handler_;
    
    // 回调函数
    ConnectionHandler connection_open_handler_;
    ConnectionHandler connection_close_handler_;
    ErrorHandler error_handler_;
    
    // 统计信息
    mutable std::mutex stats_mutex_;
    ServerStats stats_;
    std::chrono::steady_clock::time_point last_stats_update_;
    
    // 内部方法
    void setup_protocol_handlers();
    void setup_event_callbacks();
    void update_statistics();
};

// 服务器构建器 - 流式配置接口
class ServerBuilder {
public:
    ServerBuilder();
    
    // 网络配置
    ServerBuilder& listen(int port);
    ServerBuilder& bind(const std::string& host);
    ServerBuilder& max_connections(size_t max_conn);
    ServerBuilder& keepalive_timeout(std::chrono::seconds timeout);
    
    // 事件循环配置
    ServerBuilder& use_dispatcher(EventDispatcherFactory::Type type);
    ServerBuilder& worker_threads(size_t threads);
    ServerBuilder& enable_multi_threading(bool enable = true);
    
    // 协议配置
    ServerBuilder& enable_http(bool enable = true);
    ServerBuilder& enable_websocket(bool enable = true);
    
    // 性能配置
    ServerBuilder& connection_pool_size(size_t size);
    ServerBuilder& enable_statistics(bool enable = true);
    ServerBuilder& stats_report_interval(std::chrono::seconds interval);
    
    // 安全配置
    ServerBuilder& enable_rate_limiting(bool enable = true);
    ServerBuilder& max_requests_per_second(size_t max_rps);
    ServerBuilder& max_request_size(size_t max_size);
    
    // 日志配置
    ServerBuilder& enable_logging(bool enable = true);
    ServerBuilder& log_level(const std::string& level);
    ServerBuilder& log_file(const std::string& file);
    
    // 构建服务器
    std::unique_ptr<UnifiedServer> build();
    
private:
    ServerConfig config_;
};

// 服务器工厂 - 预定义配置
class ServerFactory {
public:
    // 预定义服务器类型
    enum class ServerType {
        SIMPLE_HTTP,        // 简单HTTP服务器
        HIGH_PERFORMANCE,   // 高性能服务器
        WEBSOCKET,         // WebSocket服务器
        FULL_FEATURED,     // 全功能服务器
        MICROSERVICE,      // 微服务
        DEVELOPMENT        // 开发服务器
    };
    
    // 创建预配置服务器
    static std::unique_ptr<UnifiedServer> create(ServerType type, int port = 8080);
    
    // 创建简单HTTP服务器
    static std::unique_ptr<UnifiedServer> create_http_server(int port = 8080);
    
    // 创建高性能服务器
    static std::unique_ptr<UnifiedServer> create_high_performance_server(int port = 8080);
    
    // 创建WebSocket服务器
    static std::unique_ptr<UnifiedServer> create_websocket_server(int port = 8080);
    
    // 创建开发服务器（带调试功能）
    static std::unique_ptr<UnifiedServer> create_development_server(int port = 8080);
    
    // 从配置文件创建
    static std::unique_ptr<UnifiedServer> create_from_config(const std::string& config_file);
    
    // 从JSON配置创建
    static std::unique_ptr<UnifiedServer> create_from_json(const std::string& json_config);
    
private:
    static ServerConfig get_default_config(ServerType type);
};

// 服务器集群管理器
class ServerCluster {
public:
    explicit ServerCluster(size_t num_servers = std::thread::hardware_concurrency());
    ~ServerCluster();
    
    // 添加服务器实例
    void add_server(std::unique_ptr<UnifiedServer> server);
    
    // 集群控制
    bool start_all();
    void stop_all();
    bool is_running() const;
    
    // 负载均衡配置
    enum class LoadBalanceStrategy {
        ROUND_ROBIN,
        LEAST_CONNECTIONS,
        WEIGHTED_ROUND_ROBIN,
        RANDOM
    };
    
    void set_load_balance_strategy(LoadBalanceStrategy strategy);
    
    // 健康检查
    void enable_health_check(std::chrono::seconds interval = std::chrono::seconds(30));
    void disable_health_check();
    
    // 集群统计
    struct ClusterStats {
        size_t total_servers = 0;
        size_t active_servers = 0;
        uint64_t total_connections = 0;
        uint64_t total_requests = 0;
        double average_cpu_usage = 0.0;
        double average_memory_usage = 0.0;
    };
    
    ClusterStats get_cluster_statistics() const;
    
    // 动态扩缩容
    void scale_up(size_t additional_servers);
    void scale_down(size_t servers_to_remove);
    
private:
    std::vector<std::unique_ptr<UnifiedServer>> servers_;
    std::vector<std::unique_ptr<std::thread>> server_threads_;
    LoadBalanceStrategy load_balance_strategy_;
    
    std::atomic<bool> running_;
    std::atomic<bool> health_check_enabled_;
    std::unique_ptr<std::thread> health_check_thread_;
    std::chrono::seconds health_check_interval_;
    
    mutable std::mutex cluster_mutex_;
    
    // 内部方法
    void health_check_loop();
    void restart_failed_servers();
};

// 配置加载器
class ConfigLoader {
public:
    // 从文件加载配置
    static ServerConfig load_from_file(const std::string& filename);
    
    // 从JSON字符串加载配置
    static ServerConfig load_from_json(const std::string& json_str);
    
    // 保存配置到文件
    static bool save_to_file(const ServerConfig& config, const std::string& filename);
    
    // 转换为JSON字符串
    static std::string to_json(const ServerConfig& config);
    
    // 验证配置
    static bool validate_config(const ServerConfig& config, std::string& error_message);
    
    // 合并配置（用于配置继承）
    static ServerConfig merge_configs(const ServerConfig& base, const ServerConfig& override);
};

#endif // SERVER_FACTORY_H