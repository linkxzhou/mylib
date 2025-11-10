#include "server_factory.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>

// UnifiedServer 实现
UnifiedServer::UnifiedServer(const ServerConfig& config)
    : config_(config), http_handler_(nullptr), websocket_handler_(nullptr) {
    
    stats_.start_time = std::chrono::steady_clock::now();
    last_stats_update_ = stats_.start_time;
    
    // 创建协议管理器
    protocol_manager_ = std::make_unique<ProtocolManager>();
    setup_protocol_handlers();
    
    // 创建事件循环
    if (config_.use_multi_thread && config_.worker_threads > 1) {
        multi_thread_loop_ = std::make_unique<MultiThreadEventLoop>(config_.worker_threads);
        setup_event_callbacks();
    } else {
        event_loop_ = std::make_unique<EnhancedEventLoop>(config_.dispatcher_type, config_.max_connections);
        event_loop_->set_keepalive_timeout(config_.keepalive_timeout);
        setup_event_callbacks();
    }
}

UnifiedServer::~UnifiedServer() {
    stop();
}

bool UnifiedServer::start() {
    if (is_running()) {
        return false;
    }
    
    if (config_.use_multi_thread && multi_thread_loop_) {
        if (!multi_thread_loop_->bind_and_listen(config_.port, config_.host)) {
            return false;
        }
        
        std::thread([this]() {
            multi_thread_loop_->run();
        }).detach();
        
    } else if (event_loop_) {
        if (!event_loop_->bind_and_listen(config_.port, config_.host)) {
            return false;
        }
        
        event_loop_->run_in_thread();
    }
    
    std::cout << "Unified Server started on " << config_.host << ":" << config_.port << std::endl;
    return true;
}

void UnifiedServer::stop() {
    if (event_loop_) {
        event_loop_->stop();
    }
    if (multi_thread_loop_) {
        multi_thread_loop_->stop();
    }
}

bool UnifiedServer::is_running() const {
    if (event_loop_) {
        return event_loop_->is_running();
    }
    if (multi_thread_loop_) {
        // 多线程版本需要检查状态
        return true;  // 简化实现
    }
    return false;
}

void UnifiedServer::get(const std::string& path, RequestHandler handler) {
    if (http_handler_) {
        http_handler_->get(path, [handler](const HttpRequest& req, HttpResponse& resp) {
            handler(req, resp);
        });
    }
}

void UnifiedServer::post(const std::string& path, RequestHandler handler) {
    if (http_handler_) {
        http_handler_->post(path, [handler](const HttpRequest& req, HttpResponse& resp) {
            handler(req, resp);
        });
    }
}

void UnifiedServer::put(const std::string& path, RequestHandler handler) {
    if (http_handler_) {
        http_handler_->put(path, [handler](const HttpRequest& req, HttpResponse& resp) {
            handler(req, resp);
        });
    }
}

void UnifiedServer::delete_(const std::string& path, RequestHandler handler) {
    if (http_handler_) {
        http_handler_->delete_(path, [handler](const HttpRequest& req, HttpResponse& resp) {
            handler(req, resp);
        });
    }
}

void UnifiedServer::serve_static(const std::string& url_prefix, const std::string& file_path) {
    if (http_handler_) {
        http_handler_->serve_static(url_prefix, file_path);
    }
}

void UnifiedServer::on_websocket_message(WebSocketMessageHandler handler) {
    if (websocket_handler_) {
        websocket_handler_->set_message_handler(handler);
    }
}

void UnifiedServer::broadcast_websocket(const std::string& message) {
    if (websocket_handler_) {
        websocket_handler_->broadcast_text(message);
    }
}

void UnifiedServer::on_connection_open(ConnectionHandler handler) {
    connection_open_handler_ = std::move(handler);
}

void UnifiedServer::on_connection_close(ConnectionHandler handler) {
    connection_close_handler_ = std::move(handler);
}

void UnifiedServer::on_error(ErrorHandler handler) {
    error_handler_ = std::move(handler);
}

void UnifiedServer::use_middleware(Middleware middleware) {
    if (http_handler_) {
        http_handler_->add_middleware(middleware);
    }
}

UnifiedServer::ServerStats UnifiedServer::get_statistics() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    ServerStats current_stats = stats_;
    
    // 更新实时统计
    if (event_loop_) {
        const auto& loop_stats = event_loop_->get_statistics();
        current_stats.active_connections = loop_stats.active_connections;
        current_stats.total_connections = loop_stats.total_connections;
        current_stats.total_requests = loop_stats.total_requests;
        current_stats.total_bytes_read = loop_stats.bytes_read;
        current_stats.total_bytes_written = loop_stats.bytes_written;
    } else if (multi_thread_loop_) {
        const auto& loop_stats = multi_thread_loop_->get_total_statistics();
        current_stats.active_connections = loop_stats.active_connections;
        current_stats.total_connections = loop_stats.total_connections;
        current_stats.total_requests = loop_stats.total_requests;
        current_stats.total_bytes_read = loop_stats.bytes_read;
        current_stats.total_bytes_written = loop_stats.bytes_written;
    }
    
    // 计算RPS
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - current_stats.start_time);
    if (elapsed.count() > 0) {
        current_stats.requests_per_second = static_cast<double>(current_stats.total_requests) / elapsed.count();
    }
    
    return current_stats;
}

void UnifiedServer::reset_statistics() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_ = ServerStats{};
    stats_.start_time = std::chrono::steady_clock::now();
    
    if (event_loop_) {
        event_loop_->reset_statistics();
    }
}

void UnifiedServer::update_config(const ServerConfig& config) {
    // 注意：某些配置更改需要重启服务器
    config_ = config;
    
    if (event_loop_) {
        event_loop_->set_keepalive_timeout(config_.keepalive_timeout);
        event_loop_->set_max_connections(config_.max_connections);
    }
}

void UnifiedServer::setup_protocol_handlers() {
    if (config_.enable_http) {
        auto http_handler = std::make_unique<HttpProtocolHandler>();
        http_handler_ = http_handler.get();
        protocol_manager_->register_handler(std::move(http_handler));
    }
    
    if (config_.enable_websocket) {
        auto websocket_handler = std::make_unique<WebSocketProtocolHandler>();
        websocket_handler_ = websocket_handler.get();
        protocol_manager_->register_handler(std::move(websocket_handler));
    }
}

void UnifiedServer::setup_event_callbacks() {
    auto message_callback = [this](std::shared_ptr<Connection> conn, const char* data, size_t size) {
        protocol_manager_->handle_connection_data(conn, data, size);
        update_statistics();
    };
    
    auto close_callback = [this](std::shared_ptr<Connection> conn) {
        protocol_manager_->on_connection_closed(conn);
        if (connection_close_handler_) {
            connection_close_handler_(conn);
        }
    };
    
    auto error_callback = [this](std::shared_ptr<Connection> conn, const std::string& error) {
        if (error_handler_) {
            error_handler_(error);
        }
    };
    
    if (event_loop_) {
        event_loop_->set_message_callback(message_callback);
        event_loop_->set_close_callback(close_callback);
        event_loop_->set_error_callback(error_callback);
        
        event_loop_->set_accept_callback([this](int client_fd, const struct sockaddr_in& addr) {
            if (connection_open_handler_) {
                auto conn = event_loop_->get_connection(client_fd);
                if (conn) {
                    connection_open_handler_(conn);
                }
            }
        });
    }
    
    if (multi_thread_loop_) {
        multi_thread_loop_->set_message_callback(message_callback);
        multi_thread_loop_->set_close_callback(close_callback);
        multi_thread_loop_->set_error_callback(error_callback);
    }
}

void UnifiedServer::update_statistics() {
    auto now = std::chrono::steady_clock::now();
    if (now - last_stats_update_ >= std::chrono::seconds(1)) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        last_stats_update_ = now;
        // 这里可以添加更多统计更新逻辑
    }
}

// ServerBuilder 实现
ServerBuilder::ServerBuilder() {
    // 设置默认配置
}

ServerBuilder& ServerBuilder::listen(int port) {
    config_.port = port;
    return *this;
}

ServerBuilder& ServerBuilder::bind(const std::string& host) {
    config_.host = host;
    return *this;
}

ServerBuilder& ServerBuilder::max_connections(size_t max_conn) {
    config_.max_connections = max_conn;
    return *this;
}

ServerBuilder& ServerBuilder::keepalive_timeout(std::chrono::seconds timeout) {
    config_.keepalive_timeout = timeout;
    return *this;
}

ServerBuilder& ServerBuilder::use_dispatcher(EventDispatcherFactory::Type type) {
    config_.dispatcher_type = type;
    return *this;
}

ServerBuilder& ServerBuilder::worker_threads(size_t threads) {
    config_.worker_threads = threads;
    return *this;
}

ServerBuilder& ServerBuilder::enable_multi_threading(bool enable) {
    config_.use_multi_thread = enable;
    return *this;
}

ServerBuilder& ServerBuilder::enable_http(bool enable) {
    config_.enable_http = enable;
    return *this;
}

ServerBuilder& ServerBuilder::enable_websocket(bool enable) {
    config_.enable_websocket = enable;
    return *this;
}

ServerBuilder& ServerBuilder::connection_pool_size(size_t size) {
    config_.connection_pool_size = size;
    return *this;
}

ServerBuilder& ServerBuilder::enable_statistics(bool enable) {
    config_.enable_statistics = enable;
    return *this;
}

ServerBuilder& ServerBuilder::stats_report_interval(std::chrono::seconds interval) {
    config_.stats_report_interval = interval;
    return *this;
}

ServerBuilder& ServerBuilder::enable_rate_limiting(bool enable) {
    config_.enable_rate_limiting = enable;
    return *this;
}

ServerBuilder& ServerBuilder::max_requests_per_second(size_t max_rps) {
    config_.max_requests_per_second = max_rps;
    return *this;
}

ServerBuilder& ServerBuilder::max_request_size(size_t max_size) {
    config_.max_request_size = max_size;
    return *this;
}

ServerBuilder& ServerBuilder::enable_logging(bool enable) {
    config_.enable_logging = enable;
    return *this;
}

ServerBuilder& ServerBuilder::log_level(const std::string& level) {
    config_.log_level = level;
    return *this;
}

ServerBuilder& ServerBuilder::log_file(const std::string& file) {
    config_.log_file = file;
    return *this;
}

std::unique_ptr<UnifiedServer> ServerBuilder::build() {
    return std::make_unique<UnifiedServer>(config_);
}

// ServerFactory 实现
std::unique_ptr<UnifiedServer> ServerFactory::create(ServerType type, int port) {
    ServerConfig config = get_default_config(type);
    config.port = port;
    return std::make_unique<UnifiedServer>(config);
}

std::unique_ptr<UnifiedServer> ServerFactory::create_http_server(int port) {
    return create(ServerType::SIMPLE_HTTP, port);
}

std::unique_ptr<UnifiedServer> ServerFactory::create_high_performance_server(int port) {
    return create(ServerType::HIGH_PERFORMANCE, port);
}

std::unique_ptr<UnifiedServer> ServerFactory::create_websocket_server(int port) {
    return create(ServerType::WEBSOCKET, port);
}

std::unique_ptr<UnifiedServer> ServerFactory::create_development_server(int port) {
    return create(ServerType::DEVELOPMENT, port);
}

std::unique_ptr<UnifiedServer> ServerFactory::create_from_config(const std::string& config_file) {
    ServerConfig config = ConfigLoader::load_from_file(config_file);
    return std::make_unique<UnifiedServer>(config);
}

std::unique_ptr<UnifiedServer> ServerFactory::create_from_json(const std::string& json_config) {
    ServerConfig config = ConfigLoader::load_from_json(json_config);
    return std::make_unique<UnifiedServer>(config);
}

ServerConfig ServerFactory::get_default_config(ServerType type) {
    ServerConfig config;
    
    switch (type) {
        case ServerType::SIMPLE_HTTP:
            config.enable_http = true;
            config.enable_websocket = false;
            config.use_multi_thread = false;
            config.worker_threads = 1;
            config.max_connections = 1000;
            break;
            
        case ServerType::HIGH_PERFORMANCE:
            config.enable_http = true;
            config.enable_websocket = true;
            config.use_multi_thread = true;
            config.worker_threads = std::thread::hardware_concurrency();
            config.max_connections = 50000;
            config.dispatcher_type = EventDispatcherFactory::Type::EPOLL;
            config.enable_statistics = true;
            break;
            
        case ServerType::WEBSOCKET:
            config.enable_http = false;
            config.enable_websocket = true;
            config.use_multi_thread = false;
            config.worker_threads = 1;
            config.max_connections = 10000;
            break;
            
        case ServerType::FULL_FEATURED:
            config.enable_http = true;
            config.enable_websocket = true;
            config.enable_custom_protocol = true;
            config.use_multi_thread = true;
            config.worker_threads = std::thread::hardware_concurrency();
            config.max_connections = 20000;
            config.enable_statistics = true;
            config.enable_rate_limiting = true;
            config.enable_logging = true;
            break;
            
        case ServerType::MICROSERVICE:
            config.enable_http = true;
            config.enable_websocket = false;
            config.use_multi_thread = true;
            config.worker_threads = 4;
            config.max_connections = 5000;
            config.enable_statistics = true;
            config.enable_logging = true;
            break;
            
        case ServerType::DEVELOPMENT:
            config.enable_http = true;
            config.enable_websocket = true;
            config.use_multi_thread = false;
            config.worker_threads = 1;
            config.max_connections = 100;
            config.enable_statistics = true;
            config.enable_logging = true;
            config.log_level = "DEBUG";
            break;
    }
    
    return config;
}

// ServerCluster 实现
ServerCluster::ServerCluster(size_t num_servers)
    : load_balance_strategy_(LoadBalanceStrategy::ROUND_ROBIN),
      running_(false),
      health_check_enabled_(false),
      health_check_interval_(std::chrono::seconds(30)) {
}

ServerCluster::~ServerCluster() {
    stop_all();
}

void ServerCluster::add_server(std::unique_ptr<UnifiedServer> server) {
    std::lock_guard<std::mutex> lock(cluster_mutex_);
    servers_.push_back(std::move(server));
}

bool ServerCluster::start_all() {
    std::lock_guard<std::mutex> lock(cluster_mutex_);
    
    bool all_started = true;
    for (auto& server : servers_) {
        if (!server->start()) {
            all_started = false;
        }
    }
    
    if (all_started) {
        running_ = true;
        
        if (health_check_enabled_) {
            health_check_thread_ = std::make_unique<std::thread>([this]() {
                health_check_loop();
            });
        }
    }
    
    return all_started;
}

void ServerCluster::stop_all() {
    running_ = false;
    health_check_enabled_ = false;
    
    if (health_check_thread_ && health_check_thread_->joinable()) {
        health_check_thread_->join();
        health_check_thread_.reset();
    }
    
    std::lock_guard<std::mutex> lock(cluster_mutex_);
    
    for (auto& server : servers_) {
        server->stop();
    }
    
    for (auto& thread : server_threads_) {
        if (thread->joinable()) {
            thread->join();
        }
    }
    
    server_threads_.clear();
}

bool ServerCluster::is_running() const {
    return running_;
}

void ServerCluster::set_load_balance_strategy(LoadBalanceStrategy strategy) {
    load_balance_strategy_ = strategy;
}

void ServerCluster::enable_health_check(std::chrono::seconds interval) {
    health_check_enabled_ = true;
    health_check_interval_ = interval;
}

void ServerCluster::disable_health_check() {
    health_check_enabled_ = false;
}

ServerCluster::ClusterStats ServerCluster::get_cluster_statistics() const {
    std::lock_guard<std::mutex> lock(cluster_mutex_);
    
    ClusterStats cluster_stats;
    cluster_stats.total_servers = servers_.size();
    
    for (const auto& server : servers_) {
        if (server->is_running()) {
            cluster_stats.active_servers++;
            
            auto stats = server->get_statistics();
            cluster_stats.total_connections += stats.active_connections;
            cluster_stats.total_requests += stats.total_requests;
        }
    }
    
    return cluster_stats;
}

void ServerCluster::scale_up(size_t additional_servers) {
    // 简化实现：这里需要根据配置创建新的服务器实例
    std::cout << "Scaling up by " << additional_servers << " servers" << std::endl;
}

void ServerCluster::scale_down(size_t servers_to_remove) {
    std::lock_guard<std::mutex> lock(cluster_mutex_);
    
    size_t to_remove = std::min(servers_to_remove, servers_.size());
    
    for (size_t i = 0; i < to_remove; ++i) {
        if (!servers_.empty()) {
            servers_.back()->stop();
            servers_.pop_back();
        }
    }
}

void ServerCluster::health_check_loop() {
    while (health_check_enabled_ && running_) {
        std::this_thread::sleep_for(health_check_interval_);
        
        if (!health_check_enabled_) break;
        
        restart_failed_servers();
    }
}

void ServerCluster::restart_failed_servers() {
    std::lock_guard<std::mutex> lock(cluster_mutex_);
    
    for (auto& server : servers_) {
        if (!server->is_running()) {
            std::cout << "Restarting failed server..." << std::endl;
            server->start();
        }
    }
}

// ConfigLoader 实现
ServerConfig ConfigLoader::load_from_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open config file: " + filename);
    }
    
    std::string json_content((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
    
    return load_from_json(json_content);
}

ServerConfig ConfigLoader::load_from_json(const std::string& json_str) {
    // 简化的JSON解析实现
    // 在实际项目中，应该使用专业的JSON库如nlohmann/json
    
    ServerConfig config;
    
    // 这里应该实现完整的JSON解析
    // 为了简化，我们只提供基本的解析逻辑
    
    if (json_str.find("\"port\"") != std::string::npos) {
        // 简单的端口解析
        size_t pos = json_str.find("\"port\":");
        if (pos != std::string::npos) {
            pos = json_str.find(":", pos) + 1;
            size_t end = json_str.find(",", pos);
            if (end == std::string::npos) end = json_str.find("}", pos);
            
            std::string port_str = json_str.substr(pos, end - pos);
            // 去除空格和引号
            port_str.erase(std::remove_if(port_str.begin(), port_str.end(),
                          [](char c) { return std::isspace(c) || c == '"'; }),
                          port_str.end());
            
            try {
                config.port = std::stoi(port_str);
            } catch (...) {
                // 使用默认端口
            }
        }
    }
    
    return config;
}

bool ConfigLoader::save_to_file(const ServerConfig& config, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    file << to_json(config);
    return true;
}

std::string ConfigLoader::to_json(const ServerConfig& config) {
    std::ostringstream oss;
    
    oss << "{\n";
    oss << "  \"host\": \"" << config.host << "\",\n";
    oss << "  \"port\": " << config.port << ",\n";
    oss << "  \"max_connections\": " << config.max_connections << ",\n";
    oss << "  \"keepalive_timeout\": " << config.keepalive_timeout.count() << ",\n";
    oss << "  \"worker_threads\": " << config.worker_threads << ",\n";
    oss << "  \"use_multi_thread\": " << (config.use_multi_thread ? "true" : "false") << ",\n";
    oss << "  \"enable_http\": " << (config.enable_http ? "true" : "false") << ",\n";
    oss << "  \"enable_websocket\": " << (config.enable_websocket ? "true" : "false") << ",\n";
    oss << "  \"enable_statistics\": " << (config.enable_statistics ? "true" : "false") << ",\n";
    oss << "  \"enable_logging\": " << (config.enable_logging ? "true" : "false") << ",\n";
    oss << "  \"log_level\": \"" << config.log_level << "\",\n";
    oss << "  \"log_file\": \"" << config.log_file << "\"\n";
    oss << "}";
    
    return oss.str();
}

bool ConfigLoader::validate_config(const ServerConfig& config, std::string& error_message) {
    if (config.port <= 0 || config.port > 65535) {
        error_message = "Invalid port number: " + std::to_string(config.port);
        return false;
    }
    
    if (config.max_connections == 0) {
        error_message = "max_connections must be greater than 0";
        return false;
    }
    
    if (config.worker_threads == 0) {
        error_message = "worker_threads must be greater than 0";
        return false;
    }
    
    if (!config.enable_http && !config.enable_websocket && !config.enable_custom_protocol) {
        error_message = "At least one protocol must be enabled";
        return false;
    }
    
    return true;
}

ServerConfig ConfigLoader::merge_configs(const ServerConfig& base, const ServerConfig& override) {
    ServerConfig result = base;
    
    // 合并配置（override中的非默认值会覆盖base中的值）
    if (override.port != 8080) result.port = override.port;
    if (override.host != "0.0.0.0") result.host = override.host;
    if (override.max_connections != 10000) result.max_connections = override.max_connections;
    // ... 其他字段的合并逻辑
    
    return result;
}