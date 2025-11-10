#include "base/server_factory.h"
#include "base/monitoring.h"
#include <iostream>
#include <signal.h>

// 全局服务器实例
std::unique_ptr<UnifiedServer> server;

// 信号处理函数
void signal_handler(int signal) {
    std::cout << "\nReceived signal " << signal << ", shutting down server..." << std::endl;
    
    if (server) {
        server->stop();
    }
    
    shutdown_monitoring();
    exit(0);
}

int main(int argc, char* argv[]) {
    // 初始化监控系统
    init_monitoring();
    
    // 设置信号处理
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // 获取全局日志器
    auto logger = LogManager::instance().get_logger("main");
    LOG_INFO(logger, "Starting unified server example");
    
    // 获取性能监控器
    auto& monitor = PerformanceMonitor::instance();
    auto request_counter = monitor.register_counter("http_requests_total");
    auto response_timer = monitor.register_timer("http_response_time");
    auto active_connections = monitor.register_gauge("active_connections");
    
    try {
        // 方式1: 使用ServerBuilder构建自定义服务器
        server = ServerBuilder()
            .listen(8080)
            .bind("0.0.0.0")
            .max_connections(10000)
            .enable_multi_threading(true)
            .worker_threads(4)
            .use_dispatcher(EventDispatcherFactory::Type::AUTO)
            .enable_http(true)
            .enable_websocket(true)
            .enable_statistics(true)
            .enable_logging(true)
            .build();
        
        // 方式2: 使用ServerFactory创建预配置服务器
        // server = ServerFactory::create_high_performance_server(8080);
        
        // 设置HTTP路由
        server->get("/", [&](const HttpRequest& req, HttpResponse& resp) {
            TIMER_SCOPE(response_timer);
            request_counter->increment();
            
            resp.set_content_type("text/html");
            resp.body = R"(
                <!DOCTYPE html>
                <html>
                <head><title>Unified Server Example</title></head>
                <body>
                    <h1>Welcome to Unified Server!</h1>
                    <p>This server is built with the optimized concurrency framework.</p>
                    <ul>
                        <li><a href="/api/stats">Server Statistics</a></li>
                        <li><a href="/api/metrics">Performance Metrics</a></li>
                        <li><a href="/websocket">WebSocket Test</a></li>
                    </ul>
                </body>
                </html>
            )";
            resp.set_content_length(resp.body.length());
            
            LOG_INFO(logger, "Served homepage to client");
        });
        
        server->get("/api/stats", [&](const HttpRequest& req, HttpResponse& resp) {
            TIMER_SCOPE(response_timer);
            request_counter->increment();
            
            auto stats = server->get_statistics();
            
            std::ostringstream json;
            json << "{\n";
            json << "  \"total_connections\": " << stats.total_connections << ",\n";
            json << "  \"active_connections\": " << stats.active_connections << ",\n";
            json << "  \"total_requests\": " << stats.total_requests << ",\n";
            json << "  \"requests_per_second\": " << stats.requests_per_second << ",\n";
            json << "  \"total_bytes_read\": " << stats.total_bytes_read << ",\n";
            json << "  \"total_bytes_written\": " << stats.total_bytes_written << "\n";
            json << "}";
            
            resp.set_content_type("application/json");
            resp.body = json.str();
            resp.set_content_length(resp.body.length());
            
            LOG_INFO(logger, "Served statistics API");
        });
        
        server->get("/api/metrics", [&](const HttpRequest& req, HttpResponse& resp) {
            TIMER_SCOPE(response_timer);
            request_counter->increment();
            
            std::string metrics = monitor.export_prometheus();
            
            resp.set_content_type("text/plain");
            resp.body = metrics;
            resp.set_content_length(resp.body.length());
            
            LOG_INFO(logger, "Served metrics API");
        });
        
        server->get("/websocket", [&](const HttpRequest& req, HttpResponse& resp) {
            resp.set_content_type("text/html");
            resp.body = R"(
                <!DOCTYPE html>
                <html>
                <head><title>WebSocket Test</title></head>
                <body>
                    <h1>WebSocket Test</h1>
                    <div id="messages"></div>
                    <input type="text" id="messageInput" placeholder="Enter message">
                    <button onclick="sendMessage()">Send</button>
                    <script>
                        const ws = new WebSocket('ws://localhost:8080/ws');
                        const messages = document.getElementById('messages');
                        
                        ws.onmessage = function(event) {
                            const div = document.createElement('div');
                            div.textContent = 'Received: ' + event.data;
                            messages.appendChild(div);
                        };
                        
                        function sendMessage() {
                            const input = document.getElementById('messageInput');
                            ws.send(input.value);
                            input.value = '';
                        }
                    </script>
                </body>
                </html>
            )";
            resp.set_content_length(resp.body.length());
        });
        
        // 设置POST路由示例
        server->post("/api/echo", [&](const HttpRequest& req, HttpResponse& resp) {
            TIMER_SCOPE(response_timer);
            request_counter->increment();
            
            resp.set_content_type("application/json");
            
            std::ostringstream json;
            json << "{\n";
            json << "  \"method\": \"" << "POST" << "\",\n";
            json << "  \"path\": \"" << req.path << "\",\n";
            json << "  \"body\": \"" << req.body << "\"\n";
            json << "}";
            
            resp.body = json.str();
            resp.set_content_length(resp.body.length());
            
            LOG_INFO(logger, "Served echo API");
        });
        
        // 设置中间件
        server->use_middleware([&](const HttpRequest& req, HttpResponse& resp) -> bool {
            // 记录请求
            LOG_INFO(logger, "Request: " + req.path);
            
            // 设置CORS头
            resp.set_header("Access-Control-Allow-Origin", "*");
            resp.set_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
            resp.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
            
            return true;  // 继续处理
        });
        
        // 设置WebSocket消息处理
        server->on_websocket_message([&](std::shared_ptr<Connection> conn, const std::string& message) {
            LOG_INFO(logger, "WebSocket message received: " + message);
            
            // 回显消息
            std::string response = "Echo: " + message;
            // 这里需要通过WebSocket处理器发送响应
            // websocket_handler->send_text(conn, response);
        });
        
        // 设置连接事件处理
        server->on_connection_open([&](std::shared_ptr<Connection> conn) {
            active_connections->increment();
            LOG_INFO(logger, "New connection from " + conn->get_remote_ip());
        });
        
        server->on_connection_close([&](std::shared_ptr<Connection> conn) {
            active_connections->decrement();
            LOG_INFO(logger, "Connection closed from " + conn->get_remote_ip());
        });
        
        server->on_error([&](const std::string& error) {
            LOG_ERROR(logger, "Server error: " + error);
        });
        
        // 启动系统监控
        SystemMonitor sys_monitor;
        sys_monitor.set_stats_callback([&](const SystemMonitor::SystemStats& stats) {
            auto cpu_gauge = monitor.register_gauge("system_cpu_usage_percent");
            auto memory_gauge = monitor.register_gauge("system_memory_usage_percent");
            auto fd_gauge = monitor.register_gauge("system_open_file_descriptors");
            
            cpu_gauge->set(stats.cpu_usage_percent);
            memory_gauge->set(stats.memory_usage_percent);
            fd_gauge->set(stats.open_file_descriptors);
            
            LOG_DEBUG(logger, "System stats updated - CPU: " + std::to_string(stats.cpu_usage_percent) + "%");
        });
        sys_monitor.start_monitoring(std::chrono::seconds(10));
        
        // 启动服务器
        LOG_INFO(logger, "Starting server on port 8080...");
        if (!server->start()) {
            LOG_FATAL(logger, "Failed to start server");
            return 1;
        }
        
        LOG_INFO(logger, "Server started successfully!");
        std::cout << "Server is running on http://localhost:8080" << std::endl;
        std::cout << "Available endpoints:" << std::endl;
        std::cout << "  GET  /              - Homepage" << std::endl;
        std::cout << "  GET  /api/stats     - Server statistics" << std::endl;
        std::cout << "  GET  /api/metrics   - Performance metrics" << std::endl;
        std::cout << "  GET  /websocket     - WebSocket test page" << std::endl;
        std::cout << "  POST /api/echo      - Echo API" << std::endl;
        std::cout << "Press Ctrl+C to stop the server" << std::endl;
        
        // 定期输出统计信息
        while (server->is_running()) {
            std::this_thread::sleep_for(std::chrono::seconds(30));
            
            auto stats = server->get_statistics();
            LOG_INFO(logger, "Server stats - Connections: " + std::to_string(stats.active_connections) + 
                            ", Requests: " + std::to_string(stats.total_requests) + 
                            ", RPS: " + std::to_string(stats.requests_per_second));
        }
        
    } catch (const std::exception& e) {
        LOG_FATAL(logger, "Server exception: " + std::string(e.what()));
        std::cerr << "Server error: " << e.what() << std::endl;
        return 1;
    }
    
    LOG_INFO(logger, "Server shutdown complete");
    shutdown_monitoring();
    return 0;
}

/*
编译命令:
g++ -std=c++17 -O2 -pthread \
    example_unified_server.cpp \
    base/server_factory.cpp \
    base/enhanced_event_loop.cpp \
    base/connection_manager.cpp \
    base/protocol_handler.cpp \
    base/monitoring.cpp \
    base/event_dispatcher.cpp \
    -lssl -lcrypto \
    -o unified_server_example

运行:
./unified_server_example

测试:
curl http://localhost:8080/
curl http://localhost:8080/api/stats
curl http://localhost:8080/api/metrics
curl -X POST -d '{"test": "data"}' -H "Content-Type: application/json" http://localhost:8080/api/echo
*/