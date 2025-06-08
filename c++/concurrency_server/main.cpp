#include <iostream>
#include <string>
#include <thread>
#include <chrono>

// 包含所有服务器实现
#include "single_process_server.h"
#include "multi_process_server.h"
#include "multi_thread_server.h"
#include "process_pool1_server.h"
#include "process_pool2_server.h"
#include "thread_pool_server.h"
#include "leader_follower_server.h"
#include "select_server.h"
#include "poll_server.h"
#include "epoll_server.h"
#include "kqueue_server.h"
#include "reactor_server.h"
#include "coroutine_server.h"
#include "event_loop_server.h"
#include "work_stealing_server.h"
#include "actor_server.h"
#include "fiber_server.h"
#include "producer_consumer_server.h"
#include "half_sync_async_server.h"
#include "proactor_server.h"
#include "pipeline_server.h"
#include "hybrid_server.h"

void print_usage() {
    std::cout << "Usage: ./server <model> [port]" << std::endl;
    std::cout << "Available models:" << std::endl;
    std::cout << "  1. single_process    - Single process server" << std::endl;
    std::cout << "  2. multi_process     - Multi process server" << std::endl;
    std::cout << "  3. multi_thread      - Multi thread server" << std::endl;
    std::cout << "  4. process_pool1     - Process pool server (shared socket)" << std::endl;
    std::cout << "  5. process_pool2     - Process pool server (SO_REUSEPORT)" << std::endl;
    std::cout << "  6. thread_pool       - Thread pool server" << std::endl;
    std::cout << "  7. leader_follower   - Leader-Follower server" << std::endl;
    std::cout << "  8. select            - Select I/O multiplexing server" << std::endl;
    std::cout << "  9. poll              - Poll I/O multiplexing server" << std::endl;
    std::cout << " 10. epoll             - Epoll I/O multiplexing server (Linux)" << std::endl;
    std::cout << " 11. kqueue            - Kqueue I/O multiplexing server (BSD/macOS)" << std::endl;
    std::cout << " 12. reactor           - Reactor pattern server" << std::endl;
    std::cout << " 13. coroutine         - Coroutine-style server" << std::endl;
    std::cout << " 14. event_loop        - Event Loop server" << std::endl;
    std::cout << " 15. work_stealing     - Work Stealing server" << std::endl;
    std::cout << " 16. actor             - Actor model server" << std::endl;
    std::cout << " 17. fiber             - Fiber/Green Thread server" << std::endl;
    std::cout << " 18. producer_consumer - Producer-Consumer server" << std::endl;
    std::cout << " 19. half_sync_async   - Half-Sync/Half-Async server" << std::endl;
    std::cout << " 20. proactor          - Proactor pattern server" << std::endl;
    std::cout << " 21. pipeline          - Pipeline server" << std::endl;
    std::cout << " 22. hybrid            - Hybrid model server" << std::endl;
}

std::unique_ptr<ServerBase> create_server(const std::string& model, int port) {
    if (model == "single_process") {
        return std::unique_ptr<ServerBase>(new SingleProcessServer(port));
    } else if (model == "multi_process") {
        return std::unique_ptr<ServerBase>(new MultiProcessServer(port));
    } else if (model == "multi_thread") {
        return std::unique_ptr<ServerBase>(new MultiThreadServer(port));
    } else if (model == "process_pool1") {
        return std::unique_ptr<ServerBase>(new ProcessPool1Server(port));
    } else if (model == "process_pool2") {
        return std::unique_ptr<ServerBase>(new ProcessPool2Server(port));
    } else if (model == "thread_pool") {
        return std::unique_ptr<ServerBase>(new ThreadPoolServer(port));
    } else if (model == "leader_follower") {
        return std::unique_ptr<ServerBase>(new LeaderFollowerServer(port));
    } else if (model == "select") {
        return std::unique_ptr<ServerBase>(new SelectServer(port));
    } else if (model == "poll") {
        return std::unique_ptr<ServerBase>(new PollServer(port));
    } else if (model == "epoll") {
        return std::unique_ptr<ServerBase>(new EpollServer(port));
    } else if (model == "kqueue") {
        return std::unique_ptr<ServerBase>(new KqueueServer(port));
    } else if (model == "reactor") {
        return std::unique_ptr<ServerBase>(new ReactorServer(port));
    } else if (model == "coroutine") {
        return std::unique_ptr<ServerBase>(new CoroutineServer(port));
    } else if (model == "event_loop") {
        return std::unique_ptr<ServerBase>(new EventLoopServer(port));
    } else if (model == "work_stealing") {
        return std::unique_ptr<ServerBase>(new WorkStealingServer(port));
    } else if (model == "actor") {
        return std::unique_ptr<ServerBase>(new ActorServer(port));
    } else if (model == "fiber") {
        return std::unique_ptr<ServerBase>(new FiberServer(port));
    } else if (model == "producer_consumer") {
        return std::unique_ptr<ServerBase>(new ProducerConsumerServer(port));
    } else if (model == "half_sync_async") {
        return std::unique_ptr<ServerBase>(new HalfSyncAsyncServer(port));
    } else if (model == "proactor") {
        return std::unique_ptr<ServerBase>(new ProactorServer(port));
    } else if (model == "pipeline") {
        return std::unique_ptr<ServerBase>(new PipelineServer(port));
    } else if (model == "hybrid") {
        return std::unique_ptr<ServerBase>(new HybridServer(port));
    }
    
    return nullptr;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage();
        return 1;
    }
    
    std::string model = argv[1];
    int port = (argc >= 3) ? std::atoi(argv[2]) : 8080;
    
    auto server = create_server(model, port);
    if (!server) {
        std::cerr << "Unknown server model: " << model << std::endl;
        print_usage();
        return 1;
    }
    
    std::cout << "Starting " << model << " server on port " << port << std::endl;
    std::cout << "Press Ctrl+C to stop the server" << std::endl;
    
    // 设置信号处理
    signal(SIGINT, [](int) {
        std::cout << "\nShutting down server..." << std::endl;
        exit(0);
    });
    
    if (!server->start()) {
        std::cerr << "Failed to start server" << std::endl;
        return 1;
    }
    
    return 0;
}

// 简单的客户端测试函数
void test_client(const std::string& host, int port, int num_requests = 1) {
    for (int i = 0; i < num_requests; ++i) {
        int sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) {
            perror("socket creation failed");
            continue;
        }
        
        struct sockaddr_in server_addr;
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(port);
        
        if (inet_pton(AF_INET, host.c_str(), &server_addr.sin_addr) <= 0) {
            perror("inet_pton failed");
            close(sock);
            continue;
        }
        
        if (connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
            perror("connect failed");
            close(sock);
            continue;
        }
        
        const char* request = "GET / HTTP/1.1\r\nHost: localhost\r\n\r\n";
        send(sock, request, strlen(request), 0);
        
        char buffer[1024] = {0};
        recv(sock, buffer, sizeof(buffer) - 1, 0);
        
        std::cout << "Response " << i + 1 << ": " << buffer << std::endl;
        
        close(sock);
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}