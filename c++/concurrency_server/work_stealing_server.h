#ifndef WORK_STEALING_SERVER_H
#define WORK_STEALING_SERVER_H

#include "./base/server_base.h"
#include <thread>
#include <vector>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <random>

// 工作窃取模型 - 每个线程有自己的任务队列，空闲时从其他线程窃取任务
class WorkStealingServer : public ServerBase {
public:
    struct Task {
        int client_fd;
        Task(int fd) : client_fd(fd) {}
    };
    
    class WorkStealingQueue {
    private:
        std::deque<Task> queue;
        mutable std::mutex mutex;
        
    public:
        void push(Task task) {
            std::lock_guard<std::mutex> lock(mutex);
            queue.push_back(task);
        }
        
        bool pop(Task& task) {
            std::lock_guard<std::mutex> lock(mutex);
            if (queue.empty()) {
                return false;
            }
            task = queue.front();
            queue.pop_front();
            return true;
        }
        
        bool steal(Task& task) {
            std::lock_guard<std::mutex> lock(mutex);
            if (queue.empty()) {
                return false;
            }
            task = queue.back();
            queue.pop_back();
            return true;
        }
        
        bool empty() const {
            std::lock_guard<std::mutex> lock(mutex);
            return queue.empty();
        }
        
        size_t size() const {
            std::lock_guard<std::mutex> lock(mutex);
            return queue.size();
        }
    };
    
private:
    std::vector<std::thread> workers;
    std::vector<std::unique_ptr<WorkStealingQueue>> queues;
    std::atomic<bool> running;
    std::atomic<int> next_queue;
    int num_threads;
    
public:
    WorkStealingServer(int port, int threads = std::thread::hardware_concurrency()) 
        : ServerBase(port), running(false), next_queue(0), num_threads(threads) {
        
        if (num_threads == 0) {
            num_threads = 4;
        }
        
        // 创建每个线程的工作队列
        for (int i = 0; i < num_threads; ++i) {
            queues.emplace_back(std::unique_ptr<WorkStealingQueue>(new WorkStealingQueue()));
        }
    }
    
    ~WorkStealingServer() {
        stop();
    }
    
    bool start() override {
        // 创建监听socket
        if (!create_listen_socket()) {
            return false;
        }
        
        std::cout << "Work Stealing Server listening on port " << port_ 
                  << " with " << num_threads << " worker threads" << std::endl;
        
        running = true;
        
        // 启动工作线程
        for (int i = 0; i < num_threads; ++i) {
            workers.emplace_back(&WorkStealingServer::worker_thread, this, i);
        }
        
        // 主线程负责接受连接
        accept_loop();
        return true;
    }
    
    void stop() override {
        if (running.exchange(false)) {
            // 等待所有工作线程结束
            for (auto& worker : workers) {
                if (worker.joinable()) {
                    worker.join();
                }
            }
            workers.clear();
        }
    }
    
private:
    void accept_loop() {
        while (running) {
            struct sockaddr_in client_addr;
            socklen_t client_len = sizeof(client_addr);
            
            int client_fd = accept(server_fd_, (struct sockaddr*)&client_addr, &client_len);
            if (client_fd == -1) {
                if (errno == EINTR || !running) {
                    break;
                }
                perror("accept");
                continue;
            }
            
            #if DEBUG
            std::cout << "New client connected: " << client_fd << std::endl;
            #endif
            
            // 使用轮询方式分配任务到队列
            int queue_index = next_queue.fetch_add(1) % num_threads;
            queues[queue_index]->push(Task(client_fd));
        }
    }
    
    void worker_thread(int thread_id) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, num_threads - 1);
        
        while (running) {
            Task task(0);
            bool found_task = false;
            
            // 首先尝试从自己的队列获取任务
            if (queues[thread_id]->pop(task)) {
                found_task = true;
            } else {
                // 如果自己的队列为空，尝试从其他线程窃取任务
                for (int attempts = 0; attempts < num_threads && !found_task; ++attempts) {
                    int victim = dis(gen);
                    if (victim != thread_id && queues[victim]->steal(task)) {
                        found_task = true;
                        #if DEBUG
                        std::cout << "Thread " << thread_id << " stole task from thread " << victim << std::endl;
                        #endif
                    }
                }
            }
            
            if (found_task) {
                handle_client(task.client_fd);
            } else {
                // 没有任务时短暂休眠
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }
    
    void handle_client(int client_fd) override {
        char buffer[BUFFER_SIZE];
        ssize_t bytes_read = read(client_fd, buffer, sizeof(buffer) - 1);
        
        if (bytes_read > 0) {
            buffer[bytes_read] = '\0';
            #if DEBUG
            std::cout << "Thread " << std::this_thread::get_id() 
                      << " received from client " << client_fd << ": " << buffer;
            #endif
            
            std::string response = "HTTP/1.1 200 OK\r\n"
                                  "Content-Type: text/plain\r\n"
                                  "Content-Length: 30\r\n"
                                  "\r\n"
                                  "Hello from Work Stealing Server!";
            
            ssize_t bytes_sent = write(client_fd, response.c_str(), response.length());
            if (bytes_sent == -1) {
                perror("write");
            }
        } else if (bytes_read == -1) {
            perror("read");
        }
        
        close(client_fd);
        #if DEBUG
        std::cout << "Client " << client_fd << " disconnected" << std::endl;
        #endif
    }
    
    static const int BUFFER_SIZE = 1024;
};

#endif // WORK_STEALING_SERVER_H