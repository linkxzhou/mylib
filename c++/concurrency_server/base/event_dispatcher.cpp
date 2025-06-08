#include "event_dispatcher.h"
#include "select_dispatcher.h"
#include "poll_dispatcher.h"
#include "epoll_dispatcher.h"
#include "kqueue_dispatcher.h"
#include <stdexcept>
#include <iostream>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <errno.h>
#include <cstring>

// EventDispatcherFactory实现
std::unique_ptr<EventDispatcher> EventDispatcherFactory::create(Type type, int max_events) {
    switch (type) {
        case Type::SELECT:
            return std::unique_ptr<EventDispatcher>(new SelectDispatcher(max_events));
            
        case Type::POLL:
            return std::unique_ptr<EventDispatcher>(new PollDispatcher(max_events));
            
        case Type::EPOLL:
#ifdef __linux__
            return std::unique_ptr<EventDispatcher>(new EpollDispatcher(max_events));
#else
            throw std::runtime_error("Epoll not available on this platform");
#endif
            
        case Type::KQUEUE:
#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__)
            try {
                return std::unique_ptr<EventDispatcher>(new KqueueDispatcher(max_events));
            } catch (const std::exception& e) {
                throw std::runtime_error("Failed to create kqueue dispatcher");
            }
#else
            throw std::runtime_error("Kqueue not available on this platform");
#endif
            
        case Type::AUTO:
            return create(get_best_available(), max_events);
            
        default:
            throw std::invalid_argument("Unknown dispatcher type");
    }
}

EventDispatcherFactory::Type EventDispatcherFactory::get_best_available() {
#ifdef __linux__
    return Type::EPOLL;
#elif defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__)
    return Type::KQUEUE;
#else
    return Type::POLL;  // 回退到poll，因为它比select更高效
#endif
}

bool EventDispatcherFactory::is_available(Type type) {
    switch (type) {
        case Type::SELECT:
        case Type::POLL:
            return true;  // 这两个在所有POSIX系统上都可用
            
        case Type::EPOLL:
#ifdef __linux__
            return true;
#else
            return false;
#endif
            
        case Type::KQUEUE:
#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__)
            return true;
#else
            return false;
#endif
            
        case Type::AUTO:
            return true;  // AUTO总是可用的
            
        default:
            return false;
    }
}

// SimpleEventLoop实现
SimpleEventLoop::SimpleEventLoop(EventDispatcherFactory::Type type, int max_events) 
    : running_(false) {
    dispatcher_ = EventDispatcherFactory::create(type, max_events);
}

SimpleEventLoop::~SimpleEventLoop() {
    stop();
}

bool SimpleEventLoop::add_event(int fd, EventType events, EventCallback callback) {
    return dispatcher_->add_event(fd, events, std::move(callback));
}

bool SimpleEventLoop::remove_event(int fd) {
    return dispatcher_->remove_event(fd);
}

void SimpleEventLoop::run() {
    running_ = true;
    
    while (running_) {
        int num_events = dispatcher_->wait_events(1000);  // 1秒超时
        
        if (num_events < 0) {
            if (errno == EINTR) {
                continue;
            }
            std::cerr << "Event loop error" << std::endl;
            break;
        }
        
        if (num_events > 0) {
            dispatcher_->dispatch_events();
        }
    }
}

void SimpleEventLoop::stop() {
    running_ = false;
}

bool SimpleEventLoop::run_once(int timeout_ms) {
    int num_events = dispatcher_->wait_events(timeout_ms);
    
    if (num_events < 0) {
        return false;
    }
    
    if (num_events > 0) {
        dispatcher_->dispatch_events();
    }
    
    return true;
}

// ClientEventHandler实现
ClientEventHandler::ClientEventHandler(int client_fd) 
    : client_fd_(client_fd) {
}

ClientEventHandler::~ClientEventHandler() {
    if (client_fd_ >= 0) {
        close(client_fd_);
    }
}

void ClientEventHandler::handle_read(int fd) {
    if (read_handler_) {
        read_handler_(fd);
    } else {
        // 默认读处理：简单回显
        char buffer[1024];
        ssize_t bytes_read = read(fd, buffer, sizeof(buffer) - 1);
        
        if (bytes_read > 0) {
            buffer[bytes_read] = '\0';
            write(fd, buffer, bytes_read);  // 回显
        } else if (bytes_read == 0) {
            // 连接关闭
            handle_hangup(fd);
        }
    }
}

void ClientEventHandler::handle_write(int fd) {
    if (write_handler_) {
        write_handler_(fd);
    }
}

void ClientEventHandler::handle_error(int fd) {
    if (error_handler_) {
        error_handler_(fd);
    } else {
        std::cerr << "Error on client fd " << fd << std::endl;
        handle_hangup(fd);
    }
}

void ClientEventHandler::handle_hangup(int fd) {
    if (close_handler_) {
        close_handler_(fd);
    } else {
        std::cout << "Client disconnected: fd " << fd << std::endl;
        close(fd);
    }
}