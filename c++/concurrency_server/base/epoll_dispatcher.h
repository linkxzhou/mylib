#ifndef EPOLL_DISPATCHER_H
#define EPOLL_DISPATCHER_H

#include "event_dispatcher.h"

#ifdef __linux__
#include <sys/epoll.h>
#include <map>
#include <vector>

// Epoll事件分发器实现（Linux特有）
class EpollDispatcher : public EventDispatcher {
public:
    explicit EpollDispatcher(int max_events = 1024) 
        : max_events_(max_events), epoll_fd_(-1) {
        
        // 创建epoll实例
        epoll_fd_ = epoll_create1(EPOLL_CLOEXEC);
        if (epoll_fd_ == -1) {
            throw std::runtime_error("Failed to create epoll instance");
        }
        
        // 分配事件数组
        events_.resize(max_events);
    }
    
    ~EpollDispatcher() override {
        // 清理所有文件描述符
        for (const auto& pair : event_map_) {
            if (pair.first >= 0) {
                close(pair.first);
            }
        }
        
        if (epoll_fd_ != -1) {
            close(epoll_fd_);
        }
    }
    
    bool add_event(int fd, EventType events, EventCallback callback) override {
        if (fd < 0 || epoll_fd_ == -1) {
            return false;
        }
        
        // 检查是否已存在
        if (event_map_.find(fd) != event_map_.end()) {
            return false;
        }
        
        // 添加到事件映射
        event_map_[fd] = {fd, events, std::move(callback)};
        
        // 添加到epoll
        struct epoll_event event;
        event.events = convert_to_epoll_events(events);
        event.data.fd = fd;
        
        if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, fd, &event) == -1) {
            event_map_.erase(fd);
            return false;
        }
        
        return true;
    }
    
    bool modify_event(int fd, EventType events, EventCallback callback) override {
        auto it = event_map_.find(fd);
        if (it == event_map_.end() || epoll_fd_ == -1) {
            return false;
        }
        
        // 更新事件映射
        it->second.events = events;
        it->second.callback = std::move(callback);
        
        // 修改epoll事件
        struct epoll_event event;
        event.events = convert_to_epoll_events(events);
        event.data.fd = fd;
        
        if (epoll_ctl(epoll_fd_, EPOLL_CTL_MOD, fd, &event) == -1) {
            return false;
        }
        
        return true;
    }
    
    bool remove_event(int fd) override {
        auto it = event_map_.find(fd);
        if (it == event_map_.end() || epoll_fd_ == -1) {
            return false;
        }
        
        // 从epoll中移除
        if (epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, fd, nullptr) == -1) {
            // 即使epoll_ctl失败，也要从映射中移除
        }
        
        // 从映射中移除
        event_map_.erase(it);
        
        return true;
    }
    
    int wait_events(int timeout_ms = -1) override {
        if (epoll_fd_ == -1) {
            return -1;
        }
        
        num_ready_events_ = epoll_wait(epoll_fd_, events_.data(), max_events_, timeout_ms);
        
        if (num_ready_events_ < 0 && errno != EINTR) {
            perror("epoll_wait failed");
        }
        
        return num_ready_events_;
    }
    
    void dispatch_events() override {
        for (int i = 0; i < num_ready_events_; ++i) {
            const struct epoll_event& event = events_[i];
            int fd = event.data.fd;
            
            auto it = event_map_.find(fd);
            if (it == event_map_.end()) {
                continue;
            }
            
            EventType triggered_events = convert_from_epoll_events(event.events);
            
            if (static_cast<int>(triggered_events) != 0) {
                it->second.callback(fd, triggered_events);
            }
        }
    }
    
    int get_max_events() const override {
        return max_events_;
    }
    
    bool supports_edge_trigger() const override {
        return true;  // epoll支持边缘触发
    }
    
private:
    uint32_t convert_to_epoll_events(EventType events) {
        uint32_t epoll_events = 0;
        
        if (has_event(events, EventType::READ)) {
            epoll_events |= EPOLLIN;
        }
        if (events && EventType::WRITE) {
            epoll_events |= EPOLLOUT;
        }
        if (events && EventType::ERROR) {
            epoll_events |= EPOLLERR;
        }
        if (events && EventType::HANGUP) {
            epoll_events |= EPOLLHUP;
        }
        if (events && EventType::EDGE_TRIGGERED) {
            epoll_events |= EPOLLET;
        }
        
        return epoll_events;
    }
    
    EventType convert_from_epoll_events(uint32_t epoll_events) {
        EventType events = static_cast<EventType>(0);
        
        if (epoll_events & EPOLLIN) {
            events = events | EventType::READ;
        }
        if (epoll_events & EPOLLOUT) {
            events = events | EventType::WRITE;
        }
        if (epoll_events & EPOLLERR) {
            events = events | EventType::ERROR;
        }
        if (epoll_events & EPOLLHUP) {
            events = events | EventType::HANGUP;
        }
        if (epoll_events & EPOLLET) {
            events = events | EventType::EDGE_TRIGGERED;
        }
        
        return events;
    }
    
private:
    int max_events_;
    int epoll_fd_;
    int num_ready_events_;
    std::vector<struct epoll_event> events_;
    std::map<int, Event> event_map_;
};

#else
// 在非Linux系统上提供空实现
class EpollDispatcher : public EventDispatcher {
public:
    explicit EpollDispatcher(int max_events = 1024) {
        throw std::runtime_error("Epoll is not available on this platform");
    }
    
    bool add_event(int fd, EventType events, EventCallback callback) override { return false; }
    bool modify_event(int fd, EventType events, EventCallback callback) override { return false; }
    bool remove_event(int fd) override { return false; }
    int wait_events(int timeout_ms = -1) override { return -1; }
    void dispatch_events() override {}
    int get_max_events() const override { return 0; }
    bool supports_edge_trigger() const override { return false; }
};

#endif // __linux__

#endif // EPOLL_DISPATCHER_H