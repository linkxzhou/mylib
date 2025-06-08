#ifndef SELECT_DISPATCHER_H
#define SELECT_DISPATCHER_H

#include "event_dispatcher.h"
#include <sys/select.h>
#include <map>
#include <algorithm>

// Select事件分发器实现
class SelectDispatcher : public EventDispatcher {
public:
    SelectDispatcher(int max_events = 1024) : max_events_(max_events), max_fd_(-1) {
        FD_ZERO(&master_read_set_);
        FD_ZERO(&master_write_set_);
        FD_ZERO(&master_error_set_);
    }
    
    ~SelectDispatcher() override {
        // 清理所有文件描述符
        for (auto& pair : event_map_) {
            if (pair.first >= 0) {
                close(pair.first);
            }
        }
    }
    
    bool add_event(int fd, EventType events, EventCallback callback) override {
        if (fd < 0 || fd >= FD_SETSIZE) {
            return false;
        }
        
        // 添加到事件映射
        event_map_[fd] = {fd, events, std::move(callback)};
        
        // 更新fd_set
        update_fd_sets(fd, events, true);
        
        // 更新最大文件描述符
        max_fd_ = std::max(max_fd_, fd);
        
        return true;
    }
    
    bool modify_event(int fd, EventType events, EventCallback callback) override {
        auto it = event_map_.find(fd);
        if (it == event_map_.end()) {
            return false;
        }
        
        // 先清除旧的事件
        update_fd_sets(fd, it->second.events, false);
        
        // 更新事件
        it->second.events = events;
        it->second.callback = std::move(callback);
        
        // 设置新的事件
        update_fd_sets(fd, events, true);
        
        return true;
    }
    
    bool remove_event(int fd) override {
        auto it = event_map_.find(fd);
        if (it == event_map_.end()) {
            return false;
        }
        
        // 从fd_set中移除
        update_fd_sets(fd, it->second.events, false);
        
        // 从映射中移除
        event_map_.erase(it);
        
        // 更新最大文件描述符
        if (fd == max_fd_) {
            update_max_fd();
        }
        
        return true;
    }
    
    int wait_events(int timeout_ms = -1) override {
        // 复制master sets
        active_read_set_ = master_read_set_;
        active_write_set_ = master_write_set_;
        active_error_set_ = master_error_set_;
        
        struct timeval timeout;
        struct timeval* timeout_ptr = nullptr;
        
        if (timeout_ms >= 0) {
            timeout.tv_sec = timeout_ms / 1000;
            timeout.tv_usec = (timeout_ms % 1000) * 1000;
            timeout_ptr = &timeout;
        }
        
        int result = select(max_fd_ + 1, &active_read_set_, &active_write_set_, 
                           &active_error_set_, timeout_ptr);
        
        if (result < 0 && errno != EINTR) {
            perror("select failed");
        }
        
        return result;
    }
    
    void dispatch_events() override {
        for (auto& pair : event_map_) {
            int fd = pair.first;
            const Event& event = pair.second;
            
            EventType triggered_events = static_cast<EventType>(0);
            
            if (FD_ISSET(fd, &active_read_set_)) {
                triggered_events = triggered_events | EventType::READ;
            }
            if (FD_ISSET(fd, &active_write_set_)) {
                triggered_events = triggered_events | EventType::WRITE;
            }
            if (FD_ISSET(fd, &active_error_set_)) {
                triggered_events = triggered_events | EventType::ERROR;
            }
            
            if (static_cast<int>(triggered_events) != 0) {
                event.callback(fd, triggered_events);
            }
        }
    }
    
    int get_max_events() const override {
        return max_events_;
    }
    
    bool supports_edge_trigger() const override {
        return false;  // select不支持边缘触发
    }
    
private:
    void update_fd_sets(int fd, EventType events, bool add) {
        if (events && EventType::READ) {
            if (add) {
                FD_SET(fd, &master_read_set_);
            } else {
                FD_CLR(fd, &master_read_set_);
            }
        }
        
        if (events && EventType::WRITE) {
            if (add) {
                FD_SET(fd, &master_write_set_);
            } else {
                FD_CLR(fd, &master_write_set_);
            }
        }
        
        if (events && EventType::ERROR) {
            if (add) {
                FD_SET(fd, &master_error_set_);
            } else {
                FD_CLR(fd, &master_error_set_);
            }
        }
    }
    
    void update_max_fd() {
        max_fd_ = -1;
        for (const auto& pair : event_map_) {
            max_fd_ = std::max(max_fd_, pair.first);
        }
    }
    
private:
    int max_events_;
    int max_fd_;
    
    fd_set master_read_set_;
    fd_set master_write_set_;
    fd_set master_error_set_;
    
    fd_set active_read_set_;
    fd_set active_write_set_;
    fd_set active_error_set_;
    
    std::map<int, Event> event_map_;
};

#endif // SELECT_DISPATCHER_H