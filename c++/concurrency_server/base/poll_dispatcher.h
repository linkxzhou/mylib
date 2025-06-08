#ifndef POLL_DISPATCHER_H
#define POLL_DISPATCHER_H

#include "event_dispatcher.h"
#include <poll.h>
#include <vector>
#include <map>
#include <algorithm>

// Poll事件分发器实现
class PollDispatcher : public EventDispatcher {
public:
    explicit PollDispatcher(int max_events = 1024) : max_events_(max_events) {
        poll_fds_.reserve(max_events);
    }
    
    ~PollDispatcher() override {
        // 清理所有文件描述符
        for (const auto& pair : event_map_) {
            if (pair.first >= 0) {
                close(pair.first);
            }
        }
    }
    
    bool add_event(int fd, EventType events, EventCallback callback) override {
        if (fd < 0) {
            return false;
        }
        
        // 检查是否已存在
        if (event_map_.find(fd) != event_map_.end()) {
            return false;
        }
        
        // 添加到事件映射
        event_map_[fd] = {fd, events, std::move(callback)};
        
        // 添加到poll数组
        struct pollfd pfd;
        pfd.fd = fd;
        pfd.events = convert_to_poll_events(events);
        pfd.revents = 0;
        
        poll_fds_.push_back(pfd);
        fd_to_index_[fd] = poll_fds_.size() - 1;
        
        return true;
    }
    
    bool modify_event(int fd, EventType events, EventCallback callback) override {
        auto it = event_map_.find(fd);
        if (it == event_map_.end()) {
            return false;
        }
        
        auto index_it = fd_to_index_.find(fd);
        if (index_it == fd_to_index_.end()) {
            return false;
        }
        
        // 更新事件映射
        it->second.events = events;
        it->second.callback = std::move(callback);
        
        // 更新poll数组
        size_t index = index_it->second;
        if (index < poll_fds_.size()) {
            poll_fds_[index].events = convert_to_poll_events(events);
        }
        
        return true;
    }
    
    bool remove_event(int fd) override {
        auto it = event_map_.find(fd);
        if (it == event_map_.end()) {
            return false;
        }
        
        auto index_it = fd_to_index_.find(fd);
        if (index_it == fd_to_index_.end()) {
            return false;
        }
        
        size_t index = index_it->second;
        
        // 从poll数组中移除（使用swap-and-pop技术）
        if (index < poll_fds_.size()) {
            if (index != poll_fds_.size() - 1) {
                // 将最后一个元素移到当前位置
                poll_fds_[index] = poll_fds_.back();
                // 更新被移动元素的索引
                fd_to_index_[poll_fds_[index].fd] = index;
            }
            poll_fds_.pop_back();
        }
        
        // 从映射中移除
        event_map_.erase(it);
        fd_to_index_.erase(index_it);
        
        return true;
    }
    
    int wait_events(int timeout_ms = -1) override {
        if (poll_fds_.empty()) {
            return 0;
        }
        
        int result = poll(poll_fds_.data(), poll_fds_.size(), timeout_ms);
        
        if (result < 0 && errno != EINTR) {
            perror("poll failed");
        }
        
        return result;
    }
    
    void dispatch_events() override {
        for (size_t i = 0; i < poll_fds_.size(); ++i) {
            const struct pollfd& pfd = poll_fds_[i];
            
            if (pfd.revents == 0) {
                continue;
            }
            
            auto it = event_map_.find(pfd.fd);
            if (it == event_map_.end()) {
                continue;
            }
            
            EventType triggered_events = convert_from_poll_events(pfd.revents);
            
            if (static_cast<int>(triggered_events) != 0) {
                it->second.callback(pfd.fd, triggered_events);
            }
            
            // 清除revents
            poll_fds_[i].revents = 0;
        }
    }
    
    int get_max_events() const override {
        return max_events_;
    }
    
    bool supports_edge_trigger() const override {
        return false;  // poll不支持边缘触发
    }
    
private:
    short convert_to_poll_events(EventType events) {
        short poll_events = 0;
        
        if (events && EventType::READ) {
            poll_events |= POLLIN;
        }
        if (events && EventType::WRITE) {
            poll_events |= POLLOUT;
        }
        if (events && EventType::ERROR) {
            poll_events |= POLLERR;
        }
        if (events && EventType::HANGUP) {
            poll_events |= POLLHUP;
        }
        
        return poll_events;
    }
    
    EventType convert_from_poll_events(short poll_events) {
        EventType events = static_cast<EventType>(0);
        
        if (poll_events & POLLIN) {
            events = events | EventType::READ;
        }
        if (poll_events & POLLOUT) {
            events = events | EventType::WRITE;
        }
        if (poll_events & POLLERR) {
            events = events | EventType::ERROR;
        }
        if (poll_events & POLLHUP) {
            events = events | EventType::HANGUP;
        }
        
        return events;
    }
    
private:
    int max_events_;
    std::vector<struct pollfd> poll_fds_;
    std::map<int, Event> event_map_;
    std::map<int, size_t> fd_to_index_;  // fd到poll_fds_索引的映射
};

#endif // POLL_DISPATCHER_H