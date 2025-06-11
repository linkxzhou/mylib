#ifndef KQUEUE_DISPATCHER_H
#define KQUEUE_DISPATCHER_H

#include "event_dispatcher.h"

#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__)
#include <sys/event.h>
#include <unistd.h>
#include <map>
#include <vector>

// Kqueue事件分发器实现（BSD/macOS特有）
class KqueueDispatcher : public EventDispatcher {
public:
    explicit KqueueDispatcher(int max_events = 1024) 
        : max_events_(max_events), kqueue_fd_(-1) {
        
        // 创建kqueue实例
        kqueue_fd_ = kqueue();
        if (kqueue_fd_ == -1) {
            throw std::runtime_error("Failed to create kqueue instance");
        }
        
        // 分配事件数组
        events_.resize(max_events);
    }
    
    ~KqueueDispatcher() override {
        // 清理所有文件描述符
        for (const auto& pair : event_map_) {
            if (pair.first >= 0) {
                close(pair.first);
            }
        }
        
        if (kqueue_fd_ != -1) {
            close(kqueue_fd_);
        }
    }
    
    bool add_event(int fd, EventType events, EventCallback callback) override {
        if (fd < 0 || kqueue_fd_ == -1) {
            return false;
        }
        
        // 检查是否已存在
        if (event_map_.find(fd) != event_map_.end()) {
            return false;
        }
        
        // 添加到事件映射
        event_map_[fd] = {fd, events, std::move(callback)};
        
        // 添加到kqueue
        std::vector<struct kevent> change_list;
        
        if (has_event(events, EventType::READ)) {
            struct kevent change_event;
            EV_SET(&change_event, fd, EVFILT_READ, EV_ADD | EV_ENABLE, 0, 0, nullptr);
            change_list.push_back(change_event);
        }
        
        if (events && EventType::WRITE) {
            struct kevent change_event;
            EV_SET(&change_event, fd, EVFILT_WRITE, EV_ADD | EV_ENABLE, 0, 0, nullptr);
            change_list.push_back(change_event);
        }
        
        if (!change_list.empty()) {
            if (kevent(kqueue_fd_, change_list.data(), change_list.size(), nullptr, 0, nullptr) == -1) {
                event_map_.erase(fd);
                return false;
            }
        }
        
        return true;
    }
    
    bool modify_event(int fd, EventType events, EventCallback callback) override {
        auto it = event_map_.find(fd);
        if (it == event_map_.end() || kqueue_fd_ == -1) {
            return false;
        }
        
        EventType old_events = it->second.events;
        
        // 更新事件映射
        it->second.events = events;
        it->second.callback = std::move(callback);
        
        // 修改kqueue事件
        std::vector<struct kevent> change_list;
        
        // 移除旧事件
        if (has_event(old_events, EventType::READ)) {
            struct kevent change_event;
            EV_SET(&change_event, fd, EVFILT_READ, EV_DELETE, 0, 0, nullptr);
            change_list.push_back(change_event);
        }
        
        if (old_events && EventType::WRITE) {
            struct kevent change_event;
            EV_SET(&change_event, fd, EVFILT_WRITE, EV_DELETE, 0, 0, nullptr);
            change_list.push_back(change_event);
        }
        
        // 添加新事件
        if (has_event(events, EventType::READ)) {
            struct kevent change_event;
            EV_SET(&change_event, fd, EVFILT_READ, EV_ADD | EV_ENABLE, 0, 0, nullptr);
            change_list.push_back(change_event);
        }
        
        if (events && EventType::WRITE) {
            struct kevent change_event;
            EV_SET(&change_event, fd, EVFILT_WRITE, EV_ADD | EV_ENABLE, 0, 0, nullptr);
            change_list.push_back(change_event);
        }
        
        if (!change_list.empty()) {
            if (kevent(kqueue_fd_, change_list.data(), change_list.size(), nullptr, 0, nullptr) == -1) {
                return false;
            }
        }
        
        return true;
    }
    
    bool remove_event(int fd) override {
        auto it = event_map_.find(fd);
        if (it == event_map_.end() || kqueue_fd_ == -1) {
            return false;
        }
        
        EventType events = it->second.events;
        
        // 从kqueue中移除
        std::vector<struct kevent> change_list;
        
        if (has_event(events, EventType::READ)) {
            struct kevent change_event;
            EV_SET(&change_event, fd, EVFILT_READ, EV_DELETE, 0, 0, nullptr);
            change_list.push_back(change_event);
        }
        
        if (events && EventType::WRITE) {
            struct kevent change_event;
            EV_SET(&change_event, fd, EVFILT_WRITE, EV_DELETE, 0, 0, nullptr);
            change_list.push_back(change_event);
        }
        
        if (!change_list.empty()) {
            kevent(kqueue_fd_, change_list.data(), change_list.size(), nullptr, 0, nullptr);
            // 即使失败也继续移除映射
        }
        
        // 从映射中移除
        event_map_.erase(it);
        
        return true;
    }
    
    int wait_events(int timeout_ms = -1) override {
        if (kqueue_fd_ == -1) {
            return -1;
        }
        
        struct timespec timeout;
        struct timespec* timeout_ptr = nullptr;
        
        if (timeout_ms >= 0) {
            timeout.tv_sec = timeout_ms / 1000;
            timeout.tv_nsec = (timeout_ms % 1000) * 1000000;
            timeout_ptr = &timeout;
        }
        
        num_ready_events_ = kevent(kqueue_fd_, nullptr, 0, events_.data(), max_events_, timeout_ptr);
        
        if (num_ready_events_ < 0 && errno != EINTR) {
            perror("kevent failed");
        }
        
        return num_ready_events_;
    }
    
    void dispatch_events() override {
        for (int i = 0; i < num_ready_events_; ++i) {
            const struct kevent& event = events_[i];
            int fd = static_cast<int>(event.ident);
            
            // 检查错误
            if (event.flags & EV_ERROR) {
                // 处理错误事件
                auto it = event_map_.find(fd);
                if (it != event_map_.end()) {
                    it->second.callback(fd, EventType::ERROR);
                }
                continue;
            }
            
            auto it = event_map_.find(fd);
            if (it == event_map_.end()) {
                continue;
            }
            
            EventType triggered_events = convert_from_kqueue_events(event);
            
            if (static_cast<int>(triggered_events) != 0) {
                it->second.callback(fd, triggered_events);
            }
        }
    }
    
    int get_max_events() const override {
        return max_events_;
    }
    
    bool supports_edge_trigger() const override {
        return true;  // kqueue默认是边缘触发
    }
    
private:
    EventType convert_from_kqueue_events(const struct kevent& event) {
        EventType events = static_cast<EventType>(0);
        
        if (event.filter == EVFILT_READ) {
            events = events | EventType::READ;
        }
        if (event.filter == EVFILT_WRITE) {
            events = events | EventType::WRITE;
        }
        
        // kqueue中的EOF表示连接关闭
        if (event.flags & EV_EOF) {
            events = events | EventType::HANGUP;
        }
        
        return events;
    }
    
private:
    int max_events_;
    int kqueue_fd_;
    int num_ready_events_;
    std::vector<struct kevent> events_;
    std::map<int, Event> event_map_;
};

#else
// 在非BSD/macOS系统上提供空实现
class KqueueDispatcher : public EventDispatcher {
public:
    explicit KqueueDispatcher(int max_events = 1024) {
        throw std::runtime_error("Kqueue is not available on this platform");
    }
    
    bool add_event(int fd, EventType events, EventCallback callback) override { return false; }
    bool modify_event(int fd, EventType events, EventCallback callback) override { return false; }
    bool remove_event(int fd) override { return false; }
    int wait_events(int timeout_ms = -1) override { return -1; }
    void dispatch_events() override {}
    int get_max_events() const override { return 0; }
    bool supports_edge_trigger() const override { return false; }
};

#endif // BSD/macOS

#endif // KQUEUE_DISPATCHER_H