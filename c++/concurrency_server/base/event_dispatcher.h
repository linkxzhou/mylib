#ifndef EVENT_DISPATCHER_H
#define EVENT_DISPATCHER_H

#include "server_base.h"
#include <functional>
#include <map>
#include <vector>
#include <memory>
#include <sys/socket.h>
#include <errno.h>

// 事件类型枚举
enum class EventType {
    READ = 1,
    WRITE = 2,
    ERROR = 4,
    HANGUP = 8,
    EDGE_TRIGGERED = 16  // 边缘触发标志
};

// 事件类型位运算操作符
inline EventType operator|(EventType a, EventType b) {
    return static_cast<EventType>(static_cast<int>(a) | static_cast<int>(b));
}

inline EventType operator&(EventType a, EventType b) {
    return static_cast<EventType>(static_cast<int>(a) & static_cast<int>(b));
}

// 检查EventType是否包含特定标志
inline bool operator&&(EventType a, EventType b) {
    return static_cast<int>(a & b) != 0;
}

// 将EventType转换为bool（用于if条件判断）
inline bool operator!(EventType a) {
    return static_cast<int>(a) == 0;
}

// 隐式转换为bool的辅助函数
inline bool has_event(EventType events, EventType flag) {
    return static_cast<int>(events & flag) != 0;
}

// 事件回调函数类型
using EventCallback = std::function<void(int fd, EventType events)>;

// 事件结构体
struct Event {
    int fd;
    EventType events;
    EventCallback callback;
    
    Event() : fd(-1), events(static_cast<EventType>(0)), callback(nullptr) {}
    Event(int f, EventType e, EventCallback cb) 
        : fd(f), events(e), callback(std::move(cb)) {}
};

// 事件处理器抽象基类
class EventHandler {
public:
    virtual ~EventHandler() = default;
    virtual void handle_read(int fd) = 0;
    virtual void handle_write(int fd) = 0;
    virtual void handle_error(int fd) = 0;
    virtual void handle_hangup(int fd) = 0;
};

// 事件分发器抽象基类
class EventDispatcher {
public:
    virtual ~EventDispatcher() = default;
    
    // 添加事件监听
    virtual bool add_event(int fd, EventType events, EventCallback callback) = 0;
    
    // 修改事件监听
    virtual bool modify_event(int fd, EventType events, EventCallback callback) = 0;
    
    // 移除事件监听
    virtual bool remove_event(int fd) = 0;
    
    // 事件循环 - 阻塞等待事件
    virtual int wait_events(int timeout_ms = -1) = 0;
    
    // 处理就绪的事件
    virtual void dispatch_events() = 0;
    
    // 获取最大文件描述符数量
    virtual int get_max_events() const = 0;
    
    // 检查是否支持边缘触发
    virtual bool supports_edge_trigger() const = 0;
};

// 事件分发器工厂
class EventDispatcherFactory {
public:
    enum class Type {
        SELECT,
        POLL, 
        EPOLL,
        KQUEUE,
        AUTO  // 自动选择最优实现
    };
    
    static std::unique_ptr<EventDispatcher> create(Type type, int max_events = 1024);
    static Type get_best_available();
    static bool is_available(Type type);
};

// 简单的事件循环实现
class SimpleEventLoop {
public:
    explicit SimpleEventLoop(EventDispatcherFactory::Type type = EventDispatcherFactory::Type::AUTO, 
                           int max_events = 1024);
    ~SimpleEventLoop();
    
    // 添加事件
    bool add_event(int fd, EventType events, EventCallback callback);
    
    // 移除事件
    bool remove_event(int fd);
    
    // 运行事件循环
    void run();
    
    // 停止事件循环
    void stop();
    
    // 单次事件处理
    bool run_once(int timeout_ms = -1);
    
private:
    std::unique_ptr<EventDispatcher> dispatcher_;
    bool running_;
};

// 通用的客户端事件处理器
class ClientEventHandler : public EventHandler {
public:
    explicit ClientEventHandler(int client_fd);
    virtual ~ClientEventHandler();
    
    void handle_read(int fd) override;
    void handle_write(int fd) override;
    void handle_error(int fd) override;
    void handle_hangup(int fd) override;
    
    // 设置自定义处理函数
    void set_read_handler(std::function<void(int)> handler) { read_handler_ = std::move(handler); }
    void set_write_handler(std::function<void(int)> handler) { write_handler_ = std::move(handler); }
    void set_error_handler(std::function<void(int)> handler) { error_handler_ = std::move(handler); }
    void set_close_handler(std::function<void(int)> handler) { close_handler_ = std::move(handler); }
    
protected:
    int client_fd_;
    std::function<void(int)> read_handler_;
    std::function<void(int)> write_handler_;
    std::function<void(int)> error_handler_;
    std::function<void(int)> close_handler_;
};

#endif // EVENT_DISPATCHER_H