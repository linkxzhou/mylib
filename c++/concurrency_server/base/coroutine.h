#ifndef COROUTINE_H
#define COROUTINE_H

#include <memory>
#include <functional>
#include <vector>
#include <cstddef>

// Define _XOPEN_SOURCE for ucontext functions on macOS
#define _XOPEN_SOURCE
#if __APPLE__ && __MACH__
#include <ucontext.h>
#else
#include <sys/ucontext.h>
#endif

#if __APPLE__ && __MACH__
    #include <sys/ucontext.h>
#else
    #include <ucontext.h>
#endif

namespace coroutine {

enum class Status {
    DEAD,
    READY,
    RUNNING,
    SUSPEND
};

class Coroutine {
public:
    Coroutine(std::function<void()> func, size_t stack_size = 1024 * 1024);
    ~Coroutine();
    
    Status get_status() const { return status_; }
    ucontext_t& get_context() { return context_; }
    const ucontext_t& get_context() const { return context_; }
    
    void set_status(Status status) { status_ = status; }
    
    char* get_stack() { return stack_.get(); }
    size_t get_stack_size() const { return stack_size_; }
    
private:
    Status status_;
    ucontext_t context_;
    std::unique_ptr<char[]> stack_;
    size_t stack_size_;
    std::function<void()> func_;
    
    friend class Scheduler;
};

class Scheduler {
public:
    Scheduler();
    ~Scheduler();
    
    int create_coroutine(std::function<void()> func, size_t stack_size = 1024 * 1024);
    void resume(int id);
    void yield();
    
    Status status(int id) const;
    int running() const { return running_; }
    
private:
    std::vector<std::unique_ptr<Coroutine>> coroutines_;
    ucontext_t main_context_;
    int running_;
    size_t stack_size_;
    
    void save_stack(int id);
    void restore_stack(int id);
    static Scheduler* current_scheduler_;
    static void coroutine_entry();
};

} // namespace coroutine

#endif // COROUTINE_H