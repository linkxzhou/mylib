// Define _XOPEN_SOURCE for ucontext functions on macOS
#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE
#endif

#include "coroutine.h"
#include <cassert>
#include <cstring>
#include <cstdlib>

namespace coroutine {

// Static member for scheduler access
Scheduler* Scheduler::current_scheduler_ = nullptr;

// Static function to wrap coroutine execution
void Scheduler::coroutine_entry() {
    if (!current_scheduler_ || current_scheduler_->running_ == -1) {
        return;
    }
    
    int id = current_scheduler_->running_;
    auto& coroutines = current_scheduler_->coroutines_;
    
    if (id >= 0 && id < static_cast<int>(coroutines.size())) {
        auto& coroutine = coroutines[id];
        if (coroutine && coroutine->func_) {
            coroutine->func_();
        }
        coroutine->set_status(Status::DEAD);
    }
    
    // Return to main context when coroutine finishes
    current_scheduler_->running_ = -1;
}

Coroutine::Coroutine(std::function<void()> func, size_t stack_size)
    : status_(Status::READY), stack_size_(stack_size), func_(std::move(func)) {
    stack_.reset(new char[stack_size_]);
    
    // Initialize ucontext
    if (getcontext(&context_) == -1) {
        // Handle error
        return;
    }
    
    context_.uc_stack.ss_sp = stack_.get();
    context_.uc_stack.ss_size = stack_size_;
    context_.uc_link = nullptr;
}

Coroutine::~Coroutine() = default;

Scheduler::Scheduler() : running_(-1), stack_size_(1024 * 1024) {
    current_scheduler_ = this;
    // Initialize main context
    getcontext(&main_context_);
}

Scheduler::~Scheduler() {
    if (current_scheduler_ == this) {
        current_scheduler_ = nullptr;
    }
}

int Scheduler::create_coroutine(std::function<void()> func, size_t stack_size) {
    std::unique_ptr<Coroutine> coroutine(new Coroutine(std::move(func), stack_size));
    
    int id = static_cast<int>(coroutines_.size());
    
    // Set up the coroutine context
    coroutine->context_.uc_link = &main_context_;
    makecontext(&coroutine->context_, Scheduler::coroutine_entry, 0);
    
    coroutines_.push_back(std::move(coroutine));
    return id;
}

void Scheduler::resume(int id) {
    if (id < 0 || id >= static_cast<int>(coroutines_.size())) {
        return;
    }
    
    auto& coroutine = coroutines_[id];
    if (!coroutine) {
        return;
    }
    
    Status status = coroutine->get_status();
    if (status == Status::DEAD) {
        return;
    }
    
    int prev_running = running_;
    running_ = id;
    coroutine->set_status(Status::RUNNING);
    
    // Use swapcontext to switch to the coroutine
    swapcontext(&main_context_, &coroutine->get_context());
    
    running_ = prev_running;
}

void Scheduler::yield() {
    if (running_ == -1) {
        return;
    }
    
    auto& coroutine = coroutines_[running_];
    if (!coroutine) {
        return;
    }
    
    coroutine->set_status(Status::SUSPEND);
    
    // Use swapcontext to switch back to main context
    swapcontext(&coroutine->get_context(), &main_context_);
}

Status Scheduler::status(int id) const {
    if (id < 0 || id >= static_cast<int>(coroutines_.size())) {
        return Status::DEAD;
    }
    
    const auto& coroutine = coroutines_[id];
    return coroutine ? coroutine->get_status() : Status::DEAD;
}

} // namespace coroutine