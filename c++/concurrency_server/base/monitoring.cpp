#include "monitoring.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <ctime>
#include <sys/resource.h>
#include <sys/sysinfo.h>
#include <unistd.h>
#include <fstream>

// 全局日志器
std::shared_ptr<Logger> g_logger;

// DefaultLogFormatter 实现
std::string DefaultLogFormatter::format(const LogRecord& record) {
    std::ostringstream oss;
    
    oss << "[" << format_timestamp(record.timestamp) << "] ";
    oss << "[" << level_to_string(record.level) << "] ";
    oss << "[" << record.logger_name << "] ";
    oss << record.message;
    
    if (!record.file.empty()) {
        oss << " (" << record.file << ":" << record.line << ")";
    }
    
    return oss.str();
}

std::string DefaultLogFormatter::level_to_string(LogLevel level) {
    switch (level) {
        case LogLevel::TRACE: return "TRACE";
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO:  return "INFO ";
        case LogLevel::WARN:  return "WARN ";
        case LogLevel::ERROR: return "ERROR";
        case LogLevel::FATAL: return "FATAL";
        default: return "UNKNOWN";
    }
}

std::string DefaultLogFormatter::format_timestamp(const std::chrono::system_clock::time_point& tp) {
    auto time_t = std::chrono::system_clock::to_time_t(tp);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(tp.time_since_epoch()) % 1000;
    
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    oss << "." << std::setfill('0') << std::setw(3) << ms.count();
    
    return oss.str();
}

// JsonLogFormatter 实现
std::string JsonLogFormatter::format(const LogRecord& record) {
    std::ostringstream oss;
    
    auto time_t = std::chrono::system_clock::to_time_t(record.timestamp);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(record.timestamp.time_since_epoch());
    
    oss << "{";
    oss << "\"timestamp\":\"" << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%S") << "." 
        << std::setfill('0') << std::setw(3) << (ms.count() % 1000) << "Z\",";
    oss << "\"level\":\"" << DefaultLogFormatter().level_to_string(record.level) << "\",";
    oss << "\"logger\":\"" << record.logger_name << "\",";
    oss << "\"message\":\"" << record.message << "\",";
    oss << "\"file\":\"" << record.file << "\",";
    oss << "\"line\":" << record.line << ",";
    oss << "\"function\":\"" << record.function << "\"";
    oss << "}";
    
    return oss.str();
}

// ConsoleAppender 实现
ConsoleAppender::ConsoleAppender() {
    formatter_ = std::make_unique<DefaultLogFormatter>();
}

void ConsoleAppender::append(const LogRecord& record) {
    if (record.level < level_) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (record.level >= LogLevel::ERROR) {
        std::cerr << formatter_->format(record) << std::endl;
    } else {
        std::cout << formatter_->format(record) << std::endl;
    }
}

void ConsoleAppender::flush() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::cout.flush();
    std::cerr.flush();
}

// FileAppender 实现
FileAppender::FileAppender(const std::string& filename)
    : filename_(filename), current_size_(0), max_file_size_(100 * 1024 * 1024), max_files_(10) {
    
    formatter_ = std::make_unique<DefaultLogFormatter>();
    file_.open(filename_, std::ios::app);
    
    if (!file_.is_open()) {
        throw std::runtime_error("Cannot open log file: " + filename_);
    }
    
    // 获取当前文件大小
    file_.seekp(0, std::ios::end);
    current_size_ = file_.tellp();
}

FileAppender::~FileAppender() {
    if (file_.is_open()) {
        file_.close();
    }
}

void FileAppender::append(const LogRecord& record) {
    if (record.level < level_) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::string formatted = formatter_->format(record);
    file_ << formatted << std::endl;
    current_size_ += formatted.length() + 1;
    
    if (current_size_ >= max_file_size_) {
        rotate_file();
    }
}

void FileAppender::flush() {
    std::lock_guard<std::mutex> lock(mutex_);
    file_.flush();
}

void FileAppender::rotate_file() {
    file_.close();
    
    // 轮转文件
    for (int i = max_files_ - 1; i > 0; --i) {
        std::string old_name = filename_ + "." + std::to_string(i);
        std::string new_name = filename_ + "." + std::to_string(i + 1);
        
        if (access(old_name.c_str(), F_OK) == 0) {
            rename(old_name.c_str(), new_name.c_str());
        }
    }
    
    // 重命名当前文件
    std::string backup_name = filename_ + ".1";
    rename(filename_.c_str(), backup_name.c_str());
    
    // 创建新文件
    file_.open(filename_, std::ios::out | std::ios::trunc);
    current_size_ = 0;
}

// AsyncAppender 实现
AsyncAppender::AsyncAppender(std::unique_ptr<LogAppender> appender)
    : appender_(std::move(appender)), stop_flag_(false) {
    
    worker_thread_ = std::thread([this]() {
        worker_loop();
    });
}

AsyncAppender::~AsyncAppender() {
    stop_flag_ = true;
    condition_.notify_all();
    
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
}

void AsyncAppender::append(const LogRecord& record) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    log_queue_.push(record);
    condition_.notify_one();
}

void AsyncAppender::flush() {
    // 等待队列清空
    std::unique_lock<std::mutex> lock(queue_mutex_);
    condition_.wait(lock, [this]() { return log_queue_.empty(); });
    
    if (appender_) {
        appender_->flush();
    }
}

void AsyncAppender::worker_loop() {
    while (!stop_flag_) {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        condition_.wait(lock, [this]() { return !log_queue_.empty() || stop_flag_; });
        
        while (!log_queue_.empty()) {
            LogRecord record = log_queue_.front();
            log_queue_.pop();
            lock.unlock();
            
            if (appender_) {
                appender_->append(record);
            }
            
            lock.lock();
        }
    }
    
    // 处理剩余的日志
    std::lock_guard<std::mutex> lock(queue_mutex_);
    while (!log_queue_.empty()) {
        if (appender_) {
            appender_->append(log_queue_.front());
        }
        log_queue_.pop();
    }
}

// Logger 实现
Logger::Logger(const std::string& name) : name_(name), level_(LogLevel::INFO) {
}

Logger::~Logger() {
    // 刷新所有appender
    for (auto& appender : appenders_) {
        appender->flush();
    }
}

void Logger::add_appender(std::unique_ptr<LogAppender> appender) {
    std::lock_guard<std::mutex> lock(mutex_);
    appenders_.push_back(std::move(appender));
}

void Logger::log(LogLevel level, const std::string& message, const char* file, int line, const char* function) {
    if (level < level_) return;
    
    LogRecord record;
    record.timestamp = std::chrono::system_clock::now();
    record.level = level;
    record.logger_name = name_;
    record.message = message;
    record.file = file ? file : "";
    record.line = line;
    record.function = function ? function : "";
    record.thread_id = std::this_thread::get_id();
    
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& appender : appenders_) {
        appender->append(record);
    }
}

void Logger::trace(const std::string& message, const char* file, int line, const char* function) {
    log(LogLevel::TRACE, message, file, line, function);
}

void Logger::debug(const std::string& message, const char* file, int line, const char* function) {
    log(LogLevel::DEBUG, message, file, line, function);
}

void Logger::info(const std::string& message, const char* file, int line, const char* function) {
    log(LogLevel::INFO, message, file, line, function);
}

void Logger::warn(const std::string& message, const char* file, int line, const char* function) {
    log(LogLevel::WARN, message, file, line, function);
}

void Logger::error(const std::string& message, const char* file, int line, const char* function) {
    log(LogLevel::ERROR, message, file, line, function);
}

void Logger::fatal(const std::string& message, const char* file, int line, const char* function) {
    log(LogLevel::FATAL, message, file, line, function);
}

template<typename... Args>
std::string Logger::format_string(const std::string& format, Args&&... args) {
    // 简化的格式化实现
    // 在实际项目中应该使用更完善的格式化库如fmt
    std::ostringstream oss;
    oss << format;
    return oss.str();
}

// LogManager 实现
LogManager& LogManager::instance() {
    static LogManager instance;
    return instance;
}

std::shared_ptr<Logger> LogManager::get_logger(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = loggers_.find(name);
    if (it != loggers_.end()) {
        return it->second;
    }
    
    auto logger = std::make_shared<Logger>(name);
    
    // 添加默认的控制台appender
    logger->add_appender(std::make_unique<ConsoleAppender>());
    
    loggers_[name] = logger;
    return logger;
}

void LogManager::configure_from_file(const std::string& config_file) {
    // 简化实现：从配置文件加载日志配置
    // 实际项目中应该支持XML、JSON等配置格式
}

void LogManager::shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    loggers_.clear();
}

// Counter 实现
std::string Counter::to_string() const {
    std::ostringstream oss;
    oss << name_ << ": " << value();
    return oss.str();
}

// Gauge 实现
std::string Gauge::to_string() const {
    std::ostringstream oss;
    oss << name_ << ": " << value();
    return oss.str();
}

// Histogram 实现
Histogram::Histogram(const std::string& name, const std::vector<double>& buckets)
    : Metric(name, MetricType::HISTOGRAM), buckets_(buckets), count_(0), sum_(0) {
    
    if (buckets_.empty()) {
        // 默认桶
        buckets_ = {0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0};
    }
    
    std::sort(buckets_.begin(), buckets_.end());
    bucket_counts_.resize(buckets_.size() + 1, 0);  // +1 for +Inf bucket
}

void Histogram::observe(double value) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    count_.fetch_add(1);
    sum_ += value;
    
    // 找到对应的桶
    for (size_t i = 0; i < buckets_.size(); ++i) {
        if (value <= buckets_[i]) {
            bucket_counts_[i]++;
            return;
        }
    }
    
    // +Inf bucket
    bucket_counts_.back()++;
}

std::string Histogram::to_string() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::ostringstream oss;
    oss << name_ << " {count: " << count() << ", sum: " << sum_ << "}";
    return oss.str();
}

void Histogram::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    count_ = 0;
    sum_ = 0;
    std::fill(bucket_counts_.begin(), bucket_counts_.end(), 0);
}

// Timer 实现
Timer::Timer(const std::string& name)
    : Metric(name, MetricType::TIMER), histogram_(name + "_duration") {
}

void Timer::observe(double milliseconds) {
    histogram_.observe(milliseconds);
}

std::string Timer::to_string() const {
    std::ostringstream oss;
    oss << name_ << " {count: " << count() << ", avg: " << average() << "ms}";
    return oss.str();
}

// PerformanceMonitor 实现
PerformanceMonitor& PerformanceMonitor::instance() {
    static PerformanceMonitor instance;
    return instance;
}

std::shared_ptr<Counter> PerformanceMonitor::register_counter(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = metrics_.find(name);
    if (it != metrics_.end()) {
        return std::dynamic_pointer_cast<Counter>(it->second);
    }
    
    auto counter = std::make_shared<Counter>(name);
    metrics_[name] = counter;
    return counter;
}

std::shared_ptr<Gauge> PerformanceMonitor::register_gauge(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = metrics_.find(name);
    if (it != metrics_.end()) {
        return std::dynamic_pointer_cast<Gauge>(it->second);
    }
    
    auto gauge = std::make_shared<Gauge>(name);
    metrics_[name] = gauge;
    return gauge;
}

std::shared_ptr<Histogram> PerformanceMonitor::register_histogram(const std::string& name, const std::vector<double>& buckets) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = metrics_.find(name);
    if (it != metrics_.end()) {
        return std::dynamic_pointer_cast<Histogram>(it->second);
    }
    
    auto histogram = std::make_shared<Histogram>(name, buckets);
    metrics_[name] = histogram;
    return histogram;
}

std::shared_ptr<Timer> PerformanceMonitor::register_timer(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = metrics_.find(name);
    if (it != metrics_.end()) {
        return std::dynamic_pointer_cast<Timer>(it->second);
    }
    
    auto timer = std::make_shared<Timer>(name);
    metrics_[name] = timer;
    return timer;
}

std::shared_ptr<Metric> PerformanceMonitor::get_metric(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = metrics_.find(name);
    return (it != metrics_.end()) ? it->second : nullptr;
}

std::vector<std::shared_ptr<Metric>> PerformanceMonitor::get_all_metrics() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<std::shared_ptr<Metric>> result;
    for (const auto& pair : metrics_) {
        result.push_back(pair.second);
    }
    
    return result;
}

std::string PerformanceMonitor::export_prometheus() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::ostringstream oss;
    
    for (const auto& pair : metrics_) {
        const auto& metric = pair.second;
        
        switch (metric->type()) {
            case MetricType::COUNTER:
                oss << "# TYPE " << metric->name() << " counter\n";
                oss << metric->name() << " " << std::dynamic_pointer_cast<Counter>(metric)->value() << "\n";
                break;
                
            case MetricType::GAUGE:
                oss << "# TYPE " << metric->name() << " gauge\n";
                oss << metric->name() << " " << std::dynamic_pointer_cast<Gauge>(metric)->value() << "\n";
                break;
                
            case MetricType::HISTOGRAM:
                {
                    auto hist = std::dynamic_pointer_cast<Histogram>(metric);
                    oss << "# TYPE " << metric->name() << " histogram\n";
                    oss << metric->name() << "_count " << hist->count() << "\n";
                    oss << metric->name() << "_sum " << hist->sum() << "\n";
                }
                break;
                
            case MetricType::TIMER:
                {
                    auto timer = std::dynamic_pointer_cast<Timer>(metric);
                    oss << "# TYPE " << metric->name() << " histogram\n";
                    oss << metric->name() << "_count " << timer->count() << "\n";
                    oss << metric->name() << "_sum " << timer->sum() << "\n";
                }
                break;
        }
        
        oss << "\n";
    }
    
    return oss.str();
}

std::string PerformanceMonitor::export_json() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::ostringstream oss;
    oss << "{\n";
    oss << "  \"metrics\": [\n";
    
    bool first = true;
    for (const auto& pair : metrics_) {
        if (!first) oss << ",\n";
        first = false;
        
        const auto& metric = pair.second;
        oss << "    {\n";
        oss << "      \"name\": \"" << metric->name() << "\",\n";
        oss << "      \"type\": \"";
        
        switch (metric->type()) {
            case MetricType::COUNTER: oss << "counter"; break;
            case MetricType::GAUGE: oss << "gauge"; break;
            case MetricType::HISTOGRAM: oss << "histogram"; break;
            case MetricType::TIMER: oss << "timer"; break;
        }
        
        oss << "\",\n";
        oss << "      \"value\": \"" << metric->to_string() << "\"\n";
        oss << "    }";
    }
    
    oss << "\n  ]\n";
    oss << "}";
    
    return oss.str();
}

void PerformanceMonitor::reset_all() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (auto& pair : metrics_) {
        pair.second->reset();
    }
}

// SystemMonitor 实现
SystemMonitor::SystemMonitor() : monitoring_(false) {
}

SystemMonitor::~SystemMonitor() {
    stop_monitoring();
}

void SystemMonitor::start_monitoring(std::chrono::seconds interval) {
    if (monitoring_) return;
    
    monitoring_ = true;
    monitor_thread_ = std::make_unique<std::thread>([this, interval]() {
        monitor_loop(interval);
    });
}

void SystemMonitor::stop_monitoring() {
    monitoring_ = false;
    
    if (monitor_thread_ && monitor_thread_->joinable()) {
        monitor_thread_->join();
        monitor_thread_.reset();
    }
}

SystemMonitor::SystemStats SystemMonitor::get_current_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return current_stats_;
}

void SystemMonitor::monitor_loop(std::chrono::seconds interval) {
    while (monitoring_) {
        SystemStats stats = collect_system_stats();
        
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            current_stats_ = stats;
        }
        
        if (stats_callback_) {
            stats_callback_(stats);
        }
        
        std::this_thread::sleep_for(interval);
    }
}

SystemMonitor::SystemStats SystemMonitor::collect_system_stats() {
    SystemStats stats;
    
    // 获取内存信息
    struct sysinfo si;
    if (sysinfo(&si) == 0) {
        stats.memory_total_bytes = si.totalram * si.mem_unit;
        stats.memory_used_bytes = (si.totalram - si.freeram) * si.mem_unit;
        stats.memory_usage_percent = (double)stats.memory_used_bytes / stats.memory_total_bytes * 100.0;
    }
    
    // 获取CPU使用率（简化实现）
    static unsigned long long last_idle = 0, last_total = 0;
    std::ifstream stat_file("/proc/stat");
    if (stat_file.is_open()) {
        std::string line;
        std::getline(stat_file, line);
        
        std::istringstream iss(line);
        std::string cpu;
        unsigned long long user, nice, system, idle, iowait, irq, softirq, steal;
        
        iss >> cpu >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal;
        
        unsigned long long total = user + nice + system + idle + iowait + irq + softirq + steal;
        unsigned long long total_diff = total - last_total;
        unsigned long long idle_diff = idle - last_idle;
        
        if (total_diff > 0) {
            stats.cpu_usage_percent = (double)(total_diff - idle_diff) / total_diff * 100.0;
        }
        
        last_total = total;
        last_idle = idle;
    }
    
    // 获取文件描述符数量
    std::ifstream fd_file("/proc/self/fd");
    if (fd_file.is_open()) {
        std::string line;
        uint32_t count = 0;
        while (std::getline(fd_file, line)) {
            count++;
        }
        stats.open_file_descriptors = count;
    }
    
    // 获取线程数量
    std::ifstream status_file("/proc/self/status");
    if (status_file.is_open()) {
        std::string line;
        while (std::getline(status_file, line)) {
            if (line.find("Threads:") == 0) {
                std::istringstream iss(line);
                std::string label;
                iss >> label >> stats.thread_count;
                break;
            }
        }
    }
    
    return stats;
}

// 全局函数实现
void init_monitoring(const std::string& config_file) {
    // 初始化全局日志器
    g_logger = LogManager::instance().get_logger("global");
    
    if (!config_file.empty()) {
        LogManager::instance().configure_from_file(config_file);
    }
    
    LOG_INFO(g_logger, "Monitoring system initialized");
}

void shutdown_monitoring() {
    if (g_logger) {
        LOG_INFO(g_logger, "Shutting down monitoring system");
    }
    
    LogManager::instance().shutdown();
    PerformanceMonitor::instance().reset_all();
    
    g_logger.reset();
}