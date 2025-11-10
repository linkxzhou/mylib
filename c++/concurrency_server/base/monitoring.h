#ifndef MONITORING_H
#define MONITORING_H

#include <string>
#include <memory>
#include <chrono>
#include <atomic>
#include <mutex>
#include <vector>
#include <unordered_map>
#include <functional>
#include <fstream>
#include <thread>
#include <queue>

// 日志级别
enum class LogLevel {
    TRACE = 0,
    DEBUG = 1,
    INFO = 2,
    WARN = 3,
    ERROR = 4,
    FATAL = 5
};

// 日志记录结构
struct LogRecord {
    std::chrono::system_clock::time_point timestamp;
    LogLevel level;
    std::string logger_name;
    std::string message;
    std::string file;
    int line;
    std::string function;
    std::thread::id thread_id;
};

// 日志格式化器
class LogFormatter {
public:
    virtual ~LogFormatter() = default;
    virtual std::string format(const LogRecord& record) = 0;
};

// 默认日志格式化器
class DefaultLogFormatter : public LogFormatter {
public:
    std::string format(const LogRecord& record) override;
    
private:
    std::string level_to_string(LogLevel level);
    std::string format_timestamp(const std::chrono::system_clock::time_point& tp);
};

// JSON日志格式化器
class JsonLogFormatter : public LogFormatter {
public:
    std::string format(const LogRecord& record) override;
};

// 日志输出器基类
class LogAppender {
public:
    virtual ~LogAppender() = default;
    virtual void append(const LogRecord& record) = 0;
    virtual void flush() = 0;
    
    void set_formatter(std::unique_ptr<LogFormatter> formatter) {
        formatter_ = std::move(formatter);
    }
    
    void set_level(LogLevel level) { level_ = level; }
    LogLevel get_level() const { return level_; }
    
protected:
    std::unique_ptr<LogFormatter> formatter_;
    LogLevel level_ = LogLevel::INFO;
};

// 控制台日志输出器
class ConsoleAppender : public LogAppender {
public:
    ConsoleAppender();
    void append(const LogRecord& record) override;
    void flush() override;
    
private:
    std::mutex mutex_;
};

// 文件日志输出器
class FileAppender : public LogAppender {
public:
    explicit FileAppender(const std::string& filename);
    ~FileAppender();
    
    void append(const LogRecord& record) override;
    void flush() override;
    
    // 日志轮转
    void set_max_file_size(size_t max_size) { max_file_size_ = max_size; }
    void set_max_files(int max_files) { max_files_ = max_files; }
    
private:
    std::string filename_;
    std::ofstream file_;
    std::mutex mutex_;
    size_t current_size_;
    size_t max_file_size_;
    int max_files_;
    
    void rotate_file();
};

// 异步日志输出器
class AsyncAppender : public LogAppender {
public:
    explicit AsyncAppender(std::unique_ptr<LogAppender> appender);
    ~AsyncAppender();
    
    void append(const LogRecord& record) override;
    void flush() override;
    
private:
    std::unique_ptr<LogAppender> appender_;
    std::queue<LogRecord> log_queue_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::thread worker_thread_;
    std::atomic<bool> stop_flag_;
    
    void worker_loop();
};

// 日志器
class Logger {
public:
    explicit Logger(const std::string& name);
    ~Logger();
    
    // 添加输出器
    void add_appender(std::unique_ptr<LogAppender> appender);
    
    // 设置日志级别
    void set_level(LogLevel level) { level_ = level; }
    LogLevel get_level() const { return level_; }
    
    // 日志记录方法
    void log(LogLevel level, const std::string& message, 
             const char* file = __builtin_FILE(), 
             int line = __builtin_LINE(),
             const char* function = __builtin_FUNCTION());
    
    void trace(const std::string& message, const char* file = __builtin_FILE(), int line = __builtin_LINE(), const char* function = __builtin_FUNCTION());
    void debug(const std::string& message, const char* file = __builtin_FILE(), int line = __builtin_LINE(), const char* function = __builtin_FUNCTION());
    void info(const std::string& message, const char* file = __builtin_FILE(), int line = __builtin_LINE(), const char* function = __builtin_FUNCTION());
    void warn(const std::string& message, const char* file = __builtin_FILE(), int line = __builtin_LINE(), const char* function = __builtin_FUNCTION());
    void error(const std::string& message, const char* file = __builtin_FILE(), int line = __builtin_LINE(), const char* function = __builtin_FUNCTION());
    void fatal(const std::string& message, const char* file = __builtin_FILE(), int line = __builtin_LINE(), const char* function = __builtin_FUNCTION());
    
    // 格式化日志
    template<typename... Args>
    void info_f(const std::string& format, Args&&... args) {
        info(format_string(format, std::forward<Args>(args)...));
    }
    
    template<typename... Args>
    void error_f(const std::string& format, Args&&... args) {
        error(format_string(format, std::forward<Args>(args)...));
    }
    
private:
    std::string name_;
    LogLevel level_;
    std::vector<std::unique_ptr<LogAppender>> appenders_;
    std::mutex mutex_;
    
    template<typename... Args>
    std::string format_string(const std::string& format, Args&&... args);
};

// 日志管理器
class LogManager {
public:
    static LogManager& instance();
    
    std::shared_ptr<Logger> get_logger(const std::string& name);
    void configure_from_file(const std::string& config_file);
    void shutdown();
    
private:
    LogManager() = default;
    std::unordered_map<std::string, std::shared_ptr<Logger>> loggers_;
    std::mutex mutex_;
};

// 性能指标类型
enum class MetricType {
    COUNTER,    // 计数器
    GAUGE,      // 仪表
    HISTOGRAM,  // 直方图
    TIMER       // 计时器
};

// 性能指标基类
class Metric {
public:
    explicit Metric(const std::string& name, MetricType type)
        : name_(name), type_(type) {}
    virtual ~Metric() = default;
    
    const std::string& name() const { return name_; }
    MetricType type() const { return type_; }
    
    virtual std::string to_string() const = 0;
    virtual void reset() = 0;
    
protected:
    std::string name_;
    MetricType type_;
};

// 计数器
class Counter : public Metric {
public:
    explicit Counter(const std::string& name)
        : Metric(name, MetricType::COUNTER), value_(0) {}
    
    void increment(uint64_t delta = 1) {
        value_.fetch_add(delta);
    }
    
    uint64_t value() const {
        return value_.load();
    }
    
    std::string to_string() const override;
    void reset() override { value_ = 0; }
    
private:
    std::atomic<uint64_t> value_;
};

// 仪表
class Gauge : public Metric {
public:
    explicit Gauge(const std::string& name)
        : Metric(name, MetricType::GAUGE), value_(0) {}
    
    void set(double value) {
        std::lock_guard<std::mutex> lock(mutex_);
        value_ = value;
    }
    
    void increment(double delta = 1.0) {
        std::lock_guard<std::mutex> lock(mutex_);
        value_ += delta;
    }
    
    void decrement(double delta = 1.0) {
        std::lock_guard<std::mutex> lock(mutex_);
        value_ -= delta;
    }
    
    double value() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return value_;
    }
    
    std::string to_string() const override;
    void reset() override {
        std::lock_guard<std::mutex> lock(mutex_);
        value_ = 0;
    }
    
private:
    mutable std::mutex mutex_;
    double value_;
};

// 直方图
class Histogram : public Metric {
public:
    explicit Histogram(const std::string& name, const std::vector<double>& buckets = {});
    
    void observe(double value);
    
    uint64_t count() const { return count_.load(); }
    double sum() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return sum_;
    }
    
    std::vector<uint64_t> bucket_counts() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return bucket_counts_;
    }
    
    std::string to_string() const override;
    void reset() override;
    
private:
    std::vector<double> buckets_;
    std::vector<uint64_t> bucket_counts_;
    std::atomic<uint64_t> count_;
    mutable std::mutex mutex_;
    double sum_;
};

// 计时器
class Timer : public Metric {
public:
    explicit Timer(const std::string& name);
    
    class ScopedTimer {
    public:
        explicit ScopedTimer(Timer& timer)
            : timer_(timer), start_(std::chrono::high_resolution_clock::now()) {}
        
        ~ScopedTimer() {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
            timer_.observe(duration.count() / 1000.0);  // 转换为毫秒
        }
        
    private:
        Timer& timer_;
        std::chrono::high_resolution_clock::time_point start_;
    };
    
    ScopedTimer scoped() {
        return ScopedTimer(*this);
    }
    
    void observe(double milliseconds);
    
    uint64_t count() const { return histogram_.count(); }
    double sum() const { return histogram_.sum(); }
    double average() const {
        uint64_t c = count();
        return c > 0 ? sum() / c : 0.0;
    }
    
    std::string to_string() const override;
    void reset() override { histogram_.reset(); }
    
private:
    Histogram histogram_;
};

// 性能监控器
class PerformanceMonitor {
public:
    static PerformanceMonitor& instance();
    
    // 注册指标
    std::shared_ptr<Counter> register_counter(const std::string& name);
    std::shared_ptr<Gauge> register_gauge(const std::string& name);
    std::shared_ptr<Histogram> register_histogram(const std::string& name, const std::vector<double>& buckets = {});
    std::shared_ptr<Timer> register_timer(const std::string& name);
    
    // 获取指标
    std::shared_ptr<Metric> get_metric(const std::string& name);
    std::vector<std::shared_ptr<Metric>> get_all_metrics();
    
    // 导出指标
    std::string export_prometheus();
    std::string export_json();
    
    // 重置所有指标
    void reset_all();
    
private:
    PerformanceMonitor() = default;
    std::unordered_map<std::string, std::shared_ptr<Metric>> metrics_;
    std::mutex mutex_;
};

// 系统监控器
class SystemMonitor {
public:
    struct SystemStats {
        double cpu_usage_percent = 0.0;
        double memory_usage_percent = 0.0;
        uint64_t memory_used_bytes = 0;
        uint64_t memory_total_bytes = 0;
        double disk_usage_percent = 0.0;
        uint64_t network_bytes_sent = 0;
        uint64_t network_bytes_received = 0;
        uint32_t open_file_descriptors = 0;
        uint32_t thread_count = 0;
    };
    
    SystemMonitor();
    ~SystemMonitor();
    
    void start_monitoring(std::chrono::seconds interval = std::chrono::seconds(10));
    void stop_monitoring();
    
    SystemStats get_current_stats() const;
    
    // 设置回调函数
    using StatsCallback = std::function<void(const SystemStats&)>;
    void set_stats_callback(StatsCallback callback) {
        stats_callback_ = std::move(callback);
    }
    
private:
    std::atomic<bool> monitoring_;
    std::unique_ptr<std::thread> monitor_thread_;
    mutable std::mutex stats_mutex_;
    SystemStats current_stats_;
    StatsCallback stats_callback_;
    
    void monitor_loop(std::chrono::seconds interval);
    SystemStats collect_system_stats();
};

// 便利宏定义
#define LOG_TRACE(logger, msg) logger->trace(msg, __FILE__, __LINE__, __FUNCTION__)
#define LOG_DEBUG(logger, msg) logger->debug(msg, __FILE__, __LINE__, __FUNCTION__)
#define LOG_INFO(logger, msg) logger->info(msg, __FILE__, __LINE__, __FUNCTION__)
#define LOG_WARN(logger, msg) logger->warn(msg, __FILE__, __LINE__, __FUNCTION__)
#define LOG_ERROR(logger, msg) logger->error(msg, __FILE__, __LINE__, __FUNCTION__)
#define LOG_FATAL(logger, msg) logger->fatal(msg, __FILE__, __LINE__, __FUNCTION__)

#define TIMER_SCOPE(timer) auto _scoped_timer = timer->scoped()

// 全局日志器
extern std::shared_ptr<Logger> g_logger;

// 初始化监控系统
void init_monitoring(const std::string& config_file = "");
void shutdown_monitoring();

#endif // MONITORING_H