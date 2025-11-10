#include "simd_strlen.h"
#include <chrono>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cstring>
#include <algorithm>
#include <memory>
#include <thread>

using namespace std;
using namespace std::chrono;
using namespace simd_strlen;

class AdvancedTimer {
public:
    AdvancedTimer() { reset(); }
    
    void reset() {
        start_time = high_resolution_clock::now();
    }
    
    double elapsed_ns() const {
        auto end_time = high_resolution_clock::now();
        return duration_cast<nanoseconds>(end_time - start_time).count();
    }
    
    double elapsed_us() const {
        return elapsed_ns() / 1000.0;
    }
    
    double elapsed_ms() const {
        return elapsed_us() / 1000.0;
    }
    
private:
    high_resolution_clock::time_point start_time;
};

// 统计信息结构
struct BenchmarkStats {
    double min_time = numeric_limits<double>::max();
    double max_time = 0.0;
    double total_time = 0.0;
    vector<double> times;
    
    void add_time(double time) {
        times.push_back(time);
        min_time = min(min_time, time);
        max_time = max(max_time, time);
        total_time += time;
    }
    
    double avg_time() const {
        return times.empty() ? 0.0 : total_time / times.size();
    }
    
    double median_time() const {
        if (times.empty()) return 0.0;
        auto sorted_times = times;
        sort(sorted_times.begin(), sorted_times.end());
        size_t n = sorted_times.size();
        return n % 2 == 0 ? 
            (sorted_times[n/2-1] + sorted_times[n/2]) / 2.0 :
            sorted_times[n/2];
    }
    
    double std_dev() const {
        if (times.size() < 2) return 0.0;
        double avg = avg_time();
        double sum_sq_diff = 0.0;
        for (double time : times) {
            double diff = time - avg;
            sum_sq_diff += diff * diff;
        }
        return sqrt(sum_sq_diff / (times.size() - 1));
    }
};

// 生成高质量测试数据
class TestDataGenerator {
public:
    TestDataGenerator(uint32_t seed = 42) : gen(seed) {}
    
    // 生成随机长度字符串
    vector<string> generate_random_strings(size_t count, size_t min_len, size_t max_len) {
        vector<string> strings;
        strings.reserve(count);
        
        uniform_int_distribution<size_t> len_dist(min_len, max_len);
        uniform_int_distribution<int> char_dist(33, 126); // 可打印字符
        
        for (size_t i = 0; i < count; i++) {
            size_t len = len_dist(gen);
            string s;
            s.reserve(len + 1);
            for (size_t j = 0; j < len; j++) {
                s += static_cast<char>(char_dist(gen));
            }
            strings.push_back(move(s));
        }
        
        return strings;
    }
    
    // 生成固定长度字符串
    vector<string> generate_fixed_strings(size_t count, size_t length) {
        vector<string> strings;
        strings.reserve(count);
        
        uniform_int_distribution<int> char_dist(33, 126);
        
        for (size_t i = 0; i < count; i++) {
            string s;
            s.reserve(length + 1);
            for (size_t j = 0; j < length; j++) {
                s += static_cast<char>(char_dist(gen));
            }
            strings.push_back(move(s));
        }
        
        return strings;
    }
    
    // 生成对齐测试数据
    vector<unique_ptr<char[]>> generate_aligned_strings(size_t count, size_t length, size_t alignment = 64) {
        vector<unique_ptr<char[]>> strings;
        strings.reserve(count);
        
        uniform_int_distribution<int> char_dist(33, 126);
        
        for (size_t i = 0; i < count; i++) {
            // 分配对齐内存
            auto ptr = make_unique<char[]>(length + alignment);
            char* aligned_ptr = reinterpret_cast<char*>(
                (reinterpret_cast<uintptr_t>(ptr.get()) + alignment - 1) & ~(alignment - 1)
            );
            
            for (size_t j = 0; j < length; j++) {
                aligned_ptr[j] = static_cast<char>(char_dist(gen));
            }
            aligned_ptr[length] = '\0';
            
            strings.push_back(move(ptr));
        }
        
        return strings;
    }
    
private:
    mutable mt19937 gen;
};

// 基准测试函数
template<typename Func>
BenchmarkStats run_benchmark(const string& name, Func func, const vector<string>& test_data, 
                            int iterations = 1000, int warmup_iterations = 100) {
    BenchmarkStats stats;
    
    // 预热
    for (int i = 0; i < warmup_iterations; i++) {
        for (const auto& s : test_data) {
            volatile size_t result = func(s.c_str());
            (void)result;
        }
    }
    
    // 实际测试
    for (int i = 0; i < iterations; i++) {
        AdvancedTimer timer;
        for (const auto& s : test_data) {
            volatile size_t result = func(s.c_str());
            (void)result;
        }
        stats.add_time(timer.elapsed_us());
    }
    
    return stats;
}

// 详细的性能测试
void detailed_benchmark() {
    cout << "\n=== 详细性能测试 ===\n";
    
    TestDataGenerator generator;
    vector<size_t> test_sizes = {8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    const int iterations = 10000;
    const int test_strings_per_size = 100;
    
    cout << setw(12) << "长度" << setw(15) << "标准(μs)" << setw(15) << "glibc(μs)" 
         << setw(12) << "SIMD(μs)" << setw(12) << "glibc提升" << setw(12) << "SIMD提升" << setw(15) << "吞吐量(GB/s)" << "\n";
    cout << string(85, '-') << "\n";
    
    for (size_t size : test_sizes) {
        auto test_strings = generator.generate_fixed_strings(test_strings_per_size, size);
        
        auto std_stats = run_benchmark("标准", strlen_standard, test_strings, iterations);
        auto glibc_stats = run_benchmark("glibc", strlen_glibc, test_strings, iterations);
        auto simd_stats = run_benchmark("SIMD", strlen_simd, test_strings, iterations);
        
        double glibc_speedup = std_stats.median_time() / glibc_stats.median_time();
        double simd_speedup = std_stats.median_time() / simd_stats.median_time();
        double throughput = (size * test_strings_per_size * iterations) / 
                           (simd_stats.median_time() * 1e-6) / (1024.0 * 1024.0 * 1024.0);
        
        cout << setw(8) << size 
             << setw(12) << fixed << setprecision(2) << std_stats.median_time()
             << setw(12) << glibc_stats.median_time()
             << setw(12) << simd_stats.median_time()
             << setw(12) << setprecision(2) << glibc_speedup << "x"
             << setw(12) << setprecision(2) << simd_speedup << "x"
             << setw(15) << setprecision(2) << throughput << "\n";
    }
}

// 对齐性能测试
void alignment_benchmark() {
    cout << "\n=== 内存对齐性能测试 ===\n";
    
    TestDataGenerator generator;
    const size_t string_length = 1024;
    const int iterations = 10000;
    const int test_count = 100;
    
    vector<size_t> alignments = {1, 4, 8, 16, 32, 64};
    
    cout << setw(8) << "对齐" << setw(15) << "SIMD时间(μs)" << setw(12) << "相对性能" << "\n";
    cout << string(35, '-') << "\n";
    
    double baseline_time = 0.0;
    
    for (size_t alignment : alignments) {
        auto aligned_strings = generator.generate_aligned_strings(test_count, string_length, alignment);
        
        // 转换为字符串向量进行测试
        vector<string> test_strings;
        for (const auto& ptr : aligned_strings) {
            char* aligned_ptr = reinterpret_cast<char*>(
                (reinterpret_cast<uintptr_t>(ptr.get()) + alignment - 1) & ~(alignment - 1)
            );
            test_strings.emplace_back(aligned_ptr);
        }
        
        auto stats = run_benchmark("SIMD", strlen_simd, test_strings, iterations);
        double median_time = stats.median_time();
        
        if (alignment == alignments[0]) {
            baseline_time = median_time;
        }
        
        double relative_perf = baseline_time / median_time;
        
        cout << setw(8) << alignment 
             << setw(15) << fixed << setprecision(2) << median_time
             << setw(12) << setprecision(2) << relative_perf << "x\n";
    }
}

// 多线程性能测试
void multithreaded_benchmark() {
    cout << "\n=== 多线程性能测试 ===\n";
    
    TestDataGenerator generator;
    auto test_strings = generator.generate_random_strings(10000, 100, 2000);
    
    vector<int> thread_counts = {1, 2, 4, 8};
    const int iterations_per_thread = 100;
    
    cout << setw(8) << "线程数" << setw(15) << "总时间(ms)" << setw(12) << "加速比" << "\n";
    cout << string(35, '-') << "\n";
    
    double single_thread_time = 0.0;
    
    for (int thread_count : thread_counts) {
        AdvancedTimer timer;
        
        vector<thread> threads;
        for (int t = 0; t < thread_count; t++) {
            threads.emplace_back([&test_strings, iterations_per_thread]() {
                for (int i = 0; i < iterations_per_thread; i++) {
                    for (const auto& s : test_strings) {
                        volatile size_t len = strlen_simd(s.c_str());
                        (void)len;
                    }
                }
            });
        }
        
        for (auto& t : threads) {
            t.join();
        }
        
        double total_time = timer.elapsed_ms();
        
        if (thread_count == 1) {
            single_thread_time = total_time;
        }
        
        double speedup = single_thread_time / total_time;
        
        cout << setw(8) << thread_count 
             << setw(15) << fixed << setprecision(2) << total_time
             << setw(12) << setprecision(2) << speedup << "x\n";
    }
}

int main() {
    cout << "SIMD 字符串函数高级性能测试\n";
    cout << "==============================\n";
    
    cout << "当前架构: ";
#ifdef SIMD_ARCH_ARM64
    cout << "ARM64 (Apple Silicon)";
#elif defined(SIMD_ARCH_X86_64)
    cout << "x86_64 (Intel/AMD)";
    #ifdef __AVX512F__
    cout << " + AVX-512";
    #elif defined(__AVX2__)
    cout << " + AVX2";
    #elif defined(__SSE2__)
    cout << " + SSE2";
    #endif
#else
    cout << "未知架构";
#endif
    cout << "\n";
    
    cout << "CPU核心数: " << thread::hardware_concurrency() << "\n";
    
    detailed_benchmark();
    alignment_benchmark();
    multithreaded_benchmark();
    
    cout << "\n测试完成!\n";
    return 0;
}