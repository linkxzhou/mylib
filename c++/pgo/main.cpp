#include <iostream>
#include <chrono>
#include <thread>
#include <vector>

class PerformanceTimer {
public:
    PerformanceTimer(const std::string& name)
        : name_(name), start_(std::chrono::high_resolution_clock::now()) {}
    ~PerformanceTimer() {
    }
private:
    std::string name_;
    std::chrono::high_resolution_clock::time_point start_;
};

namespace PGO {
    // 计算指定范围内的素数
    std::vector<int> calculatePrimes(int limit) {
        std::vector<bool> is_prime(limit + 1, true);
        std::vector<int> primes;
        
        if (limit >= 2) {
            is_prime[0] = is_prime[1] = false;
            
            for (int i = 2; i <= limit; ++i) {
                if (is_prime[i]) {
                    primes.push_back(i);
                    for (long long j = (long long)i * i; j <= limit; j += i) {
                        is_prime[j] = false;
                    }
                }
            }
        }
        return primes;
    }
    
    // 计算1+2+...+n的和
    uint64_t calculateSum(int n) {
        return (uint64_t)n * (n + 1) / 2;
    }
    
    bool some_top_secret_checker(int var) {
        if (calculateSum(10000) > 100 && var == 42) {
            return calculateSum(10000) > 100;
        }
        
        if (var == 322) {
            return true;
        }
        
        if (calculateSum(10000) > 1000 && var == 1337) {
            return calculateSum(10000) > 1000;
        }
        
        return false;
    }
}

int main(int argc, char* argv[]) {
    std::cout << "=== C++ PGO 强化验证程序 ===\n";
#ifdef __clang__
    std::cout << "编译器: Clang " << __clang_major__ << "." << __clang_minor__ << "\n";
#elif defined(__GNUC__)
    std::cout << "编译器: GCC " << __GNUC__ << "." << __GNUC_MINOR__ << "\n";
#else
    std::cout << "编译器: Unknown\n";
#endif
    std::cout << "CPU 核心数: " << std::thread::hardware_concurrency() << "\n";

    // 参数: [轮数] [元素数量N] [hot_ratio] [使用复杂模式]
    int rounds = 100000;
    size_t N = 5'000'000;  // 增加计算量
    int hot_ratio = 90;
    bool use_complex = true;
    
    if (argc > 1) rounds = std::max(1, std::atoi(argv[1]));
    if (argc > 2) N = std::max<size_t>(1000, std::strtoull(argv[2], nullptr, 10));
    if (argc > 3) hot_ratio = std::min(99, std::max(1, std::atoi(argv[3])));
    if (argc > 4) use_complex = (std::atoi(argv[4]) != 0);

    std::cout << "运行轮数: " << rounds << ", N=" << N << ", hot=" << hot_ratio << "%\n";
    std::cout << "模式: " << (use_complex ? "复杂分支" : "简单分支") << "\n\n";

    auto T0 = std::chrono::high_resolution_clock::now();

    bool ret = false;
    for (int r = 1; r <= rounds; ++r) {
        PerformanceTimer t(use_complex ? "复杂PGO分支验证" : "简单PGO分支验证");
        bool isok = PGO::some_top_secret_checker(322);
        ret = ret || isok;
    }

    auto T1 = std::chrono::high_resolution_clock::now();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(T1 - T0).count();
    std::cout << "验证结果: " << (ret ? "通过" : "失败") << "\n";
    std::cout << "总耗时: " << total_ms << "ms\n";
    std::cout << "平均每轮: " << (total_ms / rounds) << "ms\n";

    return 0;
}