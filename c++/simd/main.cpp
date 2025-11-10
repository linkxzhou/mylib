#include "simd_strlen.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <cassert>
#include <iomanip>

using namespace std;
using namespace simd_strlen;

// 增强的测试函数
void test_correctness() {
    cout << "=== 正确性测试 ===\n";
    
    // 测试字符串
    vector<string> test_strings = {
        "",                           // 空字符串
        "a",                         // 单字符
        "hello",                     // 短字符串
        "world",                     // 另一个短字符串
        "这是一个测试字符串",           // 中文字符串
        string(15, 'x'),             // 15字符（未对齐）
        string(16, 'y'),             // 16字符（对齐）
        string(31, 'z'),             // 31字符
        string(32, 'w'),             // 32字符（AVX对齐）
        string(100, 'a'),            // 中等长度
        string(1000, 'b'),           // 长字符串
        string(4096, 'c')            // 很长字符串
    };
    
    bool all_passed = true;
    
    for (size_t i = 0; i < test_strings.size(); i++) {
        const auto& s = test_strings[i];
        size_t len_std = strlen_standard(s.c_str());
        size_t len_glibc = strlen_glibc(s.c_str());
        size_t len_simd = strlen_simd(s.c_str());
        size_t len_builtin = strlen(s.c_str());
        
        bool test_passed = (len_std == len_glibc && len_glibc == len_simd && len_simd == len_builtin);
        all_passed &= test_passed;
        
        cout << "测试 " << setw(2) << i+1 << " (长度 " << setw(4) << len_std << "): ";
        if (test_passed) {
            cout << "✓ 通过";
        } else {
            cout << "✗ 失败 - 标准:" << len_std << " glibc:" << len_glibc 
                 << " SIMD:" << len_simd << " builtin:" << len_builtin;
        }
        cout << "\n";
    }
    
    cout << "\n总体结果: " << (all_passed ? "✓ 所有测试通过" : "✗ 存在失败测试") << "\n\n";
}

// 简单性能对比
void quick_performance_test() {
    cout << "=== 快速性能测试 ===\n";
    
    const int iterations = 10000;
    vector<string> test_data;
    
    // 生成多种长度的测试数据
    for (int len : {16, 64, 256, 1024}) {
        for (int i = 0; i < 100; i++) {
            test_data.push_back(string(len, 'a' + (i % 26)));
        }
    }
    
    // 预热
    for (const auto& s : test_data) {
        volatile size_t len = strlen_simd(s.c_str());
        (void)len;
    }
    
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        for (const auto& s : test_data) {
            volatile size_t len = strlen_standard(s.c_str());
            (void)len;
        }
    }
    auto std_time = chrono::duration_cast<chrono::microseconds>(
        chrono::high_resolution_clock::now() - start).count();
    
    start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        for (const auto& s : test_data) {
            volatile size_t len = strlen_simd(s.c_str());
            (void)len;
        }
    }
    auto simd_time = chrono::duration_cast<chrono::microseconds>(
        chrono::high_resolution_clock::now() - start).count();
    
    cout << "标准实现: " << std_time << "μs\n";
    cout << "SIMD实现: " << simd_time << "μs\n";
    cout << "性能提升: " << fixed << setprecision(2) 
         << static_cast<double>(std_time) / simd_time << "x\n\n";
}

int main() {
    cout << "SIMD 字符串函数测试 (";
#ifdef SIMD_ARCH_ARM64
    cout << "ARM64 NEON";
#elif defined(SIMD_ARCH_X86_64)
    cout << "x86_64 SSE/AVX";
    #ifdef __AVX512F__
    cout << "/AVX-512";
    #endif
#else
    cout << "标准实现";
#endif
    cout << ")\n\n";
    
    // 显示编译器信息
    cout << "编译器: ";
#ifdef __clang__
    cout << "Clang " << __clang_major__ << "." << __clang_minor__;
#elif defined(__GNUC__)
    cout << "GCC " << __GNUC__ << "." << __GNUC_MINOR__;
#elif defined(_MSC_VER)
    cout << "MSVC " << _MSC_VER;
#else
    cout << "未知";
#endif
    cout << "\n\n";
    
    test_correctness();
    quick_performance_test();
    
    cout << "测试完成!\n";
    return 0;
}