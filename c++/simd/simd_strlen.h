#ifndef SIMD_STRLEN_H
#define SIMD_STRLEN_H

#include <cstring>
#include <cstdint>
#include <algorithm>

// 根据架构选择不同的SIMD头文件
#ifdef __aarch64__
    #include <arm_neon.h>
    #define SIMD_ARCH_ARM64
#elif defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>
    #define SIMD_ARCH_X86_64
#endif

namespace simd_strlen {

// ============= STRLEN 实现 =============

// 标准版本
inline size_t strlen_standard(const char* str) {
    if (!str) return 0;
    const char* s = str;
    while (*s) ++s;
    return s - str;
}

#ifdef SIMD_ARCH_ARM64
// ARM64 NEON版本 - 优化版
inline size_t strlen_neon(const char* str) {
    if (!str) return 0;
    const char* ptr = str;
    
    // 处理未对齐的前缀
    while (((uintptr_t)ptr & 15) && *ptr) {
        ptr++;
    }
    
    if (!*ptr) return ptr - str;
    
    // 16字节对齐后，使用NEON并行处理
    const uint8x16_t zero = vdupq_n_u8(0);
    
    while (true) {
        uint8x16_t chunk = vld1q_u8(reinterpret_cast<const uint8_t*>(ptr));
        uint8x16_t cmp = vceqq_u8(chunk, zero);
        
        // 使用更高效的方法检查零字节
        uint64x2_t paired = vreinterpretq_u64_u8(cmp);
        uint64_t combined = vgetq_lane_u64(paired, 0) | vgetq_lane_u64(paired, 1);
        
        if (combined) {
            // 使用位操作快速找到第一个零字节位置
            uint16x8_t cmp16 = vreinterpretq_u16_u8(cmp);
            uint64_t mask = vget_lane_u64(vreinterpret_u64_u16(vorr_u16(vget_low_u16(cmp16), vget_high_u16(cmp16))), 0);
            
            for (int i = 0; i < 16; i++) {
                if (ptr[i] == 0) {
                    return ptr - str + i;
                }
            }
        }
        ptr += 16;
    }
}
#endif

#ifdef SIMD_ARCH_X86_64
// SSE2版本 - 优化版
inline size_t strlen_sse2(const char* str) {
    if (!str) return 0;
    const char* ptr = str;
    
    // 处理未对齐的前缀
    while (((uintptr_t)ptr & 15) && *ptr) {
        ptr++;
    }
    
    if (!*ptr) return ptr - str;
    
    // 16字节对齐后，使用SSE2并行处理
    const __m128i zero = _mm_setzero_si128();
    
    while (true) {
        __m128i chunk = _mm_load_si128(reinterpret_cast<const __m128i*>(ptr));
        __m128i cmp = _mm_cmpeq_epi8(chunk, zero);
        int mask = _mm_movemask_epi8(cmp);
        
        if (mask) {
            return ptr - str + __builtin_ctz(static_cast<unsigned>(mask));
        }
        ptr += 16;
    }
}

// AVX2版本 - 优化版
inline size_t strlen_avx2(const char* str) {
    if (!str) return 0;
    const char* ptr = str;
    
    // 处理未对齐的前缀
    while (((uintptr_t)ptr & 31) && *ptr) {
        ptr++;
    }
    
    if (!*ptr) return ptr - str;
    
    const __m256i zero = _mm256_setzero_si256();
    
    while (true) {
        __m256i chunk = _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr));
        __m256i cmp = _mm256_cmpeq_epi8(chunk, zero);
        int mask = _mm256_movemask_epi8(cmp);
        
        if (mask) {
            return ptr - str + __builtin_ctz(static_cast<unsigned>(mask));
        }
        ptr += 32;
    }
}

// AVX-512版本 - 新增高性能版本
#ifdef __AVX512F__
inline size_t strlen_avx512(const char* str) {
    if (!str) return 0;
    const char* ptr = str;
    
    // 处理未对齐的前缀
    while (((uintptr_t)ptr & 63) && *ptr) {
        ptr++;
    }
    
    if (!*ptr) return ptr - str;
    
    while (true) {
        __m512i chunk = _mm512_load_si512(reinterpret_cast<const __m512i*>(ptr));
        __mmask64 mask = _mm512_cmpeq_epi8_mask(chunk, _mm512_setzero_si512());
        
        if (mask) {
            return ptr - str + __builtin_ctzll(mask);
        }
        ptr += 64;
    }
}
#endif
#endif

// 自适应SIMD接口 - 运行时检测最佳版本
inline size_t strlen_simd(const char* str) {
    if (!str) return 0;
    
#ifdef SIMD_ARCH_ARM64
    return strlen_neon(str);
#elif defined(SIMD_ARCH_X86_64)
    #ifdef __AVX512F__
    return strlen_avx512(str);
    #else
    return strlen_avx2(str);
    #endif
#else
    return strlen_standard(str);
#endif
}

// ============= GLIBC 风格实现 =============

// glibc 风格 strlen - 优化版
inline size_t strlen_glibc(const char *str) {
    if (!str) return 0;
    
    const char *char_ptr;
    const unsigned long int *longword_ptr;
    unsigned long int longword, himagic, lomagic;

    // 处理前几个字符直到对齐
    for (char_ptr = str; 
         ((unsigned long int) char_ptr & (sizeof(longword) - 1)) != 0; 
         ++char_ptr) {
        if (*char_ptr == '\0')
            return char_ptr - str;
    }

    // 转换为 longword 指针
    longword_ptr = reinterpret_cast<const unsigned long int *>(char_ptr);

    // 设置魔数用于并行零字节检测
    himagic = 0x80808080UL;
    lomagic = 0x01010101UL;
    if (sizeof(longword) > 4) {
        himagic = ((himagic << 32) | himagic);
        lomagic = ((lomagic << 32) | lomagic);
    }

    // 主循环：每次处理一个 longword
    for (;;) {
        longword = *longword_ptr++;

        // 使用位技巧检测零字节
        if (((longword - lomagic) & ~longword & himagic) != 0) {
            const char *cp = reinterpret_cast<const char *>(longword_ptr - 1);

            // 找到具体的零字节位置
            for (size_t i = 0; i < sizeof(longword); ++i) {
                if (cp[i] == 0) {
                    return cp - str + i;
                }
            }
        }
    }
}

// 函数指针类型定义
using strlen_func_t = size_t(*)(const char*);

// 获取最优实现
inline strlen_func_t get_optimal_strlen() {
#ifdef SIMD_ARCH_ARM64
    return strlen_neon;
#elif defined(SIMD_ARCH_X86_64)
    #ifdef __AVX512F__
    return strlen_avx512;
    #else
    return strlen_avx2;
    #endif
#else
    return strlen_glibc;
#endif
}

} // namespace simd_strlen

#endif // SIMD_STRLEN_H