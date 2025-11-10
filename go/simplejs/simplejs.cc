/*
 * SimpleJS - 轻量级JavaScript解释器
 * 
 * 这是一个用C语言实现的轻量级JavaScript解释器，支持基本的JavaScript语法和功能。
 * 主要特性：
 * - 支持基本数据类型：数字、字符串、布尔值、对象、函数等
 * - 支持变量声明和赋值
 * - 支持算术运算、逻辑运算、比较运算
 * - 支持函数定义和调用
 * - 支持对象字面量和属性访问
 * - 支持控制流语句：if/else、for循环、break/continue、return
 * - 内置垃圾回收机制
 * - 支持C函数绑定
 * 
 * 内存管理：
 * - 使用自定义的内存管理器，所有JS对象都存储在连续的内存块中
 * - 采用标记-清除垃圾回收算法
 * - 支持内存使用统计和监控
 * 
 * 值编码：
 * - 使用NaN-boxing技术将所有JS值编码为64位整数
 * - 数字直接存储为IEEE 754双精度浮点数
 * - 其他类型使用NaN的高位存储类型信息和数据
 */

// 编译优化设置：在支持的GCC编译器上启用O3优化（排除Apple平台和Arduino）
#if defined(__GNUC__) && !defined(JS_OPT) && !defined(ARDUINO_AVR_UNO) && \
    !defined(ARDUINO_AVR_NANO) && !defined(ARDUINO_AVR_PRO) &&            \
    !defined(__APPLE__)
#pragma GCC optimize("O3,inline")
#endif

// 标准C库头文件
#include <assert.h>    // 断言宏
#include <math.h>      // 数学函数
#include <stdarg.h>    // 可变参数
#include <stdio.h>     // 标准输入输出
#include <stdlib.h>    // 标准库函数
#include <string.h>    // 字符串处理

#include "simplejs.h"  // SimpleJS公共接口

// 配置宏定义
#ifndef JS_EXPR_MAX
#define JS_EXPR_MAX 20        // 表达式解析的最大递归深度
#endif

#ifndef JS_GC_THRESHOLD
#define JS_GC_THRESHOLD 0.75  // 垃圾回收触发阈值（内存使用率）
#endif

// 类型定义
typedef uint32_t jsoff_t;  // JS内存偏移量类型，用于索引JS内存中的位置

/**
 * JS解释器核心状态结构体
 * 包含解析器状态、内存管理、作用域管理等所有必要信息
 */
struct js {
  // 性能监控相关
  jsoff_t css;        // 观察到的最大C栈大小
  jsoff_t lwm;        // JS内存低水位标记：观察到的最小可用内存
  
  // 代码解析相关
  const char *code;   // 当前正在解析的代码片段
  char errmsg[33];    // 错误消息缓冲区
  uint8_t tok;        // 最后解析的令牌值
  uint8_t consumed;   // 标识最后解析的令牌是否已被消费
  uint8_t flags;      // 执行标志，见下面的F_*常量定义
  
  // 执行标志常量定义
#define F_NOEXEC 1U   // 只解析代码，不执行
#define F_LOOP 2U     // 当前在循环内部
#define F_CALL 4U     // 当前在函数调用内部
#define F_BREAK 8U    // 退出循环
#define F_RETURN 16U  // return语句已执行
  
  // 词法分析相关
  jsoff_t clen;       // 代码片段长度
  jsoff_t pos;        // 当前解析位置
  jsoff_t toff;       // 最后解析令牌的偏移量
  jsoff_t tlen;       // 最后解析令牌的长度
  
  // 垃圾回收相关
  jsoff_t nogc;       // 排除在垃圾回收之外的实体偏移量
  
  // 值存储
  jsval_t tval;       // 保存最后解析的数字或字符串字面量值
  jsval_t scope;      // 当前作用域
  
  // 内存管理
  uint8_t *mem;       // 可用的JS内存指针
  jsoff_t size;       // 内存总大小
  jsoff_t brk;        // 当前内存使用边界
  jsoff_t gct;        // 垃圾回收阈值，当brk > gct时触发GC
  jsoff_t maxcss;     // 允许的最大C栈大小
  void *cstk;         // js_eval()开始时的C栈指针
};

/**
 * JS内存布局说明
 * 
 * JS内存存储不同类型的实体：对象、属性、字符串等
 * 所有实体都紧密排列在缓冲区的开头部分
 * `brk`标记已使用内存的结束位置：
 *
 *    | 实体1   | 实体2  | .... | 实体N |         未使用内存           |
 *    |---------|--------|------|-------|------------------------------|
 *  js.mem                           js.brk                        js.size
 *
 * 每个实体都是4字节对齐的，因此最低2位用于存储实体类型
 * 对象:   8字节: 第一个属性的偏移量, 上级对象的偏移量
 * 属性:   8字节 + 值: 4字节下一个属性, 4字节键偏移量, N字节值
 * 字符串: 4xN字节: 4字节长度<<2, 4字节对齐的0结尾数据
 *
 * 如果导入了C函数，它们使用内存的上半部分作为栈来传递参数
 * 每个参数作为jsval_t被推送到内存顶部，js.size减少sizeof(jsval_t)即8字节
 * 函数返回时，js.size被恢复。因此js.size被用作栈指针
 */

// clang-format off
/**
 * 词法分析令牌类型枚举
 * 定义了JavaScript词法分析器识别的所有令牌类型
 */
enum { 
  // 基本令牌
  TOK_ERR,         // 错误令牌
  TOK_EOF,         // 文件结束
  TOK_IDENTIFIER,  // 标识符
  TOK_NUMBER,      // 数字字面量
  TOK_STRING,      // 字符串字面量
  TOK_SEMICOLON,   // 分号 ;
  TOK_LPAREN,      // 左括号 (
  TOK_RPAREN,      // 右括号 )
  TOK_LBRACE,      // 左大括号 {
  TOK_RBRACE,      // 右大括号 }
  
  // JavaScript关键字令牌（从50开始）
  TOK_BREAK = 50, TOK_CASE, TOK_CATCH, TOK_CLASS, TOK_CONST, TOK_CONTINUE,
  TOK_DEFAULT, TOK_DELETE, TOK_DO, TOK_ELSE, TOK_FINALLY, TOK_FOR, TOK_FUNC,
  TOK_IF, TOK_IN, TOK_INSTANCEOF, TOK_LET, TOK_NEW, TOK_RETURN, TOK_SWITCH,
  TOK_THIS, TOK_THROW, TOK_TRY, TOK_VAR, TOK_VOID, TOK_WHILE, TOK_WITH,
  TOK_YIELD, TOK_UNDEF, TOK_NULL, TOK_TRUE, TOK_FALSE,
  
  // JavaScript操作符令牌（从100开始）
  TOK_DOT = 100,   // 点操作符 .
  TOK_CALL,        // 函数调用
  TOK_POSTINC,     // 后置递增 ++
  TOK_POSTDEC,     // 后置递减 --
  TOK_NOT,         // 逻辑非 !
  TOK_TILDA,       // 按位非 ~
  TOK_TYPEOF,      // typeof操作符
  TOK_UPLUS,       // 一元加 +
  TOK_UMINUS,      // 一元减 -
  TOK_EXP,         // 幂运算 **
  TOK_MUL,         // 乘法 *
  TOK_DIV,         // 除法 /
  TOK_REM,         // 取余 %
  TOK_PLUS,        // 加法 +
  TOK_MINUS,       // 减法 -
  TOK_SHL,         // 左移 <<
  TOK_SHR,         // 右移 >>
  TOK_ZSHR,        // 无符号右移 >>>
  TOK_LT,          // 小于 <
  TOK_LE,          // 小于等于 <=
  TOK_GT,          // 大于 >
  TOK_GE,          // 大于等于 >=
  TOK_EQ,          // 等于 ===
  TOK_NE,          // 不等于 !==
  TOK_AND,         // 按位与 &
  TOK_XOR,         // 按位异或 ^
  TOK_OR,          // 按位或 |
  TOK_LAND,        // 逻辑与 &&
  TOK_LOR,         // 逻辑或 ||
  TOK_COLON,       // 冒号 :
  TOK_Q,           // 问号 ?
  TOK_ASSIGN,      // 赋值 =
  
  // 复合赋值操作符
  TOK_PLUS_ASSIGN, TOK_MINUS_ASSIGN, TOK_MUL_ASSIGN, TOK_DIV_ASSIGN,
  TOK_REM_ASSIGN, TOK_SHL_ASSIGN, TOK_SHR_ASSIGN, TOK_ZSHR_ASSIGN,
  TOK_AND_ASSIGN, TOK_XOR_ASSIGN, TOK_OR_ASSIGN,
  
  TOK_COMMA,       // 逗号 ,
};

/**
 * JavaScript值类型枚举
 * 定义了JavaScript中所有可能的值类型
 * 注意：T_OBJ, T_PROP, T_STR必须排在前面，这是内存布局函数的要求
 * 内存实体类型编码在2位中，因此类型值必须是0,1,2,3
 */
enum {
  // 内存实体类型（必须是0,1,2）
  T_OBJ,     // 对象类型
  T_PROP,    // 属性类型
  T_STR,     // 字符串类型
  
  // 其他JavaScript类型
  T_UNDEF,   // undefined类型
  T_NULL,    // null类型
  T_NUM,     // 数字类型
  T_BOOL,    // 布尔类型
  T_FUNC,    // JavaScript函数类型
  T_CODEREF, // 代码引用类型
  T_CFUNC,   // C函数类型
  T_ERR      // 错误类型
};

/**
 * 获取类型名称字符串
 * @param t 类型枚举值
 * @return 对应的类型名称字符串
 */
static const char *typestr(uint8_t t) {
  const char *names[] = { "object", "prop", "string", "undefined", "null",
                          "number", "boolean", "function", "coderef",
                          "cfunc", "err", "nan" };
  return (t < sizeof(names) / sizeof(names[0])) ? names[t] : "??";
}

/**
 * NaN-boxing值编码系统
 * 
 * 将JS值打包到uint64_t中，利用IEEE 754双精度浮点数的NaN表示
 * 64位双精度浮点数格式：1位符号，11位指数，52位尾数
 *
 * seeeeeee|eeeemmmm|mmmmmmmm|mmmmmmmm|mmmmmmmm|mmmmmmmm|mmmmmmmm|mmmmmmmm
 * 11111111|11110000|00000000|00000000|00000000|00000000|00000000|00000000 无穷大
 * 11111111|11111000|00000000|00000000|00000000|00000000|00000000|00000000 安静NaN
 *
 * 我们的编码格式：
 * 11111111|1111tttt|vvvvvvvv|vvvvvvvv|vvvvvvvv|vvvvvvvv|vvvvvvvv|vvvvvvvv
 *  NaN标记  |类型|  48位数据占位符：用于指针、字符串等
 *
 * 在64位平台上，指针实际只有48位，所以可以完全放入，
 * 前提是进行符号扩展
 */

// 双精度浮点数转换为jsval_t
static jsval_t tov(double d) { union { double d; jsval_t v; } u = {d}; return u.v; }

// jsval_t转换为双精度浮点数
static double tod(jsval_t v) { union { jsval_t v; double d; } u = {v}; return u.d; }

// 创建指定类型和数据的JS值
static jsval_t mkval(uint8_t type, uint64_t data) { 
  return ((jsval_t) 0x7ff0U << 48U) | ((jsval_t) (type) << 48) | (data & 0xffffffffffffUL); 
}

// 检查值是否为NaN（即非数字类型）
static bool is_nan(jsval_t v) { return (v >> 52U) == 0x7ffU; }

// 获取JS值的类型
static uint8_t vtype(jsval_t v) { 
  return is_nan(v) ? ((v >> 48U) & 15U) : (uint8_t) T_NUM; 
}

// 获取JS值的数据部分
static size_t vdata(jsval_t v) { 
  return (size_t) (v & ~((jsval_t) 0x7fffUL << 48U)); 
}

// 创建代码引用值（包含偏移量和长度）
static jsval_t mkcoderef(jsval_t off, jsoff_t len) { 
  return mkval(T_CODEREF, (off & 0xffffffU) | ((jsval_t)(len & 0xffffffU) << 24U)); 
}

// 从代码引用值中提取偏移量
static jsoff_t coderefoff(jsval_t v) { return v & 0xffffffU; }

// 从代码引用值中提取长度
static jsoff_t codereflen(jsval_t v) { return (v >> 24U) & 0xffffffU; }

// 字符处理工具函数
static uint8_t unhex(uint8_t c) { 
  // 将十六进制字符转换为数值
  return (c >= '0' && c <= '9') ? (uint8_t) (c - '0') : 
         (c >= 'a' && c <= 'f') ? (uint8_t) (c - 'W') : 
         (c >= 'A' && c <= 'F') ? (uint8_t) (c - '7') : 0; 
}

// 检查是否为空白字符
static bool is_space(int c) { 
  return c == ' ' || c == '\r' || c == '\n' || c == '\t' || c == '\f' || c == '\v'; 
}

// 检查是否为数字字符
static bool is_digit(int c) { return c >= '0' && c <= '9'; }

// 检查是否为十六进制数字字符
static bool is_xdigit(int c) { 
  return is_digit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'); 
}

// 检查是否为字母字符
static bool is_alpha(int c) { 
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); 
}

// 检查是否可以作为标识符开头字符
static bool is_ident_begin(int c) { 
  return c == '_' || c == '$' || is_alpha(c); 
}

// 检查是否可以作为标识符后续字符
static bool is_ident_continue(int c) { 
  return c == '_' || c == '$' || is_alpha(c) || is_digit(c); 
}

// 检查JS值是否为错误类型
static bool is_err(jsval_t v) { return vtype(v) == T_ERR; }

// 检查令牌是否为一元操作符
static bool is_unary(uint8_t tok) { 
  return tok >= TOK_POSTINC && tok <= TOK_UMINUS; 
}

// 检查令牌是否为赋值操作符
static bool is_assign(uint8_t tok) { 
  return (tok >= TOK_ASSIGN && tok <= TOK_OR_ASSIGN); 
}
// 内存操作函数
// 保存偏移量值到指定内存位置
static void saveoff(struct js *js, jsoff_t off, jsoff_t val) { 
  memcpy(&js->mem[off], &val, sizeof(val)); 
}

// 保存JS值到指定内存位置
static void saveval(struct js *js, jsoff_t off, jsval_t val) { 
  memcpy(&js->mem[off], &val, sizeof(val)); 
}

// 从指定内存位置加载偏移量值
static jsoff_t loadoff(struct js *js, jsoff_t off) { 
  jsoff_t v = 0; 
  assert(js->brk <= js->size);  // 确保内存边界有效
  memcpy(&v, &js->mem[off], sizeof(v)); 
  return v; 
}

// 将偏移量转换为长度（字符串长度计算）
static jsoff_t offtolen(jsoff_t off) { return (off >> 2) - 1; }

// 获取字符串值的长度
static jsoff_t vstrlen(struct js *js, jsval_t v) { 
  return offtolen(loadoff(js, (jsoff_t) vdata(v))); 
}

// 从指定内存位置加载JS值
static jsval_t loadval(struct js *js, jsoff_t off) { 
  jsval_t v = 0; 
  memcpy(&v, &js->mem[off], sizeof(v)); 
  return v; 
}

// 获取作用域的上级作用域
static jsval_t upper(struct js *js, jsval_t scope) { 
  return mkval(T_OBJ, loadoff(js, (jsoff_t) (vdata(scope) + sizeof(jsoff_t)))); 
}

// 将值对齐到32位边界
static jsoff_t align32(jsoff_t v) { return ((v + 3) >> 2) << 2; }

// 错误检查宏：如果值是错误类型，则跳转到done标签
#define CHECKV(_v) do { if (is_err(_v)) { res = (_v); goto done; } } while (0)

// 期望特定令牌宏：如果下一个令牌不是期望的类型，则返回解析错误
#define EXPECT(_tok, _e) do { \
  if (next(js) != _tok) { _e; return js_mkerr(js, "parse error"); }; \
  js->consumed = 1; \
} while (0)
// clang-format on

// 私有函数前向声明
static size_t tostr(struct js *js, jsval_t value, char *buf, size_t len);  // 值转字符串
static jsval_t js_expr(struct js *js);                                    // 表达式解析
static jsval_t js_stmt(struct js *js);                                    // 语句解析
static jsval_t do_op(struct js *, uint8_t op, jsval_t l, jsval_t r);     // 操作符执行

/**
 * 设置低水位标记（Low Water Mark）
 * 更新JS内存的最小可用内存记录和C栈的最大使用记录
 * 用于性能监控和内存使用统计
 * @param js JS解释器实例
 */
static void setlwm(struct js *js) {
  jsoff_t n = 0, css = 0;
  
  // 计算当前可用内存
  if (js->brk < js->size) n = js->size - js->brk;
  
  // 更新最小可用内存记录（低水位标记）
  if (js->lwm > n) js->lwm = n;
  
  // 计算当前C栈使用量
  if ((char *) js->cstk > (char *) &n)
    css = (jsoff_t) ((char *) js->cstk - (char *) &n);
  
  // 更新最大C栈使用记录
  if (css > js->css) js->css = css;
}

/**
 * 安全字符串复制函数
 * 将src复制到dst，防止缓冲区溢出，确保以0结尾
 * @param dst 目标缓冲区
 * @param dstlen 目标缓冲区长度
 * @param src 源字符串
 * @param srclen 源字符串长度
 * @return 实际复制的字节数
 */
static size_t cpy(char *dst, size_t dstlen, const char *src, size_t srclen) {
  size_t i = 0;
  // 复制字符，确保不超出任一缓冲区边界
  for (i = 0; i < dstlen && i < srclen && src[i] != 0; i++) dst[i] = src[i];
  // 确保目标字符串以0结尾
  if (dstlen > 0) dst[i < dstlen ? i : dstlen - 1] = '\0';
  return i;
}

/**
 * 将JS对象转换为字符串表示
 * 格式：{"key1":value1,"key2":value2,...}
 * @param js JS解释器实例
 * @param obj 要字符串化的对象值
 * @param buf 输出缓冲区
 * @param len 缓冲区长度
 * @return 写入的字符数
 */
static size_t strobj(struct js *js, jsval_t obj, char *buf, size_t len) {
  size_t n = cpy(buf, len, "{", 1);  // 开始大括号
  
  // 获取第一个属性的偏移量
  jsoff_t next = loadoff(js, (jsoff_t) vdata(obj)) & ~3U;
  
  // 遍历所有属性
  while (next < js->brk && next != 0) {
    // 加载属性的键偏移量和值
    jsoff_t koff = loadoff(js, next + (jsoff_t) sizeof(next));
    jsval_t val = loadval(js, next + (jsoff_t) (sizeof(next) + sizeof(koff)));
    
    // 添加逗号分隔符（第一个属性除外）
    n += cpy(buf + n, len - n, ",", n == 1 ? 0 : 1);
    
    // 添加键名（字符串格式）
    n += tostr(js, mkval(T_STR, koff), buf + n, len - n);
    
    // 添加冒号分隔符
    n += cpy(buf + n, len - n, ":", 1);
    
    // 添加值的字符串表示
    n += tostr(js, val, buf + n, len - n);
    
    // 移动到下一个属性
    next = loadoff(js, next) & ~3U;
  }
  
  // 结束大括号
  return n + cpy(buf + n, len - n, "}", 1);
}

/**
 * 将JS数字值转换为字符串
 * @param value 数字值
 * @param buf 输出缓冲区
 * @param len 缓冲区长度
 * @return 写入的字符数
 */
static size_t strnum(jsval_t value, char *buf, size_t len) {
  double dv = tod(value), iv;
  // 如果是整数，使用高精度格式；否则使用标准格式
  const char *fmt = modf(dv, &iv) == 0.0 ? "%.17g" : "%g";
  return (size_t) snprintf(buf, len, fmt, dv);
}

/**
 * 获取JS字符串的内存偏移量和长度
 * @param js JS解释器实例
 * @param value 字符串值
 * @param len 输出参数：字符串长度
 * @return 字符串数据的内存偏移量
 */
static jsoff_t vstr(struct js *js, jsval_t value, jsoff_t *len) {
  jsoff_t off = (jsoff_t) vdata(value);
  if (len) *len = offtolen(loadoff(js, off));  // 计算字符串长度
  return (jsoff_t) (off + sizeof(off));        // 返回字符串数据偏移量
}

/**
 * 将JS字符串值转换为带引号的字符串表示
 * @param js JS解释器实例
 * @param value 字符串值
 * @param buf 输出缓冲区
 * @param len 缓冲区长度
 * @return 写入的字符数
 */
static size_t strstring(struct js *js, jsval_t value, char *buf, size_t len) {
  jsoff_t slen, off = vstr(js, value, &slen);
  size_t n = 0;
  n += cpy(buf + n, len - n, "\"", 1);                        // 开始引号
  n += cpy(buf + n, len - n, (char *) &js->mem[off], slen);   // 字符串内容
  n += cpy(buf + n, len - n, "\"", 1);                        // 结束引号
  return n;
}

/**
 * 将JS函数值转换为字符串表示
 * @param js JS解释器实例
 * @param value 函数值
 * @param buf 输出缓冲区
 * @param len 缓冲区长度
 * @return 写入的字符数
 */
static size_t strfunc(struct js *js, jsval_t value, char *buf, size_t len) {
  jsoff_t sn, off = vstr(js, value, &sn);
  size_t n = cpy(buf, len, "function", 8);                    // "function"前缀
  return n + cpy(buf + n, len - n, (char *) &js->mem[off], sn); // 函数体
}

/**
 * 创建错误值并设置错误消息
 * @param js JS解释器实例
 * @param xx 错误消息格式字符串
 * @param ... 格式化参数
 * @return 错误类型的JS值
 */
jsval_t js_mkerr(struct js *js, const char *xx, ...) {
  va_list ap;
  // 添加"ERROR: "前缀
  size_t n = cpy(js->errmsg, sizeof(js->errmsg), "ERROR: ", 7);
  
  // 格式化错误消息
  va_start(ap, xx);
  vsnprintf(js->errmsg + n, sizeof(js->errmsg) - n, xx, ap);
  va_end(ap);
  
  // 确保字符串以0结尾
  js->errmsg[sizeof(js->errmsg) - 1] = '\0';
  
  // 跳转到代码末尾，停止解析
  js->pos = js->clen, js->tok = TOK_EOF, js->consumed = 0;
  
  return mkval(T_ERR, 0);
}

/**
 * 将JS值转换为字符串表示
 * 这是主要的字符串化函数，根据值类型调用相应的转换函数
 * @param js JS解释器实例
 * @param value 要转换的JS值
 * @param buf 输出缓冲区
 * @param len 缓冲区长度
 * @return 写入的字符数
 */
static size_t tostr(struct js *js, jsval_t value, char *buf, size_t len) {
  switch (vtype(value)) {  // clang-format off
    case T_UNDEF: return cpy(buf, len, "undefined", 9);  // undefined类型
    case T_NULL:  return cpy(buf, len, "null", 4);       // null类型
    case T_BOOL:  return cpy(buf, len, vdata(value) & 1 ? "true" : "false", vdata(value) & 1 ? 4 : 5);  // 布尔类型
    case T_OBJ:   return strobj(js, value, buf, len);    // 对象类型
    case T_STR:   return strstring(js, value, buf, len); // 字符串类型
    case T_NUM:   return strnum(value, buf, len);        // 数字类型
    case T_FUNC:  return strfunc(js, value, buf, len);   // 函数类型
    case T_CFUNC: return (size_t) snprintf(buf, len, "\"c_func_0x%lx\"", (unsigned long) vdata(value));  // C函数类型
    case T_PROP:  return (size_t) snprintf(buf, len, "PROP@%lu", (unsigned long) vdata(value));  // 属性类型（调试用）
    default:      return (size_t) snprintf(buf, len, "VTYPE%d", vtype(value));  // 未知类型（调试用）
  }  // clang-format on
}

/**
 * 将JS值转换为字符串并存储在JS内存中
 * 这是一个便利函数，用于快速获取值的字符串表示
 * @param js JS解释器实例
 * @param value 要转换的JS值
 * @return 指向字符串的指针（存储在JS内存中）
 */
const char *js_str(struct js *js, jsval_t value) {
  // 在js->brk和字符串化缓冲区之间留出jsoff_t占位符，
  // 以防下一步需要将其转换为JS变量
  char *buf = (char *) &js->mem[js->brk + sizeof(jsoff_t)];
  size_t len, available = js->size - js->brk - sizeof(jsoff_t);
  
  // 如果是错误值，直接返回错误消息
  if (is_err(value)) return js->errmsg;
  
  // 检查是否有足够的内存空间
  if (js->brk + sizeof(jsoff_t) >= js->size) return "";
  
  // 转换为字符串
  len = tostr(js, value, buf, available);
  
  // 在JS内存中创建字符串对象
  js_mkstr(js, NULL, len);
  
  return buf;
}

/**
 * 判断JS值的真值性（truthiness）
 * 根据JavaScript的真值规则判断值是否为"真"
 * @param js JS解释器实例
 * @param v 要判断的JS值
 * @return true表示值为真，false表示值为假
 */
bool js_truthy(struct js *js, jsval_t v) {
  uint8_t t = vtype(v);
  return (t == T_BOOL && vdata(v) != 0) ||      // 布尔值true
         (t == T_NUM && tod(v) != 0.0) ||       // 非零数字
         (t == T_OBJ || t == T_FUNC) ||         // 对象和函数总是真值
         (t == T_STR && vstrlen(js, v) > 0);    // 非空字符串
}

/**
 * JS内存分配器
 * 在JS内存池中分配指定大小的内存块
 * @param js JS解释器实例
 * @param size 要分配的字节数
 * @return 分配的内存偏移量，失败时返回~0
 */
static jsoff_t js_alloc(struct js *js, size_t size) {
  jsoff_t ofs = js->brk;  // 当前分配位置
  
  // 4字节对齐：(n + k - 1) / k * k
  size = align32((jsoff_t) size);
  
  // 检查是否有足够的内存空间
  if (js->brk + size > js->size) return ~(jsoff_t) 0;
  
  // 更新内存使用边界
  js->brk += (jsoff_t) size;
  
  return ofs;
}

/**
 * 创建内存实体（对象、属性、字符串等）
 * 这是所有JS值创建的底层函数
 * @param js JS解释器实例
 * @param b 实体头部信息（包含类型和大小）
 * @param buf 实体数据缓冲区
 * @param len 数据长度
 * @return 创建的JS值
 */
static jsval_t mkentity(struct js *js, jsoff_t b, const void *buf, size_t len) {
  // 分配内存：头部 + 数据
  jsoff_t ofs = js_alloc(js, len + sizeof(b));
  if (ofs == (jsoff_t) ~0) return js_mkerr(js, "oom");  // 内存不足
  
  // 写入实体头部
  memcpy(&js->mem[ofs], &b, sizeof(b));
  
  // 写入实体数据（使用memmove以防数据来自JS内存本身）
  if (buf != NULL) memmove(&js->mem[ofs + sizeof(b)], buf, len);
  
  // 如果是字符串类型，确保以0结尾
  if ((b & 3) == T_STR) js->mem[ofs + sizeof(b) + len - 1] = 0;
  
  return mkval(b & 3, ofs);  // 返回JS值
}

/**
 * 创建JS字符串值
 * @param js JS解释器实例
 * @param ptr 字符串数据指针（可以为NULL）
 * @param len 字符串长度
 * @return 字符串类型的JS值
 */
jsval_t js_mkstr(struct js *js, const void *ptr, size_t len) {
  jsoff_t n = (jsoff_t) (len + 1);  // 包含结尾的0字符
  // 创建字符串实体：长度左移2位并加上类型标记
  return mkentity(js, (jsoff_t) ((n << 2) | T_STR), ptr, n);
}

/**
 * 创建JS对象
 * @param js JS解释器实例
 * @param parent 父对象的偏移量（用于作用域链）
 * @return 对象类型的JS值
 */
static jsval_t mkobj(struct js *js, jsoff_t parent) {
  // 创建对象实体：类型为T_OBJ，数据为父对象偏移量
  return mkentity(js, 0 | T_OBJ, &parent, sizeof(parent));
}

/**
 * 为对象设置属性
 * 在对象的属性链表头部插入新属性
 * @param js JS解释器实例
 * @param obj 目标对象
 * @param k 属性键（字符串值）
 * @param v 属性值
 * @return 新创建的属性值
 */
static jsval_t setprop(struct js *js, jsval_t obj, jsval_t k, jsval_t v) {
  jsoff_t koff = (jsoff_t) vdata(k);          // 键的偏移量
  jsoff_t b, head = (jsoff_t) vdata(obj);     // 属性链表头部
  char buf[sizeof(koff) + sizeof(v)];         // 属性内存布局缓冲区
  
  // 加载当前第一个属性的偏移量
  memcpy(&b, &js->mem[head], sizeof(b));
  
  // 初始化属性数据：复制键偏移量
  memcpy(buf, &koff, sizeof(koff));
  
  // 复制属性值
  memcpy(buf + sizeof(koff), &v, sizeof(v));
  
  // 新属性偏移量（当前brk位置）
  jsoff_t brk = js->brk | T_OBJ;
  
  // 将对象头部指向新属性（插入到链表头部）
  memcpy(&js->mem[head], &brk, sizeof(brk));
  
  // 创建新属性实体：下一个属性指向原来的第一个属性
  return mkentity(js, (b & ~3U) | T_PROP, buf, sizeof(buf));
}

/**
 * 根据内存中的第一个字计算T_OBJ/T_PROP/T_STR实体的大小
 * @param w 实体头部的第一个字
 * @return 实体的总大小（字节）
 */
static inline jsoff_t esize(jsoff_t w) {
  switch (w & 3U) {  // clang-format off
    case T_OBJ:   return (jsoff_t) (sizeof(jsoff_t) + sizeof(jsoff_t));  // 对象：两个偏移量
    case T_PROP:  return (jsoff_t) (sizeof(jsoff_t) + sizeof(jsoff_t) + sizeof(jsval_t));  // 属性：下一个属性+键+值
    case T_STR:   return (jsoff_t) (sizeof(jsoff_t) + align32(w >> 2U));  // 字符串：长度头+对齐的数据
    default:      return (jsoff_t) ~0U;  // 未知类型
  }  // clang-format on
}

/**
 * 检查类型是否为内存实体
 * 内存实体是存储在JS内存中的对象，需要参与垃圾回收
 * @param t 类型值
 * @return true表示是内存实体
 */
static bool is_mem_entity(uint8_t t) {
  return t == T_OBJ || t == T_PROP || t == T_STR || t == T_FUNC;
}

// 垃圾回收标记：用于标记要删除的实体
#define GCMASK ~(((jsoff_t) ~0) >> 1)  // 实体删除标记
static void js_fixup_offsets(struct js *js, jsoff_t start, jsoff_t size) {
  for (jsoff_t n, v, off = 0; off < js->brk; off += n) {  // start from 0!
    v = loadoff(js, off);
    n = esize(v & ~GCMASK);
    if (v & GCMASK) continue;  // To be deleted, don't bother
    if ((v & 3) != T_OBJ && (v & 3) != T_PROP) continue;
    if (v > start) saveoff(js, off, v - size);
    if ((v & 3) == T_OBJ) {
      jsoff_t u = loadoff(js, (jsoff_t) (off + sizeof(jsoff_t)));
      if (u > start) saveoff(js, (jsoff_t) (off + sizeof(jsoff_t)), u - size);
    }
    if ((v & 3) == T_PROP) {
      jsoff_t koff = loadoff(js, (jsoff_t) (off + sizeof(off)));
      if (koff > start) saveoff(js, (jsoff_t) (off + sizeof(off)), koff - size);
      jsval_t val = loadval(js, (jsoff_t) (off + sizeof(off) + sizeof(off)));
      if (is_mem_entity(vtype(val)) && vdata(val) > start) {
        saveval(js, (jsoff_t) (off + sizeof(off) + sizeof(off)),
                mkval(vtype(val), (unsigned long) (vdata(val) - size)));
      }
    }
  }
  // Fixup js->scope
  jsoff_t off = (jsoff_t) vdata(js->scope);
  if (off > start) js->scope = mkval(T_OBJ, off - size);
  if (js->nogc >= start) js->nogc -= size;
  // Fixup code that we're executing now, if required
  if (js->code > (char *) js->mem && js->code - (char *) js->mem < js->size &&
      js->code - (char *) js->mem > start) {
    js->code -= size;
    // printf("GC-ing code under us!! %ld\n", js->code - (char *) js->mem);
  }
  // printf("FIXEDOFF %u %u\n", start, size);
}

static void js_delete_marked_entities(struct js *js) {
  for (jsoff_t n, v, off = 0; off < js->brk; off += n) {
    v = loadoff(js, off);
    n = esize(v & ~GCMASK);
    if (v & GCMASK) {  // This entity is marked for deletion, remove it
      // printf("DEL: %4u %d %x\n", off, v & 3, n);
      // assert(off + n <= js->brk);
      js_fixup_offsets(js, off, n);
      memmove(&js->mem[off], &js->mem[off + n], js->brk - off - n);
      js->brk -= n;  // Shrink brk boundary by the size of deleted entity
      n = 0;         // We shifted data, next iteration stay on this offset
    }
  }
}

static void js_mark_all_entities_for_deletion(struct js *js) {
  for (jsoff_t v, off = 0; off < js->brk; off += esize(v)) {
    v = loadoff(js, off);
    saveoff(js, off, v | GCMASK);
  }
}

static jsoff_t js_unmark_entity(struct js *js, jsoff_t off) {
  jsoff_t v = loadoff(js, off);
  if (v & GCMASK) {
    saveoff(js, off, v & ~GCMASK);
    // printf("UNMARK %5u %d\n", off, v & 3);
    if ((v & 3) == T_OBJ) js_unmark_entity(js, v & ~(GCMASK | 3));
    if ((v & 3) == T_PROP) {
      js_unmark_entity(js, v & ~(GCMASK | 3));  // Unmark next prop
      js_unmark_entity(js, loadoff(js, (jsoff_t) (off + sizeof(off))));  // key
      jsval_t val = loadval(js, (jsoff_t) (off + sizeof(off) + sizeof(off)));
      if (is_mem_entity(vtype(val))) js_unmark_entity(js, (jsoff_t) vdata(val));
    }
  }
  return v & ~(GCMASK | 3U);
}

static void js_unmark_used_entities(struct js *js) {
  jsval_t scope = js->scope;
  do {
    js_unmark_entity(js, (jsoff_t) vdata(scope));
    scope = upper(js, scope);
  } while (vdata(scope) != 0);  // When global scope is GC-ed, stop
  if (js->nogc) js_unmark_entity(js, js->nogc);
  // printf("UNMARK: nogc %u\n", js->nogc);
  // js_dump(js);
}

/**
 * 垃圾回收主函数
 * 执行标记-清除垃圾回收算法
 * 1. 标记所有实体为待删除
 * 2. 取消标记仍在使用的实体
 * 3. 删除仍被标记的实体
 * @param js JS解释器实例
 */
void js_gc(struct js *js) {
  // 更新低水位标记
  setlwm(js);
  
  // 特殊情况：~0表示禁用垃圾回收
  if (js->nogc == (jsoff_t) ~0) return;
  
  // 第一阶段：标记所有实体为待删除
  js_mark_all_entities_for_deletion(js);
  
  // 第二阶段：取消标记仍在使用的实体
  js_unmark_used_entities(js);
  
  // 第三阶段：删除仍被标记的实体
  js_delete_marked_entities(js);
}

/**
 * 跳过空白字符和注释
 * 词法分析器的辅助函数，用于跳过不需要的字符
 * @param code 源代码字符串
 * @param len 代码长度
 * @param n 当前位置
 * @return 跳过空白和注释后的新位置
 */
static jsoff_t skiptonext(const char *code, jsoff_t len, jsoff_t n) {
  while (n < len) {
    if (is_space(code[n])) {
      // 跳过空白字符
      n++;
    } else if (n + 1 < len && code[n] == '/' && code[n + 1] == '/') {
      // 跳过单行注释 //
      for (n += 2; n < len && code[n] != '\n';) n++;
    } else if (n + 3 < len && code[n] == '/' && code[n + 1] == '*') {
      // 跳过多行注释 /* */
      for (n += 4; n < len && (code[n - 2] != '*' || code[n - 1] != '/');) n++;
    } else {
      // 遇到非空白、非注释字符，停止跳过
      break;
    }
  }
  return n;
}

/**
 * 字符串相等性比较
 * @param buf 第一个字符串
 * @param len 第一个字符串长度
 * @param p 第二个字符串
 * @param n 第二个字符串长度
 * @return true表示两个字符串相等
 */
static bool streq(const char *buf, size_t len, const char *p, size_t n) {
  return n == len && memcmp(buf, p, len) == 0;
}

/**
 * 解析JavaScript关键字
 * 根据标识符字符串判断是否为JavaScript关键字
 * @param buf 标识符字符串
 * @param len 字符串长度
 * @return 对应的令牌类型，如果不是关键字则返回TOK_IDENTIFIER
 */
static uint8_t parsekeyword(const char *buf, size_t len) {
  // 根据首字母进行快速分类，提高查找效率
  switch (buf[0]) {  // clang-format off
    case 'b': if (streq("break", 5, buf, len)) return TOK_BREAK; break;
    case 'c': 
      if (streq("class", 5, buf, len)) return TOK_CLASS; 
      if (streq("case", 4, buf, len)) return TOK_CASE; 
      if (streq("catch", 5, buf, len)) return TOK_CATCH; 
      if (streq("const", 5, buf, len)) return TOK_CONST; 
      if (streq("continue", 8, buf, len)) return TOK_CONTINUE; 
      break;
    case 'd': 
      if (streq("do", 2, buf, len)) return TOK_DO;  
      if (streq("default", 7, buf, len)) return TOK_DEFAULT; 
      break;
    case 'e': if (streq("else", 4, buf, len)) return TOK_ELSE; break;
    case 'f': 
      if (streq("for", 3, buf, len)) return TOK_FOR; 
      if (streq("function", 8, buf, len)) return TOK_FUNC; 
      if (streq("finally", 7, buf, len)) return TOK_FINALLY; 
      if (streq("false", 5, buf, len)) return TOK_FALSE; 
      break;
    case 'i': 
      if (streq("if", 2, buf, len)) return TOK_IF; 
      if (streq("in", 2, buf, len)) return TOK_IN; 
      if (streq("instanceof", 10, buf, len)) return TOK_INSTANCEOF; 
      break;
    case 'l': if (streq("let", 3, buf, len)) return TOK_LET; break;
    case 'n': 
      if (streq("new", 3, buf, len)) return TOK_NEW; 
      if (streq("null", 4, buf, len)) return TOK_NULL; 
      break;
    case 'r': if (streq("return", 6, buf, len)) return TOK_RETURN; break;
    case 's': if (streq("switch", 6, buf, len)) return TOK_SWITCH; break;
    case 't': 
      if (streq("try", 3, buf, len)) return TOK_TRY; 
      if (streq("this", 4, buf, len)) return TOK_THIS; 
      if (streq("throw", 5, buf, len)) return TOK_THROW; 
      if (streq("true", 4, buf, len)) return TOK_TRUE; 
      if (streq("typeof", 6, buf, len)) return TOK_TYPEOF; 
      break;
    case 'u': if (streq("undefined", 9, buf, len)) return TOK_UNDEF; break;
    case 'v': 
      if (streq("var", 3, buf, len)) return TOK_VAR; 
      if (streq("void", 4, buf, len)) return TOK_VOID; 
      break;
    case 'w': 
      if (streq("while", 5, buf, len)) return TOK_WHILE; 
      if (streq("with", 4, buf, len)) return TOK_WITH; 
      break;
    case 'y': if (streq("yield", 5, buf, len)) return TOK_YIELD; break;
  }  // clang-format on
  
  // 不是关键字，返回标识符令牌
  return TOK_IDENTIFIER;
}

/**
 * 解析标识符
 * 识别JavaScript标识符并判断是否为关键字
 * @param buf 输入缓冲区
 * @param len 缓冲区长度
 * @param tlen 输出参数：解析的标识符长度
 * @return 令牌类型（标识符或关键字）
 */
static uint8_t parseident(const char *buf, jsoff_t len, jsoff_t *tlen) {
  // 检查首字符是否可以作为标识符开头
  if (is_ident_begin(buf[0])) {
    // 继续读取标识符的后续字符
    while (*tlen < len && is_ident_continue(buf[*tlen])) (*tlen)++;
    
    // 检查是否为JavaScript关键字
    return parsekeyword(buf, *tlen);
  }
  
  // 不是有效的标识符
  return TOK_ERR;
}

// 词法分析器核心函数 - 解析下一个token
// 这是JavaScript解析器的核心，负责将源代码字符流转换为token流
static uint8_t next(struct js *js) {
  if (js->consumed == 0) return js->tok;  // 如果当前token未被消费，直接返回
  js->consumed = 0;  // 重置消费标志
  js->tok = TOK_ERR;  // 默认设置为错误token
  js->toff = js->pos = skiptonext(js->code, js->clen, js->pos);  // 跳过空白字符，更新位置
  js->tlen = 0;  // 重置token长度
  const char *buf = js->code + js->toff;  // 指向当前解析位置的缓冲区
  // clang-format off
  if (js->toff >= js->clen) { js->tok = TOK_EOF; return js->tok; }  // 到达代码末尾
#define TOK(T, LEN) { js->tok = T; js->tlen = (LEN); break; }  // 设置token类型和长度的宏
#define LOOK(OFS, CH) js->toff + OFS < js->clen && buf[OFS] == CH  // 前瞻字符检查宏
  switch (buf[0]) {  // 根据第一个字符确定token类型
    case '?': TOK(TOK_Q, 1);  // 三元运算符的问号
    case ':': TOK(TOK_COLON, 1);  // 冒号（对象属性分隔符、三元运算符）
    case '(': TOK(TOK_LPAREN, 1);  // 左圆括号
    case ')': TOK(TOK_RPAREN, 1);  // 右圆括号
    case '{': TOK(TOK_LBRACE, 1);  // 左花括号
    case '}': TOK(TOK_RBRACE, 1);  // 右花括号
    case ';': TOK(TOK_SEMICOLON, 1);  // 分号
    case ',': TOK(TOK_COMMA, 1);  // 逗号
    case '!': if (LOOK(1, '=') && LOOK(2, '=')) TOK(TOK_NE, 3); TOK(TOK_NOT, 1);  // 不等于(!==)或逻辑非(!)
    case '.': TOK(TOK_DOT, 1);  // 点号（属性访问）
    case '~': TOK(TOK_TILDA, 1);  // 按位取反运算符
    case '-': if (LOOK(1, '-')) TOK(TOK_POSTDEC, 2); if (LOOK(1, '=')) TOK(TOK_MINUS_ASSIGN, 2); TOK(TOK_MINUS, 1);  // 减法：--、-=、-
    case '+': if (LOOK(1, '+')) TOK(TOK_POSTINC, 2); if (LOOK(1, '=')) TOK(TOK_PLUS_ASSIGN, 2); TOK(TOK_PLUS, 1);  // 加法：++、+=、+
    case '*': if (LOOK(1, '*')) TOK(TOK_EXP, 2); if (LOOK(1, '=')) TOK(TOK_MUL_ASSIGN, 2); TOK(TOK_MUL, 1);  // 乘法：**、*=、*
    case '/': if (LOOK(1, '=')) TOK(TOK_DIV_ASSIGN, 2); TOK(TOK_DIV, 1);  // 除法：/=、/
    case '%': if (LOOK(1, '=')) TOK(TOK_REM_ASSIGN, 2); TOK(TOK_REM, 1);  // 取余：%=、%
    case '&': if (LOOK(1, '&')) TOK(TOK_LAND, 2); if (LOOK(1, '=')) TOK(TOK_AND_ASSIGN, 2); TOK(TOK_AND, 1);  // 位与/逻辑与：&&、&=、&
    case '|': if (LOOK(1, '|')) TOK(TOK_LOR, 2); if (LOOK(1, '=')) TOK(TOK_OR_ASSIGN, 2); TOK(TOK_OR, 1);  // 位或/逻辑或：||、|=、|
    case '=': if (LOOK(1, '=') && LOOK(2, '=')) TOK(TOK_EQ, 3); TOK(TOK_ASSIGN, 1);  // 等于/赋值：===、=
    case '<': if (LOOK(1, '<') && LOOK(2, '=')) TOK(TOK_SHL_ASSIGN, 3); if (LOOK(1, '<')) TOK(TOK_SHL, 2); if (LOOK(1, '=')) TOK(TOK_LE, 2); TOK(TOK_LT, 1);  // 小于/左移：<<=、<<、<=、<
    case '>': if (LOOK(1, '>') && LOOK(2, '=')) TOK(TOK_SHR_ASSIGN, 3); if (LOOK(1, '>')) TOK(TOK_SHR, 2); if (LOOK(1, '=')) TOK(TOK_GE, 2); TOK(TOK_GT, 1);  // 大于/右移：>>=、>>、>=、>
    case '^': if (LOOK(1, '=')) TOK(TOK_XOR_ASSIGN, 2); TOK(TOK_XOR, 1);  // 异或：^=、^
    case '"': case '\'':  // 字符串字面量解析（双引号或单引号）
      js->tlen++;  // 跳过开始的引号
      while (js->toff + js->tlen < js->clen && buf[js->tlen] != buf[0]) {  // 查找结束引号
        uint8_t increment = 1;  // 默认前进一个字符
        if (buf[js->tlen] == '\\') {  // 处理转义字符
          if (js->toff + js->tlen + 2 > js->clen) break;  // 防止越界
          increment = 2;  // 转义字符占两个字符
          if (buf[js->tlen + 1] == 'x') {  // 十六进制转义序列 \xNN
            if (js->toff + js->tlen + 4 > js->clen) break;  // 防止越界
            increment = 4;  // 十六进制转义占四个字符
          }
        }
        js->tlen += increment;  // 更新token长度
      }
      if (buf[0] == buf[js->tlen]) js->tok = TOK_STRING, js->tlen++;  // 找到结束引号，设置为字符串token
      break;
    case '0': case '1': case '2': case '3': case '4': case '5': case '6': case '7': case '8': case '9': {  // 数字字面量解析
      char *end;
      js->tval = tov(strtod(buf, &end)); // 使用strtod解析浮点数，TODO: 防止越界访问
      TOK(TOK_NUMBER, (jsoff_t) (end - buf));  // 设置数字token和长度
    }
    default: js->tok = parseident(buf, js->clen - js->toff, &js->tlen); break;  // 默认情况：解析标识符或关键字
  }  // clang-format on
  js->pos = js->toff + js->tlen;  // 更新解析位置到token结束处
  // printf("NEXT: %d %d [%.*s]\n", js->tok, js->pos, (int) js->tlen, buf);  // 调试输出
  return js->tok;  // 返回解析到的token类型
}

// 前瞻函数 - 查看下一个token但不消费当前token
// 用于需要预先知道下一个token类型的语法分析场景
static inline uint8_t lookahead(struct js *js) {
  uint8_t old = js->tok, tok = 0;  // 保存当前token
  jsoff_t pos = js->pos;  // 保存当前位置
  js->consumed = 1;  // 标记当前token已消费
  tok = next(js);  // 获取下一个token
  js->pos = pos, js->tok = old;  // 恢复原始状态
  return tok;  // 返回前瞻到的token类型
}

// 创建新作用域 - 用于函数调用、代码块等需要新变量作用域的场景
// 新作用域会链接到当前作用域，形成作用域链
static void mkscope(struct js *js) {
  assert((js->flags & F_NOEXEC) == 0);  // 确保不在非执行模式下
  jsoff_t prev = (jsoff_t) vdata(js->scope);  // 获取当前作用域的偏移量
  js->scope = mkobj(js, prev);  // 创建新对象作为新作用域，链接到前一个作用域
  // printf("ENTER SCOPE %u, prev %u\n", (jsoff_t) vdata(js->scope), prev);  // 调试输出
}

// 删除当前作用域 - 退出函数或代码块时恢复上一级作用域
// 通过作用域链向上回退一级
static void delscope(struct js *js) {
  js->scope = upper(js, js->scope);  // 恢复到上一级作用域
  // printf("EXIT  SCOPE %u\n", (jsoff_t) vdata(js->scope));  // 调试输出
}

// 代码块解析函数 - 解析由花括号包围的语句序列
// create_scope参数决定是否为此代码块创建新的变量作用域
static jsval_t js_block(struct js *js, bool create_scope) {
  jsval_t res = js_mkundef();  // 初始化返回值为undefined
  if (create_scope) mkscope(js);  // 如果需要，创建新作用域
  js->consumed = 1;  // 标记当前token已消费
  // jsoff_t pos = js->pos;  // 保存位置（调试用）
  // 循环解析代码块中的每个语句，直到遇到EOF、右花括号或错误
  while (next(js) != TOK_EOF && next(js) != TOK_RBRACE && !is_err(res)) {
    uint8_t t = js->tok;  // 保存当前token类型
    res = js_stmt(js);  // 解析一个语句
    // 检查语法：某些语句后必须有分号
    if (!is_err(res) && t != TOK_LBRACE && t != TOK_IF && t != TOK_WHILE &&
        js->tok != TOK_SEMICOLON) {
      res = js_mkerr(js, "; expected");  // 缺少分号错误
      break;
    }
  }
  // printf("BLOCKEND %s\n", js_str(js, res));  // 调试输出
  if (create_scope) delscope(js);  // 如果创建了作用域，则删除它
  return res;  // 返回最后一个语句的结果
}

// 在单个对象中查找属性 - 对象属性查找的核心函数
// 遍历对象的属性链表，查找指定名称的属性
static jsoff_t lkp(struct js *js, jsval_t obj, const char *buf, size_t len) {
  jsoff_t off = loadoff(js, (jsoff_t) vdata(obj)) & ~3U;  // 加载第一个属性的偏移量，清除类型标志位
  // printf("LKP: %lu %u [%.*s]\n", vdata(obj), off, (int) len, buf);  // 调试输出
  while (off < js->brk && off != 0) {  // 遍历属性链表，直到链表结束或到达内存边界
    jsoff_t koff = loadoff(js, (jsoff_t) (off + sizeof(off)));  // 加载属性键的偏移量
    jsoff_t klen = (loadoff(js, koff) >> 2) - 1;  // 计算属性键的长度
    const char *p = (char *) &js->mem[koff + sizeof(koff)];  // 获取属性键字符串的指针
    // printf("  %u %u[%.*s]\n", off, (int) klen, (int) klen, p);  // 调试输出
    if (streq(buf, len, p, klen)) return off;  // 找到匹配的属性，返回其偏移量
    off = loadoff(js, off) & ~3U;  // 加载下一个属性的偏移量
  }
  return 0;  // 未找到属性
}

// 在作用域链中查找变量 - JavaScript变量查找的核心实现
// 从当前作用域开始，沿着作用域链向上查找指定名称的变量
static jsval_t lookup(struct js *js, const char *buf, size_t len) {
  if (js->flags & F_NOEXEC) return 0;  // 如果在非执行模式下，直接返回
  for (jsval_t scope = js->scope;;) {  // 从当前作用域开始遍历作用域链
    jsoff_t off = lkp(js, scope, buf, len);  // 在当前作用域中查找变量
    if (off != 0) return mkval(T_PROP, off);  // 找到变量，返回属性引用
    if (vdata(scope) == 0) break;  // 到达全局作用域，停止查找
    // 移动到上一级作用域
    scope = mkval(T_OBJ, loadoff(js, (jsoff_t) (vdata(scope) + sizeof(jsoff_t))));
  }
  return js_mkerr(js, "'%.*s' not found", (int) len, buf);  // 变量未找到错误
}

// 解析属性引用 - 递归解析属性链，获取最终值
// 如果值不是属性引用，直接返回；否则递归解析
static jsval_t resolveprop(struct js *js, jsval_t v) {
  if (vtype(v) != T_PROP) return v;  // 如果不是属性引用，直接返回
  // 递归解析属性引用，获取属性的实际值
  return resolveprop(js, loadval(js, (jsoff_t) (vdata(v) + sizeof(jsoff_t) * 2)));
}

// 赋值操作 - 将值赋给属性引用
// 更新属性引用指向的内存位置的值
static jsval_t assign(struct js *js, jsval_t lhs, jsval_t val) {
  saveval(js, (jsoff_t) ((vdata(lhs) & ~3U) + sizeof(jsoff_t) * 2), val);  // 保存值到属性位置
  return lhs;  // 返回左值引用
}

// 复合赋值操作符实现 - 处理 +=、-=、*= 等操作
// 将复合赋值转换为对应的二元运算，然后执行赋值
static jsval_t do_assign_op(struct js *js, uint8_t op, jsval_t l, jsval_t r) {
  // 复合赋值操作符到基本运算符的映射表
  uint8_t m[] = {TOK_PLUS, TOK_MINUS, TOK_MUL, TOK_DIV, TOK_REM, TOK_SHL,
                 TOK_SHR,  TOK_ZSHR,  TOK_AND, TOK_XOR, TOK_OR};
  // 执行对应的二元运算：左值的当前值 op 右值
  jsval_t res = do_op(js, m[op - TOK_PLUS_ASSIGN], resolveprop(js, l), r);
  return assign(js, l, res);  // 将运算结果赋值给左值
}

// 字符串操作实现 - 处理字符串的加法、相等性比较等操作
// 支持字符串连接(+)、相等比较(===)、不等比较(!==)
static jsval_t do_string_op(struct js *js, uint8_t op, jsval_t l, jsval_t r) {
  jsoff_t n1, off1 = vstr(js, l, &n1);  // 获取左操作数字符串的偏移和长度
  jsoff_t n2, off2 = vstr(js, r, &n2);  // 获取右操作数字符串的偏移和长度
  if (op == TOK_PLUS) {  // 字符串连接操作
    jsval_t res = js_mkstr(js, NULL, n1 + n2);  // 创建新字符串，长度为两字符串长度之和
    // printf("STRPLUS %u %u %u %u [%.*s] [%.*s]\n", n1, off1, n2, off2, (int) n1,
    //       &js->mem[off1], (int) n2, &js->mem[off2]);  // 调试输出
    if (vtype(res) == T_STR) {  // 如果成功创建字符串
      jsoff_t n, off = vstr(js, res, &n);  // 获取新字符串的偏移
      memmove(&js->mem[off], &js->mem[off1], n1);  // 复制第一个字符串
      memmove(&js->mem[off + n1], &js->mem[off2], n2);  // 复制第二个字符串
    }
    return res;
  } else if (op == TOK_EQ) {  // 字符串相等比较
    bool eq = n1 == n2 && memcmp(&js->mem[off1], &js->mem[off2], n1) == 0;  // 长度相等且内容相同
    return mkval(T_BOOL, eq ? 1 : 0);
  } else if (op == TOK_NE) {  // 字符串不等比较
    bool eq = n1 == n2 && memcmp(&js->mem[off1], &js->mem[off2], n1) == 0;  // 长度相等且内容相同
    return mkval(T_BOOL, eq ? 0 : 1);  // 返回相反的结果
  } else {
    return js_mkerr(js, "bad str op");  // 不支持的字符串操作
  }
}

// 点操作符实现 - 处理对象属性访问 (obj.prop)
// 支持字符串的length属性和对象属性访问
static jsval_t do_dot_op(struct js *js, jsval_t l, jsval_t r) {
  const char *ptr = (char *) &js->code[coderefoff(r)];  // 获取属性名字符串指针
  if (vtype(r) != T_CODEREF) return js_mkerr(js, "ident expected");  // 右操作数必须是标识符
  // 特殊处理：字符串的length属性
  if (vtype(l) == T_STR && streq(ptr, codereflen(r), "length", 6)) {
    return tov(offtolen(loadoff(js, (jsoff_t) vdata(l))));  // 返回字符串长度
  }
  if (vtype(l) != T_OBJ) return js_mkerr(js, "lookup in non-obj");  // 左操作数必须是对象
  jsoff_t off = lkp(js, l, ptr, codereflen(r));  // 在对象中查找属性
  return off == 0 ? js_mkundef() : mkval(T_PROP, off);  // 返回属性引用或undefined
}

// 解析函数调用参数 - 解析函数调用的参数列表
// 在非执行模式下解析参数，返回参数列表的代码引用
static jsval_t js_call_params(struct js *js) {
  jsoff_t pos = js->pos;  // 保存当前位置
  uint8_t flags = js->flags;  // 保存当前标志
  js->flags |= F_NOEXEC;  // 设置非执行模式，只解析不执行
  js->consumed = 1;  // 标记token已消费
  for (bool comma = false; next(js) != TOK_EOF; comma = true) {  // 解析参数列表
    if (!comma && next(js) == TOK_RPAREN) break;  // 空参数列表
    js_expr(js);  // 解析一个参数表达式
    if (next(js) == TOK_RPAREN) break;  // 参数列表结束
    EXPECT(TOK_COMMA, js->flags = flags);  // 期望逗号分隔符
  }
  EXPECT(TOK_RPAREN, js->flags = flags);  // 期望右括号
  js->flags = flags;  // 恢复标志
  return mkcoderef(pos, js->pos - pos - js->tlen);  // 返回参数列表的代码引用
}

// 反转参数数组 - 用于调整参数顺序
// 将参数数组从后往前的顺序调整为从前往后
static void reverse(jsval_t *args, int nargs) {
  for (int i = 0; i < nargs / 2; i++) {  // 只需要交换一半的元素
    jsval_t tmp = args[i];  // 临时保存
    args[i] = args[nargs - i - 1], args[nargs - i - 1] = tmp;  // 交换对称位置的元素
  }
}

// 调用原生C函数 - 执行注册的C函数
// 解析参数，将其压入栈中，然后调用C函数
static jsval_t call_c(struct js *js,
                      jsval_t (*fn)(struct js *, jsval_t *, int)) {
  int argc = 0;  // 参数计数器
  // 解析并收集所有参数
  while (js->pos < js->clen) {
    if (next(js) == TOK_RPAREN) break;  // 参数列表结束
    jsval_t arg = resolveprop(js, js_expr(js));  // 解析并解引用参数
    if (js->brk + sizeof(arg) > js->size) return js_mkerr(js, "call oom");  // 检查内存溢出
    js->size -= (jsoff_t) sizeof(arg);  // 从栈顶分配空间
    memcpy(&js->mem[js->size], &arg, sizeof(arg));  // 将参数压入栈
    argc++;  // 增加参数计数
    // printf("  arg %d -> %s\n", argc, js_str(js, arg));  // 调试输出
    if (next(js) == TOK_COMMA) js->consumed = 1;  // 跳过逗号分隔符
  }
  reverse((jsval_t *) &js->mem[js->size], argc);  // 反转参数顺序，使其符合调用约定
  jsval_t res = fn(js, (jsval_t *) &js->mem[js->size], argc);  // 调用C函数
  setlwm(js);  // 更新低水位标记
  js->size += (jsoff_t) sizeof(jsval_t) * (jsoff_t) argc;  // 恢复栈指针，释放参数空间
  return res;  // 返回函数调用结果
}

// 调用JavaScript函数 - 执行用户定义的JS函数
// 函数格式如: "(a,b) { return a + b; }"
// 创建新作用域，解析参数，执行函数体
static jsval_t call_js(struct js *js, const char *fn, jsoff_t fnlen) {
  jsoff_t fnpos = 1;  // 跳过开始的左括号
  // printf("JSCALL [%.*s] -> %.*s\n", (int) js->clen, js->code, (int) fnlen, fn);  // 调试输出
  // printf("JSCALL, nogc %u [%.*s]\n", js->nogc, (int) fnlen, fn);  // 调试输出
  mkscope(js);  // 为函数调用创建新的作用域
  // 遍历参数列表 "(a, b)" 并在作用域中设置变量
  while (fnpos < fnlen) {
    fnpos = skiptonext(fn, fnlen, fnpos);  // 跳到标识符位置
    if (fnpos < fnlen && fn[fnpos] == ')') break;  // 遇到右括号则结束
    jsoff_t identlen = 0;  // 标识符长度
    uint8_t tok = parseident(&fn[fnpos], fnlen - fnpos, &identlen);  // 解析参数名
    if (tok != TOK_IDENTIFIER) break;  // 必须是标识符
    // 现在我们有了参数名，计算参数值
    // printf("  [%.*s] -> %u [%.*s] -> ", (int) identlen, &fn[fnpos], js->pos,
    //       (int) js->clen, js->code);  // 调试输出
    js->pos = skiptonext(js->code, js->clen, js->pos);  // 跳到下一个参数值
    js->consumed = 1;  // 标记token已消费
    jsval_t v = js->code[js->pos] == ')' ? js_mkundef() : js_expr(js);  // 解析参数值或使用undefined
    // 在函数作用域中设置参数
    setprop(js, js->scope, js_mkstr(js, &fn[fnpos], identlen), v);
    js->pos = skiptonext(js->code, js->clen, js->pos);  // 跳过空白
    if (js->pos < js->clen && js->code[js->pos] == ',') js->pos++;  // 跳过逗号
    fnpos = skiptonext(fn, fnlen, fnpos + identlen);  // 跳过当前标识符
    if (fnpos < fnlen && fn[fnpos] == ',') fnpos++;  // 跳过逗号
  }
  if (fnpos < fnlen && fn[fnpos] == ')') fnpos++;  // 跳到函数体
  fnpos = skiptonext(fn, fnlen, fnpos);  // 跳到左花括号
  if (fnpos < fnlen && fn[fnpos] == '{') fnpos++;  // 跳过左花括号
  size_t n = fnlen - fnpos - 1U;  // 去掉花括号后的函数代码长度
  // printf("flags: %d, body: %zu [%.*s]\n", js->flags, n, (int) n, &fn[fnpos]);  // 调试输出
  js->flags = F_CALL;  // 标记我们在函数调用中
  jsval_t res = js_eval(js, &fn[fnpos], n);  // 执行函数体，禁用GC
  if (!is_err(res) && !(js->flags & F_RETURN)) res = js_mkundef();  // 如果没有return语句，返回undefined
  delscope(js);  // 删除函数调用作用域
  // printf("  -> %d [%s], tok %d\n", js->flags, js_str(js, res), js->tok);  // 调试输出
  return res;  // 返回函数执行结果
}

// 函数调用操作实现 - 处理函数调用表达式
// 根据函数类型（JS函数或C函数）选择相应的调用方式
static jsval_t do_call_op(struct js *js, jsval_t func, jsval_t args) {
  if (vtype(args) != T_CODEREF) return js_mkerr(js, "bad call");  // 参数必须是代码引用
  if (vtype(func) != T_FUNC && vtype(func) != T_CFUNC)  // 被调用对象必须是函数
    return js_mkerr(js, "calling non-function");
  const char *code = js->code;  // 保存当前解析器状态
  jsoff_t clen = js->clen, pos = js->pos;  // 保存代码、位置和长度
  js->code = &js->code[coderefoff(args)];  // 将解析器指向参数
  js->clen = codereflen(args);  // 设置参数长度
  js->pos = skiptonext(js->code, js->clen, 0);  // 跳到第一个参数
  uint8_t tok = js->tok, flags = js->flags;  // 保存标志
  jsoff_t nogc = js->nogc;  // 保存GC状态
  jsval_t res = js_mkundef();  // 初始化返回值
  if (vtype(func) == T_FUNC) {  // 如果是JavaScript函数
    jsoff_t fnlen, fnoff = vstr(js, func, &fnlen);  // 获取函数字符串
    js->nogc = (jsoff_t) (fnoff - sizeof(jsoff_t));  // 设置GC保护
    res = call_js(js, (const char *) (&js->mem[fnoff]), fnlen);  // 调用JS函数
  } else {  // 如果是C函数
    res = call_c(js, (jsval_t(*)(struct js *, jsval_t *, int)) vdata(func));  // 调用C函数
  }
  js->code = code, js->clen = clen, js->pos = pos;  // 恢复解析器状态
  js->flags = flags, js->tok = tok, js->nogc = nogc;  // 恢复标志和GC状态
  js->consumed = 1;  // 标记token已消费
  return res;  // 返回函数调用结果
}

// 操作符执行核心函数 - 处理所有二元和一元操作符
// 这是JavaScript运算的核心，处理算术、逻辑、比较、赋值等所有操作
// clang-format off
static jsval_t do_op(struct js *js, uint8_t op, jsval_t lhs, jsval_t rhs) {
  if (js->flags & F_NOEXEC) return 0;  // 如果在非执行模式下，直接返回
  jsval_t l = resolveprop(js, lhs), r = resolveprop(js, rhs);  // 解析属性引用，获取实际值
  // printf("OP %d %d %d\n", op, vtype(lhs), vtype(r));  // 调试输出
  setlwm(js);  // 更新低水位标记
  if (is_err(l)) return l;  // 如果左操作数是错误，返回错误
  if (is_err(r)) return r;  // 如果右操作数是错误，返回错误
  if (is_assign(op) && vtype(lhs) != T_PROP) return js_mkerr(js, "bad lhs");  // 赋值操作的左值必须是属性引用
  switch (op) {  // 根据操作符类型进行分发
    case TOK_TYPEOF:  return js_mkstr(js, typestr(vtype(r)), strlen(typestr(vtype(r))));  // typeof操作符
    case TOK_CALL:    return do_call_op(js, l, r);  // 函数调用操作
    case TOK_ASSIGN:  return assign(js, lhs, r);  // 赋值操作
    case TOK_POSTINC: { do_assign_op(js, TOK_PLUS_ASSIGN, lhs, tov(1)); return l; }  // 后置递增
    case TOK_POSTDEC: { do_assign_op(js, TOK_MINUS_ASSIGN, lhs, tov(1)); return l; }  // 后置递减
    case TOK_NOT:     if (vtype(r) == T_BOOL) return mkval(T_BOOL, !vdata(r)); break;  // 逻辑非（布尔值）
  }
  if (is_assign(op))    return do_assign_op(js, op, lhs, r);  // 复合赋值操作
  if (vtype(l) == T_STR && vtype(r) == T_STR) return do_string_op(js, op, l, r);  // 字符串操作
  if (is_unary(op) && vtype(r) != T_NUM) return js_mkerr(js, "type mismatch");  // 一元操作数类型检查
  if (!is_unary(op) && op != TOK_DOT && (vtype(l) != T_NUM || vtype(r) != T_NUM)) return js_mkerr(js, "type mismatch");  // 二元操作数类型检查
  double a = tod(l), b = tod(r);  // 将操作数转换为双精度浮点数
  switch (op) {  // 数值运算操作符
    //case TOK_EXP:     return tov(pow(a, b));  // 幂运算（未实现）
    case TOK_DIV:     return tod(r) == 0 ? js_mkerr(js, "div by zero") : tov(a / b);  // 除法（检查除零）
    case TOK_REM:     return tov(a - b * ((double) (long) (a / b)));  // 取余运算
    case TOK_MUL:     return tov(a * b);  // 乘法
    case TOK_PLUS:    return tov(a + b);  // 加法
    case TOK_MINUS:   return tov(a - b);  // 减法
    case TOK_XOR:     return tov((double)((long) a ^ (long) b));  // 按位异或
    case TOK_AND:     return tov((double)((long) a & (long) b));  // 按位与
    case TOK_OR:      return tov((double)((long) a | (long) b));  // 按位或
    case TOK_UMINUS:  return tov(-b);  // 一元负号
    case TOK_UPLUS:   return r;  // 一元正号
    case TOK_TILDA:   return tov((double)(~(long) b));  // 按位取反
    case TOK_NOT:     return mkval(T_BOOL, b == 0);  // 逻辑非（数值）
    case TOK_SHL:     return tov((double)((long) a << (long) b));  // 左移
    case TOK_SHR:     return tov((double)((long) a >> (long) b));  // 右移
    case TOK_DOT:     return do_dot_op(js, l, r);  // 点操作符
    case TOK_EQ:      return mkval(T_BOOL, (long) a == (long) b);  // 相等比较
    case TOK_NE:      return mkval(T_BOOL, (long) a != (long) b);  // 不等比较
    case TOK_LT:      return mkval(T_BOOL, a < b);  // 小于比较
    case TOK_LE:      return mkval(T_BOOL, a <= b);  // 小于等于比较
    case TOK_GT:      return mkval(T_BOOL, a > b);  // 大于比较
    case TOK_GE:      return mkval(T_BOOL, a >= b);  // 大于等于比较
    default:          return js_mkerr(js, "unknown op %d", (int) op);  // 未知操作符错误
  }
}  // clang-format on

// 字符串字面量解析 - 处理字符串中的转义序列
// 将源代码中的字符串字面量转换为内存中的字符串对象
static jsval_t js_str_literal(struct js *js) {
  uint8_t *in = (uint8_t *) &js->code[js->toff];  // 输入字符串指针
  uint8_t *out = &js->mem[js->brk + sizeof(jsoff_t)];  // 输出缓冲区指针
  size_t n1 = 0, n2 = 0;  // n1: 输出长度, n2: 输入位置
  // printf("STR %u %lu %lu\n", js->brk, js->tlen, js->clen);  // 调试输出
  if (js->brk + sizeof(jsoff_t) + js->tlen > js->size)  // 检查内存是否足够
    return js_mkerr(js, "oom");
  while (n2++ + 2 < js->tlen) {  // 遍历字符串内容（跳过引号）
    if (in[n2] == '\\') {  // 处理转义字符
      if (in[n2 + 1] == in[0]) {  // 转义引号
        out[n1++] = in[0];
      } else if (in[n2 + 1] == 'n') {  // 换行符
        out[n1++] = '\n';
      } else if (in[n2 + 1] == 't') {  // 制表符
        out[n1++] = '\t';
      } else if (in[n2 + 1] == 'r') {  // 回车符
        out[n1++] = '\r';
      } else if (in[n2 + 1] == 'x' && is_xdigit(in[n2 + 2]) &&  // 十六进制转义 \xNN
                 is_xdigit(in[n2 + 3])) {
        out[n1++] = (uint8_t) ((unhex(in[n2 + 2]) << 4U) | unhex(in[n2 + 3]));  // 转换十六进制
        n2 += 2;  // 跳过额外的两个字符
      } else {
        return js_mkerr(js, "bad str literal");  // 无效的转义序列
      }
      n2++;  // 跳过转义字符
    } else {
      out[n1++] = ((uint8_t *) js->code)[js->toff + n2];  // 普通字符直接复制
    }
  }
  return js_mkstr(js, NULL, n1);  // 创建字符串对象
}

// 对象字面量解析 - 解析 {key: value, ...} 形式的对象
// 创建对象并设置其属性
static jsval_t js_obj_literal(struct js *js) {
  uint8_t exe = !(js->flags & F_NOEXEC);  // 检查是否在执行模式
  // printf("OLIT1\n");  // 调试输出
  jsval_t obj = exe ? mkobj(js, 0) : js_mkundef();  // 创建对象或返回undefined
  if (is_err(obj)) return obj;  // 检查对象创建是否成功
  js->consumed = 1;  // 标记token已消费
  while (next(js) != TOK_RBRACE) {  // 解析对象属性，直到遇到右花括号
    jsval_t key = 0;  // 属性键
    if (js->tok == TOK_IDENTIFIER) {  // 标识符作为键
      if (exe) key = js_mkstr(js, js->code + js->toff, js->tlen);
    } else if (js->tok == TOK_STRING) {  // 字符串作为键
      if (exe) key = js_str_literal(js);
    } else {
      return js_mkerr(js, "parse error");  // 无效的属性键
    }
    js->consumed = 1;  // 消费键token
    EXPECT(TOK_COLON, );  // 期望冒号分隔符
    jsval_t val = js_expr(js);  // 解析属性值表达式
    if (exe) {  // 如果在执行模式下
      // printf("XXXX [%s] scope: %lu\n", js_str(js, val), vdata(js->scope));  // 调试输出
      if (is_err(val)) return val;  // 检查值是否有错误
      if (is_err(key)) return key;  // 检查键是否有错误
      jsval_t res = setprop(js, obj, key, resolveprop(js, val));  // 设置对象属性
      if (is_err(res)) return res;  // 检查属性设置是否成功
    }
    if (next(js) == TOK_RBRACE) break;  // 如果遇到右花括号，结束解析
    EXPECT(TOK_COMMA, );  // 期望逗号分隔符
  }
  EXPECT(TOK_RBRACE, );  // 期望右花括号
  return obj;  // 返回创建的对象
}

// 函数字面量解析 - 解析 function(args) { body } 形式的函数
// 解析函数参数列表和函数体，创建函数对象
static jsval_t js_func_literal(struct js *js) {
  uint8_t flags = js->flags;  // 保存当前标志
  js->consumed = 1;  // 标记function关键字已消费
  EXPECT(TOK_LPAREN, js->flags = flags);  // 期望左括号
  jsoff_t pos = js->pos - 1;  // 保存函数开始位置
  for (bool comma = false; next(js) != TOK_EOF; comma = true) {  // 解析参数列表
    if (!comma && next(js) == TOK_RPAREN) break;  // 空参数列表
    EXPECT(TOK_IDENTIFIER, js->flags = flags);  // 期望参数名
    if (next(js) == TOK_RPAREN) break;  // 参数列表结束
    EXPECT(TOK_COMMA, js->flags = flags);  // 期望逗号分隔符
  }
  EXPECT(TOK_RPAREN, js->flags = flags);  // 期望右括号
  EXPECT(TOK_LBRACE, js->flags = flags);  // 期望左花括号
  js->consumed = 0;  // 重置消费标志
  js->flags |= F_NOEXEC;  // 设置非执行标志，只解析不执行函数体
  jsval_t res = js_block(js, false);  // 跳过函数体 - 不执行，只解析
  if (is_err(res)) {  // 如果解析出错，提前返回
    js->flags = flags;
    return res;
  }
  js->flags = flags;  // 恢复标志
  jsval_t str = js_mkstr(js, &js->code[pos], js->pos - pos);  // 创建包含整个函数定义的字符串
  js->consumed = 1;  // 标记token已消费
  // printf("FUNC: %u [%.*s]\n", pos, js->pos - pos, &js->code[pos]);  // 调试输出
  return mkval(T_FUNC, (unsigned long) vdata(str));  // 返回函数对象
}

// 右结合二元操作符宏 - 用于赋值等右结合操作符
// _f1: 左操作数解析函数, _f2: 右操作数解析函数, _cond: 继续条件
#define RTL_BINOP(_f1, _f2, _cond)  \
  jsval_t res = _f1(js);            \
  while (!is_err(res) && (_cond)) { \
    uint8_t op = js->tok;           \
    js->consumed = 1;               \
    jsval_t rhs = _f2(js);          \
    if (is_err(rhs)) return rhs;    \
    res = do_op(js, op, res, rhs);  \
  }                                 \
  return res;

// 左结合二元操作符宏 - 用于算术、比较等左结合操作符
// _f: 操作数解析函数, _cond: 继续条件
#define LTR_BINOP(_f, _cond)        \
  jsval_t res = _f(js);             \
  while (!is_err(res) && (_cond)) { \
    uint8_t op = js->tok;           \
    js->consumed = 1;               \
    jsval_t rhs = _f(js);           \
    if (is_err(rhs)) return rhs;    \
    res = do_op(js, op, res, rhs);  \
  }                                 \
  return res;

// 字面量解析函数 - 解析各种基本字面量
// 包括数字、字符串、对象、函数、布尔值、null、undefined等
static jsval_t js_literal(struct js *js) {
  next(js);  // 获取下一个token
  setlwm(js);  // 更新低水位标记
  // printf("css : %u\n", js->css);  // 调试输出
  if (js->maxcss > 0 && js->css > js->maxcss) return js_mkerr(js, "C stack");  // 检查C栈溢出
  js->consumed = 1;  // 标记token已消费
  switch (js->tok) {  // 根据token类型解析不同的字面量 // clang-format off
    case TOK_ERR:         return js_mkerr(js, "parse error");  // 解析错误
    case TOK_NUMBER:      return js->tval;  // 数字字面量
    case TOK_STRING:      return js_str_literal(js);  // 字符串字面量
    case TOK_LBRACE:      return js_obj_literal(js);  // 对象字面量
    case TOK_FUNC:        return js_func_literal(js);  // 函数字面量
    case TOK_NULL:        return js_mknull();  // null字面量
    case TOK_UNDEF:       return js_mkundef();  // undefined字面量
    case TOK_TRUE:        return js_mktrue();  // true字面量
    case TOK_FALSE:       return js_mkfalse();  // false字面量
    case TOK_IDENTIFIER:  return mkcoderef((jsoff_t) js->toff, (jsoff_t) js->tlen);  // 标识符引用
    default:              return js_mkerr(js, "bad expr");  // 无效表达式
  }  // clang-format on
}

// 分组表达式解析 - 处理括号表达式和基本字面量
// 解析 (expression) 或直接解析字面量
static jsval_t js_group(struct js *js) {
  if (next(js) == TOK_LPAREN) {  // 如果是左括号
    js->consumed = 1;  // 消费左括号
    jsval_t v = js_expr(js);  // 解析括号内的表达式
    if (is_err(v)) return v;  // 检查表达式是否有错误
    if (next(js) != TOK_RPAREN) return js_mkerr(js, ") expected");  // 期望右括号
    js->consumed = 1;  // 消费右括号
    return v;  // 返回表达式结果
  } else {
    return js_literal(js);  // 否则解析字面量
  }
}

// 函数调用和属性访问解析 - 处理 obj.prop 和 func() 操作
// 解析链式调用如 obj.method().prop
static jsval_t js_call_dot(struct js *js) {
  jsval_t res = js_group(js);  // 解析基础表达式
  if (is_err(res)) return res;  // 检查错误
  if (vtype(res) == T_CODEREF) {  // 如果是标识符引用
    res = lookup(js, &js->code[coderefoff(res)], codereflen(res));  // 查找变量
  }
  while (next(js) == TOK_LPAREN || next(js) == TOK_DOT) {  // 处理链式调用
    if (js->tok == TOK_DOT) {  // 属性访问
      js->consumed = 1;  // 消费点号
      res = do_op(js, TOK_DOT, res, js_group(js));  // 执行点操作
    } else {  // 函数调用
      jsval_t params = js_call_params(js);  // 解析参数列表
      if (is_err(params)) return params;  // 检查参数错误
      res = do_op(js, TOK_CALL, res, params);  // 执行函数调用
    }
  }
  return res;  // 返回最终结果
}

// 后缀表达式解析 - 处理后置递增递减操作符
// 解析 expr++ 和 expr-- 操作
static jsval_t js_postfix(struct js *js) {
  jsval_t res = js_call_dot(js);  // 解析基础表达式
  if (is_err(res)) return res;  // 检查错误
  next(js);  // 获取下一个token
  if (js->tok == TOK_POSTINC || js->tok == TOK_POSTDEC) {  // 如果是后置递增或递减
    js->consumed = 1;  // 消费操作符
    res = do_op(js, js->tok, res, 0);  // 执行后置操作
  }
  return res;  // 返回结果
}

// 一元表达式解析 - 处理前置一元操作符
// 解析 !expr, ~expr, typeof expr, -expr, +expr 等
static jsval_t js_unary(struct js *js) {
  if (next(js) == TOK_NOT || js->tok == TOK_TILDA || js->tok == TOK_TYPEOF ||  // 检查一元操作符
      js->tok == TOK_MINUS || js->tok == TOK_PLUS) {
    uint8_t t = js->tok;  // 保存操作符
    if (t == TOK_MINUS) t = TOK_UMINUS;  // 转换为一元减号
    if (t == TOK_PLUS) t = TOK_UPLUS;  // 转换为一元加号
    js->consumed = 1;  // 消费操作符
    return do_op(js, t, js_mkundef(), js_unary(js));  // 递归解析一元表达式
  } else {
    return js_postfix(js);  // 否则解析后缀表达式
  }
}

// 乘除取余运算解析 - 优先级14 (*, /, %)
// 左结合，优先级高于加减运算
static jsval_t js_mul_div_rem(struct js *js) {
  LTR_BINOP(js_unary,
            (next(js) == TOK_MUL || js->tok == TOK_DIV || js->tok == TOK_REM));
}

// 加减运算解析 - 优先级13 (+, -)
// 左结合，优先级低于乘除运算
static jsval_t js_plus_minus(struct js *js) {
  LTR_BINOP(js_mul_div_rem, (next(js) == TOK_PLUS || js->tok == TOK_MINUS));
}

// 位移运算解析 - 优先级12 (<<, >>, >>>)
// 左结合，优先级低于加减运算
static jsval_t js_shifts(struct js *js) {
  LTR_BINOP(js_plus_minus, (next(js) == TOK_SHR || next(js) == TOK_SHL ||
                            next(js) == TOK_ZSHR));
}

// 比较运算解析 - 优先级11 (<, <=, >, >=)
// 左结合，优先级低于位移运算
static jsval_t js_comparison(struct js *js) {
  LTR_BINOP(js_shifts, (next(js) == TOK_LT || next(js) == TOK_LE ||
                        next(js) == TOK_GT || next(js) == TOK_GE));
}

// 相等性比较解析 - 优先级10 (===, !==)
// 左结合，优先级低于比较运算
static jsval_t js_equality(struct js *js) {
  LTR_BINOP(js_comparison, (next(js) == TOK_EQ || next(js) == TOK_NE));
}

// 按位与运算解析 - 优先级9 (&)
// 左结合，优先级低于相等性比较
static jsval_t js_bitwise_and(struct js *js) {
  LTR_BINOP(js_equality, (next(js) == TOK_AND));
}

// 按位异或运算解析 - 优先级8 (^)
// 左结合，优先级低于按位与
static jsval_t js_bitwise_xor(struct js *js) {
  LTR_BINOP(js_bitwise_and, (next(js) == TOK_XOR));
}

// 按位或运算解析 - 优先级7 (|)
// 左结合，优先级低于按位异或
static jsval_t js_bitwise_or(struct js *js) {
  LTR_BINOP(js_bitwise_xor, (next(js) == TOK_OR));
}

// 逻辑与运算解析 - 优先级6 (&&)
// 左结合，支持短路求值：如果左操作数为false，不计算右操作数
static jsval_t js_logical_and(struct js *js) {
  jsval_t res = js_bitwise_or(js);  // 解析左操作数
  if (is_err(res)) return res;  // 检查错误
  uint8_t flags = js->flags;  // 保存标志
  while (next(js) == TOK_LAND) {  // 处理逻辑与操作符
    js->consumed = 1;  // 消费操作符
    res = resolveprop(js, res);  // 解析左操作数的值
    if (!js_truthy(js, res)) js->flags |= F_NOEXEC;  // 短路求值：false && ... 不执行右侧
    if (js->flags & F_NOEXEC) {
      js_logical_and(js);  // 跳过右操作数的执行
    } else {
      res = js_logical_and(js);  // 递归解析右操作数
    }
  }
  js->flags = flags;  // 恢复标志
  return res;  // 返回结果
}

// 逻辑或运算解析 - 优先级5 (||)
// 左结合，支持短路求值：如果左操作数为true，不计算右操作数
static jsval_t js_logical_or(struct js *js) {
  jsval_t res = js_logical_and(js);  // 解析左操作数
  if (is_err(res)) return res;  // 检查错误
  uint8_t flags = js->flags;  // 保存标志
  while (next(js) == TOK_LOR) {  // 处理逻辑或操作符
    js->consumed = 1;  // 消费操作符
    res = resolveprop(js, res);  // 解析左操作数的值
    if (js_truthy(js, res)) js->flags |= F_NOEXEC;  // 短路求值：true || ... 不执行右侧
    if (js->flags & F_NOEXEC) {
      js_logical_or(js);  // 跳过右操作数的执行
    } else {
      res = js_logical_or(js);  // 递归解析右操作数
    }
  }
  js->flags = flags;  // 恢复标志
  return res;  // 返回结果
}

// 三元条件运算符解析 - 优先级4 (condition ? true_expr : false_expr)
// 右结合，支持条件求值：根据条件只计算其中一个分支
static jsval_t js_ternary(struct js *js) {
  jsval_t res = js_logical_or(js);  // 解析条件表达式
  if (next(js) == TOK_Q) {  // 如果遇到问号
    uint8_t flags = js->flags;  // 保存标志
    js->consumed = 1;  // 消费问号
    if (js_truthy(js, resolveprop(js, res))) {  // 如果条件为真
      res = js_ternary(js);  // 解析真值分支
      js->flags |= F_NOEXEC;  // 设置非执行模式
      EXPECT(TOK_COLON, js->flags = flags);  // 期望冒号
      js_ternary(js);  // 跳过假值分支
      js->flags = flags;  // 恢复标志
    } else {  // 如果条件为假
      js->flags |= F_NOEXEC;  // 设置非执行模式
      js_ternary(js);  // 跳过真值分支
      EXPECT(TOK_COLON, js->flags = flags);  // 期望冒号
      js->flags = flags;  // 恢复标志
      res = js_ternary(js);  // 解析假值分支
    }
  }
  return res;  // 返回结果
}

// 赋值运算符解析 - 优先级2 (=, +=, -=, *=, /=, %=, <<=, >>=, >>>=, &=, ^=, |=)
// 右结合，优先级最低的运算符之一
static jsval_t js_assignment(struct js *js) {
  RTL_BINOP(js_ternary, js_assignment,
            (next(js) == TOK_ASSIGN || js->tok == TOK_PLUS_ASSIGN ||
             js->tok == TOK_MINUS_ASSIGN || js->tok == TOK_MUL_ASSIGN ||
             js->tok == TOK_DIV_ASSIGN || js->tok == TOK_REM_ASSIGN ||
             js->tok == TOK_SHL_ASSIGN || js->tok == TOK_SHR_ASSIGN ||
             js->tok == TOK_ZSHR_ASSIGN || js->tok == TOK_AND_ASSIGN ||
             js->tok == TOK_XOR_ASSIGN || js->tok == TOK_OR_ASSIGN));
}

// 表达式解析入口函数 - 从最低优先级开始解析
// 这是表达式解析的顶层函数
static jsval_t js_expr(struct js *js) {
  return js_assignment(js);  // 从赋值运算符开始解析
}

// let变量声明处理函数
// 处理JavaScript的let语句，支持变量声明和初始化
static jsval_t js_let(struct js *js) {
  uint8_t exe = !(js->flags & F_NOEXEC);  // 检查是否需要执行（非跳过模式）
  js->consumed = 1;
  for (;;) {  // 循环处理多个变量声明（用逗号分隔）
    EXPECT(TOK_IDENTIFIER, );  // 期望变量名标识符
    js->consumed = 0;
    jsoff_t noff = js->toff, nlen = js->tlen;  // 获取变量名的偏移和长度
    char *name = (char *) &js->code[noff];     // 指向变量名字符串
    jsval_t v = js_mkundef();                  // 默认值为undefined
    js->consumed = 1;
    if (next(js) == TOK_ASSIGN) {  // 如果有赋值操作
      js->consumed = 1;
      v = js_expr(js);                         // 解析赋值表达式
      if (is_err(v)) return v;                 // 传播错误
    }
    if (exe) {  // 如果需要执行
      // 检查变量是否已经在当前作用域中声明
      if (lkp(js, js->scope, name, nlen) > 0)
        return js_mkerr(js, "'%.*s' already declared", (int) nlen, name);
      // 在当前作用域中设置变量属性
      jsval_t x =
          setprop(js, js->scope, js_mkstr(js, name, nlen), resolveprop(js, v));
      if (is_err(x)) return x;  // 检查设置是否成功
    }
    // 检查是否到达语句结束或文件结束
    if (next(js) == TOK_SEMICOLON || next(js) == TOK_EOF) break;
    EXPECT(TOK_COMMA, );  // 期望逗号分隔符（多变量声明）
  }
  return js_mkundef();  // 返回undefined
}

// 代码块或单个语句处理函数
// 根据下一个token决定是处理代码块{}还是单个语句
static jsval_t js_block_or_stmt(struct js *js) {
  if (next(js) == TOK_LBRACE) return js_block(js, !(js->flags & F_NOEXEC));  // 如果是左大括号，处理代码块
  jsval_t res = resolveprop(js, js_stmt(js));  // 否则处理单个语句
  js->consumed = 0;  // 重置消费标志
  return res;
}

// if条件语句处理函数
// 处理JavaScript的if-else语句，包括条件判断和分支执行
static jsval_t js_if(struct js *js) {
  js->consumed = 1;
  EXPECT(TOK_LPAREN, );  // 期望左括号
  jsval_t res = js_mkundef(), cond = resolveprop(js, js_expr(js));  // 解析条件表达式
  EXPECT(TOK_RPAREN, );  // 期望右括号
  bool cond_true = js_truthy(js, cond), exe = !(js->flags & F_NOEXEC);  // 判断条件真假和执行状态
  // printf("IF COND: %s, true? %d\n", js_str(js, cond), cond_true);
  if (!cond_true) js->flags |= F_NOEXEC;  // 如果条件为假，设置不执行标志
  jsval_t blk = js_block_or_stmt(js);     // 执行if分支
  if (cond_true) res = blk;               // 如果条件为真，保存if分支结果
  if (exe && !cond_true) js->flags &= (uint8_t) ~F_NOEXEC;  // 恢复执行状态
  if (lookahead(js) == TOK_ELSE) {  // 检查是否有else分支
    js->consumed = 1;
    next(js);
    js->consumed = 1;
    if (cond_true) js->flags |= F_NOEXEC;  // 如果条件为真，跳过else分支
    blk = js_block_or_stmt(js);            // 执行else分支
    if (!cond_true) res = blk;             // 如果条件为假，保存else分支结果
    if (cond_true && exe) js->flags &= (uint8_t) ~F_NOEXEC;  // 恢复执行状态
  }
  return res;
}

// 期望特定token的辅助函数
// 检查下一个token是否为期望的类型，如果不是则设置错误
static inline bool expect(struct js *js, uint8_t tok, jsval_t *res) {
  if (next(js) != tok) {
    *res = js_mkerr(js, "parse error");  // 设置解析错误
    return false;
  } else {
    js->consumed = 1;  // 标记token已消费
    return true;
  }
}

// 错误检查辅助函数
// 检查值是否为错误，如果是则传播错误
static inline bool is_err2(jsval_t *v, jsval_t *res) {
  bool r = is_err(*v);  // 检查是否为错误值
  if (r) *res = *v;     // 如果是错误，传播错误
  return r;
}

// for循环语句处理函数
// 处理JavaScript的for循环，包括初始化、条件判断、递增表达式和循环体
static jsval_t js_for(struct js *js) {
  uint8_t flags = js->flags, exe = !(flags & F_NOEXEC);  // 保存当前标志和执行状态
  jsval_t v, res = js_mkundef();
  jsoff_t pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;  // 记录各部分的位置
  if (exe) mkscope(js);  // 如果需要执行，创建新的作用域
  if (!expect(js, TOK_FOR, &res)) goto done;    // 期望for关键字
  if (!expect(js, TOK_LPAREN, &res)) goto done; // 期望左括号

  // 处理初始化部分 for(init; ...)
  if (next(js) == TOK_SEMICOLON) {  // 空的初始化
  } else if (next(js) == TOK_LET) { // let变量声明
    v = js_let(js);
    if (is_err2(&v, &res)) goto done;
  } else {                          // 表达式初始化
    v = js_expr(js);
    if (is_err2(&v, &res)) goto done;
  }
  if (!expect(js, TOK_SEMICOLON, &res)) goto done;
  
  js->flags |= F_NOEXEC;  // 设置不执行标志，先解析不执行
  pos1 = js->pos;         // 记录条件表达式位置
  
  // 处理条件部分 for(...; condition; ...)
  if (next(js) != TOK_SEMICOLON) {
    v = js_expr(js);  // 解析条件表达式
    if (is_err2(&v, &res)) goto done;
  }
  if (!expect(js, TOK_SEMICOLON, &res)) goto done;
  
  pos2 = js->pos;  // 记录递增表达式位置
  
  // 处理递增部分 for(...; ...; increment)
  if (next(js) != TOK_RPAREN) {
    v = js_expr(js);  // 解析递增表达式
    if (is_err2(&v, &res)) goto done;
  }
  if (!expect(js, TOK_RPAREN, &res)) goto done;
  
  pos3 = js->pos;  // 记录循环体开始位置
  v = js_block_or_stmt(js);  // 解析循环体
  if (is_err2(&v, &res)) goto done;
  pos4 = js->pos;  // 记录循环体结束位置
  
  // 执行循环
  while (!(flags & F_NOEXEC)) {
    // 检查条件
    js->flags = flags, js->pos = pos1, js->consumed = 1;
    if (next(js) != TOK_SEMICOLON) {     // 如果有条件表达式
      v = resolveprop(js, js_expr(js));  // 计算条件值
      if (is_err2(&v, &res)) goto done;  // 错误处理
      if (!js_truthy(js, v)) break;      // 条件为假则退出循环
    }
    
    // 执行循环体
    js->pos = pos3, js->consumed = 1, js->flags |= F_LOOP;  // 设置循环标志
    v = js_block_or_stmt(js);                               // 执行循环体
    if (is_err2(&v, &res)) goto done;                       // 错误处理
    if (js->flags & F_BREAK) break;  // 如果执行了break，退出循环
    
    // 执行递增表达式
    js->flags = flags, js->pos = pos2, js->consumed = 1;  // 跳转到递增表达式
    if (next(js) != TOK_RPAREN) {                         // 如果有递增表达式
      v = js_expr(js);                                    // 执行递增表达式
      if (is_err2(&v, &res)) goto done;  // 错误处理
    }
  }
  js->pos = pos4, js->tok = TOK_SEMICOLON, js->consumed = 0;  // 恢复位置
done:
  if (exe) delscope(js);  // 如果创建了作用域，删除作用域
  js->flags = flags;      // 恢复标志
  return res;
}

// break语句处理函数
// 处理JavaScript的break语句，用于跳出循环
static jsval_t js_break(struct js *js) {
  if (js->flags & F_NOEXEC) {  // 如果在不执行模式下，什么都不做
  } else {
    if (!(js->flags & F_LOOP)) return js_mkerr(js, "not in loop");  // 检查是否在循环中
    js->flags |= F_BREAK | F_NOEXEC;  // 设置break标志和不执行标志
  }
  js->consumed = 1;  // 标记token已消费
  return js_mkundef();
}

// continue语句处理函数
// 处理JavaScript的continue语句，用于跳过当前循环迭代
static jsval_t js_continue(struct js *js) {
  if (js->flags & F_NOEXEC) {  // 如果在不执行模式下，什么都不做
  } else {
    if (!(js->flags & F_LOOP)) return js_mkerr(js, "not in loop");  // 检查是否在循环中
    js->flags |= F_NOEXEC;  // 设置不执行标志，跳过剩余代码
  }
  js->consumed = 1;  // 标记token已消费
  return js_mkundef();
}

// return语句处理函数
// 处理JavaScript的return语句，用于从函数返回值
static jsval_t js_return(struct js *js) {
  uint8_t exe = !(js->flags & F_NOEXEC);  // 检查是否需要执行
  js->consumed = 1;
  if (exe && !(js->flags & F_CALL)) return js_mkerr(js, "not in func");  // 检查是否在函数中
  if (next(js) == TOK_SEMICOLON) return js_mkundef();  // 如果没有返回值，返回undefined
  jsval_t res = resolveprop(js, js_expr(js));  // 解析返回值表达式
  if (exe) {
    js->pos = js->clen;     // 跳转到代码末尾，退出执行
    js->flags |= F_RETURN;  // 设置返回标志，告知调用者已执行return
  }
  return resolveprop(js, res);  // 返回解析后的值
}

// 语句解析的核心分发函数
// 根据token类型分发到相应的语句处理函数
static jsval_t js_stmt(struct js *js) {
  jsval_t res;
  // jsoff_t pos = js->pos - js->tlen;
  if (js->brk > js->gct) js_gc(js);  // 如果内存使用超过阈值，触发垃圾回收
  switch (next(js)) {  // 根据下一个token类型进行分发 // clang-format off
    // 未实现的JavaScript关键字，返回错误
    case TOK_CASE: case TOK_CATCH: case TOK_CLASS: case TOK_CONST:
    case TOK_DEFAULT: case TOK_DELETE: case TOK_DO: case TOK_FINALLY:
    case TOK_IN: case TOK_INSTANCEOF: case TOK_NEW: case TOK_SWITCH:
    case TOK_THIS: case TOK_THROW: case TOK_TRY: case TOK_VAR: case TOK_VOID:
    case TOK_WITH: case TOK_WHILE: case TOK_YIELD:
      res = js_mkerr(js, "'%.*s' not implemented", (int) js->tlen, js->code + js->toff);
      break;
    // 已实现的语句类型
    case TOK_CONTINUE:  res = js_continue(js); break;  // continue语句
    case TOK_BREAK:     res = js_break(js); break;     // break语句
    case TOK_LET:       res = js_let(js); break;       // let变量声明
    case TOK_IF:        res = js_if(js); break;        // if条件语句
    case TOK_LBRACE:    res = js_block(js, !(js->flags & F_NOEXEC)); break;  // 代码块
    case TOK_FOR:       res = js_for(js); break;       // for循环语句
    case TOK_RETURN:    res = js_return(js); break;    // return语句
    default:            res = resolveprop(js, js_expr(js)); break;  // 默认作为表达式处理
  }
  //printf("STMT [%.*s] -> %s, tok %d, flags %d\n", (int) (js->pos - pos), &js->code[pos], js_str(js, res), next(js), js->flags);
  // 检查语句结束符：分号、文件结束或右大括号
  if (next(js) != TOK_SEMICOLON && next(js) != TOK_EOF && next(js) != TOK_RBRACE) 
    return js_mkerr(js, "; expected");
  js->consumed = 1;  // 标记分隔符已消费
  // clang-format on
  return res;
}

// JavaScript引擎创建函数
// 在给定的内存缓冲区中初始化JavaScript引擎实例
struct js *js_create(void *buf, size_t len) {
  struct js *js = NULL;
  if (len < sizeof(*js) + esize(T_OBJ)) return js;  // 检查内存是否足够
  memset(buf, 0, len);                       // 清零内存，这很重要！
  js = (struct js *) buf;                    // js结构体位于缓冲区开始处
  js->mem = (uint8_t *) (js + 1);            // JS数据内存紧跟在js结构体后面
  js->size = (jsoff_t) (len - sizeof(*js));  // 计算JS数据可用内存大小
  js->scope = mkobj(js, 0);                  // 创建全局作用域对象
  js->size = js->size / 8U * 8U;             // 按8字节对齐内存大小
  js->lwm = js->size;                        // 初始低水位标记：100%空闲
  js->gct = js->size / 2;                    // 设置垃圾回收触发阈值为一半内存
  return js;
}

// clang-format off
// 引擎配置函数
void js_setgct(struct js *js, size_t gct) { js->gct = (jsoff_t) gct; }  // 设置垃圾回收触发阈值
void js_setmaxcss(struct js *js, size_t max) { js->maxcss = (jsoff_t) max; }  // 设置最大调用栈大小

// JavaScript值创建函数
jsval_t js_mktrue(void) { return mkval(T_BOOL, 1); }    // 创建true值
jsval_t js_mkfalse(void) { return mkval(T_BOOL, 0); }   // 创建false值
jsval_t js_mkundef(void) { return mkval(T_UNDEF, 0); }  // 创建undefined值
jsval_t js_mknull(void) { return mkval(T_NULL, 0); }    // 创建null值
jsval_t js_mknum(double value) { return tov(value); }   // 创建数字值
jsval_t js_mkobj(struct js *js) { return mkobj(js, 0); }  // 创建对象值
jsval_t js_mkfun(jsval_t (*fn)(struct js *, jsval_t *, int)) { return mkval(T_CFUNC, (size_t) (void *) fn); }  // 创建C函数值

// JavaScript值获取函数
double js_getnum(jsval_t value) { return tod(value); }  // 获取数字值
int js_getbool(jsval_t value) { return vdata(value) & 1 ? 1 : 0; }  // 获取布尔值

// 获取全局对象
jsval_t js_glob(struct js *js) { (void) js; return mkval(T_OBJ, 0); }  // 返回全局对象引用

// 对象属性设置函数
// 为JavaScript对象设置属性值
void js_set(struct js *js, jsval_t obj, const char *key, jsval_t val) {
  if (vtype(obj) == T_OBJ) setprop(js, obj, js_mkstr(js, key, strlen(key)), val);  // 只对对象类型设置属性
}

// 字符串值获取函数
// 从JavaScript字符串值中获取C字符串指针和长度
char *js_getstr(struct js *js, jsval_t value, size_t *len) {
  if (vtype(value) != T_STR) return NULL;  // 检查是否为字符串类型
  jsoff_t n, off = vstr(js, value, &n);    // 获取字符串偏移和长度
  if (len != NULL) *len = n;               // 返回长度
  return (char *) &js->mem[off];           // 返回字符串指针
}

// JavaScript值类型判断函数
// 返回JavaScript值的类型常量
int js_type(jsval_t val) {
  switch (vtype(val)) {  
    case T_UNDEF:   return JS_UNDEF;   // undefined类型
    case T_NULL:    return JS_NULL;    // null类型
    case T_BOOL:    return vdata(val) == 0 ? JS_FALSE: JS_TRUE;  // 布尔类型（区分true/false）
    case T_STR:     return JS_STR;     // 字符串类型
    case T_NUM:     return JS_NUM;     // 数字类型
    case T_ERR:     return JS_ERR;     // 错误类型
    default:        return JS_PRIV;    // 私有类型（对象、函数等）
  }
}

// 引擎统计信息获取函数
// 获取内存使用统计信息
void js_stats(struct js *js, size_t *total, size_t *lwm, size_t *css) {
  if (total) *total = js->size;  // 总内存大小
  if (lwm) *lwm = js->lwm;       // 低水位标记（最低可用内存）
  if (css) *css = js->css;       // 当前调用栈大小
}
// clang-format on

// 函数参数类型检查函数
// 根据类型规范字符串检查参数类型是否匹配
bool js_chkargs(jsval_t *args, int nargs, const char *spec) {
  int i = 0, ok = 1;
  for (; ok && i < nargs && spec[i]; i++) {  // 遍历所有参数
    uint8_t t = vtype(args[i]), c = (uint8_t) spec[i];  // 获取实际类型和期望类型
    // 检查类型匹配：'b'=布尔, 'd'=数字, 's'=字符串, 'j'=任意类型
    ok = (c == 'b' && t == T_BOOL) || (c == 'd' && t == T_NUM) ||
         (c == 's' && t == T_STR) || (c == 'j');
  }
  if (spec[i] != '\0' || i != nargs) ok = 0;  // 检查参数数量是否匹配
  return ok;
}

// JavaScript代码执行函数
// 解析并执行JavaScript代码字符串
jsval_t js_eval(struct js *js, const char *buf, size_t len) {
  // printf("EVAL: [%.*s]\n", (int) len, buf);
  jsval_t res = js_mkundef();                    // 初始化结果为undefined
  if (len == (size_t) ~0U) len = strlen(buf);    // 如果长度未指定，计算字符串长度
  js->consumed = 1;                              // 设置token消费标志
  js->tok = TOK_ERR;                             // 初始化token为错误
  js->code = buf;                                // 设置代码缓冲区
  js->clen = (jsoff_t) len;                      // 设置代码长度
  js->pos = 0;                                   // 重置解析位置
  js->cstk = &res;                               // 设置调用栈指针
  while (next(js) != TOK_EOF && !is_err(res)) {  // 循环解析语句直到文件结束或出错
    res = js_stmt(js);                           // 解析并执行语句
  }
  return res;                                    // 返回最后一个语句的结果
}

#ifdef JS_DUMP
// JavaScript引擎内存调试函数
// 打印引擎内存中所有对象的详细信息，用于调试
void js_dump(struct js *js) {
  jsoff_t off = 0, v;
  // 打印引擎统计信息：总大小、当前使用、低水位、调用栈、垃圾回收状态
  printf("JS size %u, brk %u, lwm %u, css %u, nogc %u\n", js->size, js->brk,
         js->lwm, (unsigned) js->css, js->nogc);
  while (off < js->brk) {  // 遍历已使用的内存区域
    memcpy(&v, &js->mem[off], sizeof(v));  // 读取对象头部信息
    printf(" %5u: ", off);                 // 打印偏移地址
    if ((v & 3U) == T_OBJ) {               // 如果是对象类型
      printf("OBJ %u %u\n", v & ~3U,
             loadoff(js, (jsoff_t) (off + sizeof(off))));
    } else if ((v & 3U) == T_PROP) {       // 如果是属性类型
      jsoff_t koff = loadoff(js, (jsoff_t) (off + sizeof(v)));           // 键偏移
      jsval_t val = loadval(js, (jsoff_t) (off + sizeof(v) + sizeof(v))); // 值
      printf("PROP next %u, koff %u vtype %d vdata %lu\n", v & ~3U, koff,
             vtype(val), (unsigned long) vdata(val));
    } else if ((v & 3) == T_STR) {         // 如果是字符串类型
      jsoff_t len = offtolen(v);           // 获取字符串长度
      printf("STR %u [%.*s]\n", len, (int) len, js->mem + off + sizeof(v));
    } else {                               // 未知类型
      printf("???\n");
      break;
    }
    off += esize(v);  // 移动到下一个对象
  }
}
#endif