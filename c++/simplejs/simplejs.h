#pragma once
#define JS_VERSION "1.0.0"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string>

using namespace std;

enum
{
    JS_UNDEF,
    JS_NULL,
    JS_TRUE,
    JS_FALSE,
    JS_STR,
    JS_NUM,
    JS_ERR,
    JS_PRIV
};

// Pack JS values into uin64_t, double nan, per IEEE 754
// 64bit "double": 1 bit sign, 11 bits exponent, 52 bits mantissa
//
// seeeeeee|eeeemmmm|mmmmmmmm|mmmmmmmm|mmmmmmmm|mmmmmmmm|mmmmmmmm|mmmmmmmm
// 11111111|11110000|00000000|00000000|00000000|00000000|00000000|00000000 inf
// 11111111|11111000|00000000|00000000|00000000|00000000|00000000|00000000 qnan
//
// 11111111|1111tttt|vvvvvvvv|vvvvvvvv|vvvvvvvv|vvvvvvvv|vvvvvvvv|vvvvvvvv
//  NaN marker |type|  48-bit placeholder for values: pointers, strings
//
// On 64-bit platforms, pointers are really 48 bit only, so they can fit,
// provided they are sign extended
typedef uint64_t JSValue; // JS value

class SimpleJS
{
public:
    SimpleJS(size_t len);

public:
    JSValue eval(string);
    void setMaxCss(size_t);
    void setGCThreshold(size_t);

public:
    string toString();
};

struct js *js_create(void *buf, size_t len);        // Create JS instance
jsval_t js_eval(struct js *, const char *, size_t); // Execute JS code
jsval_t js_glob(struct js *);                       // Return global object
const char *js_str(struct js *, jsval_t val);       // Stringify JS value
bool js_chkargs(jsval_t *, int, const char *);      // Check args validity
void js_setmaxcss(struct js *, size_t);             // Set max C stack size
void js_setgct(struct js *, size_t);                // Set GC trigger threshold
void js_stats(struct js *, size_t *total, size_t *min, size_t *cstacksize);
void js_dump(struct js *); // Print debug info. Requires -DJS_DUMP

// Create JS values from C values
jsval_t js_mkundef(void);                                     // Create undefined
jsval_t js_mknull(void);                                      // Create null, null, true, false
jsval_t js_mktrue(void);                                      // Create true
jsval_t js_mkfalse(void);                                     // Create false
jsval_t js_mkstr(struct js *, const void *, size_t);          // Create string
jsval_t js_mknum(double);                                     // Create number
jsval_t js_mkerr(struct js *js, const char *fmt, ...);        // Create error
jsval_t js_mkfun(jsval_t (*fn)(struct js *, jsval_t *, int)); // Create func
jsval_t js_mkobj(struct js *);                                // Create object
void js_set(struct js *, jsval_t, const char *, jsval_t);     // Set obj attr

// Extract C values from JS values

int js_type(jsval_t val);                                 // Return JS value type
double js_getnum(jsval_t val);                            // Get number
int js_getbool(jsval_t val);                              // Get boolean, 0 or 1
char *js_getstr(struct js *js, jsval_t val, size_t *len); // Get string
jsval_t mkval(uint8_t type, uint64_t data);
void js_gc(struct js *js);

#ifndef JS_EXPR_MAX
#define JS_EXPR_MAX 20
#endif

#ifndef JS_GC_THRESHOLD
#define JS_GC_THRESHOLD 0.75
#endif

typedef uint32_t jsoff_t;

struct js
{
    jsoff_t css;      // Max observed C stack size
    jsoff_t lwm;      // JS RAM low watermark: min free RAM observed
    const char *code; // Currently parsed code snippet
    char errmsg[33];  // Error message placeholder
    uint8_t tok;      // Last parsed token value
    uint8_t consumed; // Indicator that last parsed token was consumed
    uint8_t flags;    // Execution flags, see F_* constants below
#define F_NOEXEC 1U   // Parse code, but not execute
#define F_LOOP 2U     // We're inside the loop
#define F_CALL 4U     // We're inside a function call
#define F_BREAK 8U    // Exit the loop
#define F_RETURN 16U  // Return has been executed
    jsoff_t clen;     // Code snippet length
    jsoff_t pos;      // Current parsing position
    jsoff_t toff;     // Offset of the last parsed token
    jsoff_t tlen;     // Length of the last parsed token
    jsoff_t nogc;     // Entity offset to exclude from GC
    jsval_t tval;     // Holds last parsed numeric or string literal value
    jsval_t scope;    // Current scope
    uint8_t *mem;     // Available JS memory
    jsoff_t size;     // Memory size
    jsoff_t brk;      // Current mem usage boundary
    jsoff_t gct;      // GC threshold. If brk > gct, trigger GC
    jsoff_t maxcss;   // Maximum allowed C stack size usage
    void *cstk;       // C stack pointer at the beginning of js_eval()
};

enum
{
    TOK_ERR,
    TOK_EOF,
    TOK_IDENTIFIER,
    TOK_NUMBER,
    TOK_STRING,
    TOK_SEMICOLON,
    TOK_LPAREN,
    TOK_RPAREN,
    TOK_LBRACE,
    TOK_RBRACE,
    // Keyword tokens
    TOK_BREAK = 50,
    TOK_CASE,
    TOK_CATCH,
    TOK_CLASS,
    TOK_CONST,
    TOK_CONTINUE,
    TOK_DEFAULT,
    TOK_DELETE,
    TOK_DO,
    TOK_ELSE,
    TOK_FINALLY,
    TOK_FOR,
    TOK_FUNC,
    TOK_IF,
    TOK_IN,
    TOK_INSTANCEOF,
    TOK_LET,
    TOK_NEW,
    TOK_RETURN,
    TOK_SWITCH,
    TOK_THIS,
    TOK_THROW,
    TOK_TRY,
    TOK_VAR,
    TOK_VOID,
    TOK_WHILE,
    TOK_WITH,
    TOK_YIELD,
    TOK_UNDEF,
    TOK_NULL,
    TOK_TRUE,
    TOK_FALSE,
    // JS Operator tokens
    TOK_DOT = 100,
    TOK_CALL,
    TOK_POSTINC,
    TOK_POSTDEC,
    TOK_NOT,
    TOK_TILDA, // 100
    TOK_TYPEOF,
    TOK_UPLUS,
    TOK_UMINUS,
    TOK_EXP,
    TOK_MUL,
    TOK_DIV,
    TOK_REM, // 106
    TOK_PLUS,
    TOK_MINUS,
    TOK_SHL,
    TOK_SHR,
    TOK_ZSHR,
    TOK_LT,
    TOK_LE,
    TOK_GT, // 113
    TOK_GE,
    TOK_EQ,
    TOK_NE,
    TOK_AND,
    TOK_XOR,
    TOK_OR,
    TOK_LAND,
    TOK_LOR, // 121
    TOK_COLON,
    TOK_Q,
    TOK_ASSIGN,
    TOK_PLUS_ASSIGN,
    TOK_MINUS_ASSIGN,
    TOK_MUL_ASSIGN,
    TOK_DIV_ASSIGN,
    TOK_REM_ASSIGN,
    TOK_SHL_ASSIGN,
    TOK_SHR_ASSIGN,
    TOK_ZSHR_ASSIGN,
    TOK_AND_ASSIGN,
    TOK_XOR_ASSIGN,
    TOK_OR_ASSIGN,
    TOK_COMMA,
};

enum
{
    // IMPORTANT: T_OBJ, T_PROP, T_STR must go first.  That is required by the
    // memory layout functions: memory entity types are encoded in the 2 bits,
    // thus type values must be 0,1,2,3
    T_OBJ,
    T_PROP,
    T_STR,
    T_UNDEF,
    T_NULL,
    T_NUM,
    T_BOOL,
    T_FUNC,
    T_CODEREF,
    T_CFUNC,
    T_ERR
};