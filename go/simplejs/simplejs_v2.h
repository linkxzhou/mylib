#pragma once

/**
 * SimpleJS v2.0 - Modern C++11 JavaScript Interpreter
 * 
 * A lightweight JavaScript interpreter rewritten in C++11 with modern features:
 * - Smart pointers for automatic memory management
 * - RAII for resource safety
 * - Strong typed enums
 * - Template-based type safety
 * - Lambda expressions
 * - Move semantics
 * 
 * Original SimpleJS by Sergey Lyubka
 * C++11 rewrite with architectural improvements
 */

#include <cstdint>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <stdexcept>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdarg>
#include <cassert>
#include <algorithm>
#include <limits>

namespace simplejs {

// Version information
constexpr const char* VERSION = "2.0.0";

// Configuration constants
constexpr size_t DEFAULT_EXPR_MAX = 20;
constexpr double DEFAULT_GC_THRESHOLD = 0.75;
constexpr size_t DEFAULT_ERROR_MSG_SIZE = 64;

// Forward declarations
class JSEngine;
class JSValue;
class JSObject;
class JSFunction;

// Type aliases for better readability
using JSOffset = uint32_t;
using JSValueType = uint64_t;
using JSNativeFunction = std::function<JSValue(JSEngine&, const std::vector<JSValue>&)>;

// Strong typed enumerations - 与原版simplejs.cc保持一致
enum class TokenType : uint8_t {
    // 基本令牌 (0-10)
    ERROR = 0, END_OF_FILE, IDENTIFIER, NUMBER, STRING, SEMICOLON,
    LEFT_PAREN, RIGHT_PAREN, LEFT_BRACE, RIGHT_BRACE,
    
    // JavaScript关键字令牌（从50开始）
    BREAK = 50, CASE, CATCH, CLASS, CONST, CONTINUE,
    DEFAULT, DELETE, DO, ELSE, FINALLY, FOR, FUNCTION,
    IF, IN, INSTANCEOF, LET, NEW, RETURN, SWITCH,
    THIS, THROW, TRY, VAR, VOID, WHILE, WITH,
    YIELD, UNDEFINED, NULL_TOKEN, TRUE, FALSE,
    
    // JavaScript操作符令牌（从100开始）
    DOT = 100, CALL, POST_INCREMENT, POST_DECREMENT, NOT, BITWISE_NOT,
    TYPEOF, UNARY_PLUS, UNARY_MINUS, EXPONENT, MULTIPLY, DIVIDE, REMAINDER,
    PLUS, MINUS, SHIFT_LEFT, SHIFT_RIGHT, ZERO_FILL_RIGHT_SHIFT,
    LESS_THAN, LESS_EQUAL, GREATER_THAN, GREATER_EQUAL,
    EQUAL, NOT_EQUAL, BITWISE_AND, BITWISE_XOR, BITWISE_OR,
    LOGICAL_AND, LOGICAL_OR, COLON, QUESTION,
    ASSIGN,
    
    // 复合赋值操作符
    PLUS_ASSIGN, MINUS_ASSIGN, MULTIPLY_ASSIGN, DIVIDE_ASSIGN,
    REMAINDER_ASSIGN, SHIFT_LEFT_ASSIGN, SHIFT_RIGHT_ASSIGN, ZERO_FILL_RIGHT_SHIFT_ASSIGN,
    BITWISE_AND_ASSIGN, BITWISE_XOR_ASSIGN, BITWISE_OR_ASSIGN,
    
    COMMA
};

enum class ValueType : uint8_t {
    OBJECT = 0, PROPERTY, STRING, UNDEFINED, NULL_VALUE,
    NUMBER, BOOLEAN, FUNCTION, CODE_REF, NATIVE_FUNCTION, ERROR
};

enum class ExecutionFlags : uint8_t {
    NONE = 0,
    NO_EXECUTE = 1,    // Parse only, don't execute
    IN_LOOP = 2,       // Currently inside a loop
    IN_CALL = 4,       // Currently inside a function call
    BREAK_FLAG = 8,    // Break statement executed
    RETURN_FLAG = 16   // Return statement executed
};

// Exception classes
class JSException : public std::runtime_error {
public:
    explicit JSException(const std::string& message) : std::runtime_error(message) {}
};

class JSParseException : public JSException {
public:
    explicit JSParseException(const std::string& message) : JSException("Parse error: " + message) {}
};

class JSRuntimeException : public JSException {
public:
    explicit JSRuntimeException(const std::string& message) : JSException("Runtime error: " + message) {}
};

class JSOutOfMemoryException : public JSException {
public:
    JSOutOfMemoryException() : JSException("Out of memory") {}
};

/**
 * JSValue - Represents a JavaScript value using NaN boxing technique
 * 
 * Uses IEEE 754 NaN encoding to pack all JS values into 64-bit integers:
 * - Numbers: Direct IEEE 754 double precision
 * - Other types: NaN marker + 4-bit type + 48-bit data
 */
class JSValue {
private:
    JSValueType value_;
    
    // NaN boxing constants
    static constexpr JSValueType NAN_MASK = 0x7FF0000000000000ULL;
    static constexpr JSValueType TYPE_MASK = 0x000F000000000000ULL;
    static constexpr JSValueType DATA_MASK = 0x0000FFFFFFFFFFFFULL;
    static constexpr int TYPE_SHIFT = 48;
    
    static JSValueType makeValue(ValueType type, uint64_t data) {
        return NAN_MASK | (static_cast<uint64_t>(type) << TYPE_SHIFT) | (data & DATA_MASK);
    }
    
    static bool isNaN(JSValueType v) {
        return (v >> 52) == 0x7FF;
    }
    
public:
    // Constructors
    JSValue() : value_(makeValue(ValueType::UNDEFINED, 0)) {}
    
    explicit JSValue(double num) {
        union { double d; JSValueType v; } u = {num};
        value_ = u.v;
    }
    
    explicit JSValue(bool b) : value_(makeValue(ValueType::BOOLEAN, b ? 1 : 0)) {}
    
    explicit JSValue(ValueType type, uint64_t data = 0) : value_(makeValue(type, data)) {}
    
    // Constructor from raw value (for C API compatibility)
    explicit JSValue(JSValueType rawValue) : value_(rawValue) {}
    
    // Static factory methods
    static JSValue undefined() { return JSValue(ValueType::UNDEFINED, 0); }
    static JSValue null() { return JSValue(ValueType::NULL_VALUE, 0); }
    static JSValue boolean(bool b) { return JSValue(b); }
    static JSValue number(double n) { return JSValue(n); }
    static JSValue string(JSOffset offset) { return JSValue(ValueType::STRING, offset); }
    static JSValue object(JSOffset offset) { return JSValue(ValueType::OBJECT, offset); }
    static JSValue function(JSOffset offset) { return JSValue(ValueType::FUNCTION, offset); }
    static JSValue nativeFunction(JSOffset offset) { return JSValue(ValueType::NATIVE_FUNCTION, offset); }
    static JSValue error(JSOffset offset) { return JSValue(ValueType::ERROR, offset); }
    
    // Type checking
    ValueType getType() const {
        return isNaN(value_) ? static_cast<ValueType>((value_ >> TYPE_SHIFT) & 0xF) : ValueType::NUMBER;
    }
    
    bool isUndefined() const { return getType() == ValueType::UNDEFINED; }
    bool isNull() const { return getType() == ValueType::NULL_VALUE; }
    bool isBoolean() const { return getType() == ValueType::BOOLEAN; }
    bool isNumber() const { return getType() == ValueType::NUMBER; }
    bool isString() const { return getType() == ValueType::STRING; }
    bool isObject() const { return getType() == ValueType::OBJECT; }
    bool isFunction() const { return getType() == ValueType::FUNCTION || getType() == ValueType::NATIVE_FUNCTION; }
    bool isError() const { return getType() == ValueType::ERROR; }
    
    // Value extraction
    double asNumber() const {
        if (isNumber()) {
            union { JSValueType v; double d; } u = {value_};
            return u.d;
        }
        return 0.0;
    }
    
    bool asBoolean() const {
        return getType() == ValueType::BOOLEAN && (value_ & DATA_MASK) != 0;
    }
    
    JSOffset asOffset() const {
        return static_cast<JSOffset>(value_ & DATA_MASK);
    }
    
    // Truthiness check
    bool isTruthy(JSEngine& engine) const;
    
    // Raw value access
    JSValueType getRawValue() const { return value_; }
};

/**
 * Memory entity for storing JS objects, properties, and strings
 */
struct MemoryEntity {
    JSOffset header;  // Type info + size/next pointer
    // Variable length data follows
};

/**
 * Lexical analyzer for JavaScript source code
 */
class Lexer {
public:
    const char* code_;
private:
    size_t length_;
    size_t position_;
    size_t tokenOffset_;
    size_t tokenLength_;
    TokenType currentToken_;
    JSValue tokenValue_;
    bool consumed_;
    
    // Character classification helpers
    static bool isSpace(char c) {
        return c == ' ' || c == '\r' || c == '\n' || c == '\t' || c == '\f' || c == '\v';
    }
    
    static bool isDigit(char c) { return c >= '0' && c <= '9'; }
    static bool isHexDigit(char c) {
        return isDigit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
    }
    
    static bool isAlpha(char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
    }
    
    static bool isIdentifierStart(char c) {
        return c == '_' || c == '$' || isAlpha(c);
    }
    
    static bool isIdentifierContinue(char c) {
        return c == '_' || c == '$' || isAlpha(c) || isDigit(c);
    }
    
    // Skip whitespace and comments
    void skipWhitespace();
    
    // Parse different token types
    TokenType parseNumber();
    TokenType parseString();
    TokenType parseIdentifier();
    TokenType parseOperator();
    TokenType parseKeyword(const std::string& identifier);
    
public:
    Lexer(const char* code, size_t length) 
        : code_(code), length_(length), position_(0), tokenOffset_(0), 
          tokenLength_(0), currentToken_(TokenType::ERROR), consumed_(true) {}
    
    // Get next token
    TokenType nextToken();
    
    // Token information
    TokenType getCurrentToken() const { return currentToken_; }
    JSValue getTokenValue() const { return tokenValue_; }
    size_t getTokenOffset() const { return tokenOffset_; }
    size_t getTokenLength() const { return tokenLength_; }
    bool isConsumed() const { return consumed_; }
    void setConsumed(bool consumed) { consumed_ = consumed; }
    
    // Position management
    size_t getPosition() const { return position_; }
    void setPosition(size_t pos) { position_ = pos; }
};

/**
 * Memory manager with linear allocation and mark-and-sweep garbage collection
 */
class MemoryManager {
private:
    std::unique_ptr<uint8_t[]> memory_;
    size_t size_;
    JSOffset breakPoint_;  // Current allocation boundary
    JSOffset gcThreshold_; // GC trigger threshold
    JSOffset noGcOffset_;  // Offset to exclude from GC
    
    // GC marking constants
    static constexpr JSOffset GC_MARK = 0x80000000U;
    
    // Entity size calculation
    JSOffset getEntitySize(JSOffset header) const;
    
    // GC implementation
    void markAllEntitiesForDeletion();
    void unmarkUsedEntities(JSValue rootScope);
    void deleteMarkedEntities();
    void fixupOffsets(JSOffset start, JSOffset size);
    JSOffset unmarkEntity(JSOffset offset);
    
    // Memory alignment
    static JSOffset align32(JSOffset value) {
        return ((value + 3) >> 2) << 2;
    }
    
public:
    MemoryManager(size_t size);
    ~MemoryManager() = default;
    
    // Memory allocation
    JSOffset allocate(size_t size);
    
    // Entity creation
    JSValue createString(const void* data, size_t length);
    JSValue createObject(JSOffset parentOffset = 0);
    JSValue createProperty(JSOffset nextProp, JSOffset keyOffset, JSValue value);
    
    // Memory access
    template<typename T>
    T load(JSOffset offset) const {
        T value;
        std::memcpy(&value, &memory_[offset], sizeof(T));
        return value;
    }
    
    template<typename T>
    void store(JSOffset offset, const T& value) {
        std::memcpy(&memory_[offset], &value, sizeof(T));
    }
    
    // String operations
    std::string getString(JSValue stringValue) const;
    size_t getStringLength(JSValue stringValue) const;
    
    // Garbage collection
    void runGC(JSValue rootScope);
    void setGCThreshold(JSOffset threshold) { gcThreshold_ = threshold; }
    void setNoGCOffset(JSOffset offset) { noGcOffset_ = offset; }
    
    // Statistics
    size_t getTotalSize() const { return size_; }
    size_t getUsedSize() const { return breakPoint_; }
    size_t getFreeSize() const { return size_ - breakPoint_; }
    
    // Raw memory access
    const uint8_t* getRawMemory() const { return memory_.get(); }
    uint8_t* getRawMemory() { return memory_.get(); }
};

/**
 * Scope management for variable resolution
 */
class Scope : public std::enable_shared_from_this<Scope> {
private:
    JSValue scopeObject_;
    std::shared_ptr<Scope> parent_;
    
public:
    Scope(JSValue scopeObj, std::shared_ptr<Scope> parent = nullptr)
        : scopeObject_(scopeObj), parent_(std::move(parent)) {}
    
    JSValue getScopeObject() const { return scopeObject_; }
    std::shared_ptr<Scope> getParent() const { return parent_; }
    
    // Variable operations
    JSValue getVariable(JSEngine& engine, const std::string& name);
    void setVariable(JSEngine& engine, const std::string& name, JSValue value);
    bool hasVariable(JSEngine& engine, const std::string& name);
};

/**
 * Parser for JavaScript expressions and statements
 */
class Parser {
private:
    JSEngine& engine_;
    Lexer& lexer_;
    
    // Operator precedence levels
    enum class Precedence {
        NONE = 0, ASSIGNMENT, TERNARY, LOGICAL_OR, LOGICAL_AND,
        BITWISE_OR, BITWISE_XOR, BITWISE_AND, EQUALITY, RELATIONAL,
        SHIFT, ADDITIVE, MULTIPLICATIVE, EXPONENTIATION, UNARY, POSTFIX, CALL, PRIMARY
    };
    
    // Expression parsing
    JSValue parseExpression(Precedence minPrec = Precedence::ASSIGNMENT);
    JSValue parsePrimaryExpression();
    JSValue parseUnaryExpression();
    JSValue parsePostfixExpression(JSValue left);
    JSValue parseBinaryExpression(JSValue left, TokenType op, Precedence prec);
    JSValue parseTernaryExpression(JSValue condition);
    JSValue parseCallExpression(JSValue function);
    JSValue parsePropertyAccess(JSValue object);
    
    // Literal parsing
    JSValue parseObjectLiteral();
    JSValue parseArrayLiteral();
    JSValue parseStringLiteral();
    JSValue parseNumberLiteral();
    
    // Statement parsing
    JSValue parseStatement();
    JSValue parseBlockStatement();
    JSValue parseVariableDeclaration();
    JSValue parseFunctionDeclaration();
    JSValue parseIfStatement();
    JSValue parseWhileStatement();
    JSValue parseForStatement();
    JSValue parseReturnStatement();
    JSValue parseBreakStatement();
    JSValue parseContinueStatement();
    JSValue parseExpressionStatement();
    
    // Utility functions
    void expectToken(TokenType expected);
    bool matchToken(TokenType token);
    Precedence getOperatorPrecedence(TokenType op);
    bool isRightAssociative(TokenType op);
    
public:
    Parser(JSEngine& engine, Lexer& lexer) : engine_(engine), lexer_(lexer) {}
    
    JSValue parse();
};

/**
 * Main JavaScript engine class
 */
class JSEngine {
private:
    std::unique_ptr<MemoryManager> memory_;
    std::shared_ptr<Scope> currentScope_;
    std::shared_ptr<Scope> globalScope_;
    std::unordered_map<std::string, JSNativeFunction> nativeFunctions_;
    
    // Execution state
    ExecutionFlags flags_;
    std::string errorMessage_;
    
    // Stack management for C function calls
    void* cStackBase_;
    size_t maxCStackSize_;
    size_t currentCStackSize_;
    
    // Built-in functions
    void registerBuiltinFunctions();
    
    // Object operations
    JSValue getProperty(JSValue object, const std::string& key);
    void setProperty(JSValue object, const std::string& key, JSValue value);
    bool hasProperty(JSValue object, const std::string& key);
    
    // Function execution
    JSValue callFunction(JSValue function, const std::vector<JSValue>& args);
    JSValue callNativeFunction(JSNativeFunction func, const std::vector<JSValue>& args);
    
    // Operator implementation
    JSValue executeUnaryOperator(TokenType op, JSValue operand);
    JSValue executeBinaryOperator(TokenType op, JSValue left, JSValue right);
    
    // Type conversion helpers
    double toNumber(JSValue value);
    std::string toString(JSValue value);
    bool toBoolean(JSValue value);
    
    // Error handling
    JSValue createError(const std::string& message);
    void setError(const std::string& message);
    
public:
    explicit JSEngine(size_t memorySize = 64 * 1024);
    ~JSEngine() = default;
    
    // Main evaluation function
    JSValue evaluate(const std::string& code);
    JSValue evaluate(const char* code, size_t length);
    
    // Value creation
    JSValue createUndefined() { return JSValue::undefined(); }
    JSValue createNull() { return JSValue::null(); }
    JSValue createBoolean(bool value) { return JSValue::boolean(value); }
    JSValue createNumber(double value) { return JSValue::number(value); }
    JSValue createString(const std::string& str);
    JSValue createString(const char* str, size_t length);
    JSValue createObject();
    JSValue createArray();
    
    // Native function registration
    void registerNativeFunction(const std::string& name, JSNativeFunction func);
    
    // Object property operations
    JSValue getObjectProperty(JSValue object, const std::string& key);
    void setObjectProperty(JSValue object, const std::string& key, JSValue value);
    
    // Global object access
    JSValue getGlobalObject();
    
    // Variable operations
    JSValue getVariable(const std::string& name);
    void setVariable(const std::string& name, JSValue value);
    
    // Utility functions
    std::string valueToString(JSValue value);
    bool isTruthy(JSValue value);
    
    // Memory management
    void runGarbageCollection();
    void setGCThreshold(size_t threshold);
    
    // Statistics and debugging
    size_t getTotalMemory() const;
    size_t getUsedMemory() const;
    size_t getFreeMemory() const;
    
    // Error handling
    bool hasError() const { return !errorMessage_.empty(); }
    std::string getErrorMessage() const { return errorMessage_; }
    void clearError() { errorMessage_.clear(); }
    
    // Execution flags
    void setExecutionFlag(ExecutionFlags flag) {
        flags_ = static_cast<ExecutionFlags>(static_cast<uint8_t>(flags_) | static_cast<uint8_t>(flag));
    }
    
    void clearExecutionFlag(ExecutionFlags flag) {
        flags_ = static_cast<ExecutionFlags>(static_cast<uint8_t>(flags_) & ~static_cast<uint8_t>(flag));
    }
    
    bool hasExecutionFlag(ExecutionFlags flag) const {
        return (static_cast<uint8_t>(flags_) & static_cast<uint8_t>(flag)) != 0;
    }
    
    // Memory manager access
    MemoryManager& getMemoryManager() { return *memory_; }
    const MemoryManager& getMemoryManager() const { return *memory_; }
    
    // Scope management
    std::shared_ptr<Scope> getCurrentScope() const { return currentScope_; }
    void pushScope(JSValue scopeObject);
    void popScope();
};

} // namespace simplejs