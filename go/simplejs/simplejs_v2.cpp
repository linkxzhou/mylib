/**
 * SimpleJS v2.0 - Modern C++11 JavaScript Interpreter Implementation
 * 
 * Implementation file for the SimpleJS v2.0 JavaScript interpreter.
 * Contains all method implementations for the classes declared in simplejs_v2.h
 */

#include "simplejs_v2.h"

namespace simplejs {

// ============================================================================
// JSValue Implementation
// ============================================================================

bool JSValue::isTruthy(JSEngine& engine) const {
    switch (getType()) {
        case ValueType::UNDEFINED:
        case ValueType::NULL_VALUE:
            return false;
        case ValueType::BOOLEAN:
            return asBoolean();
        case ValueType::NUMBER: {
            double num = asNumber();
            return num != 0.0 && !std::isnan(num);
        }
        case ValueType::STRING:
            return engine.getMemoryManager().getStringLength(*this) > 0;
        case ValueType::OBJECT:
        case ValueType::FUNCTION:
        case ValueType::NATIVE_FUNCTION:
            return true;
        default:
            return false;
    }
}

// ============================================================================
// MemoryManager Implementation
// ============================================================================

// Template functions are defined in the header file

MemoryManager::MemoryManager(size_t size) 
    : memory_(new uint8_t[size]), size_(size), 
      breakPoint_(0), gcThreshold_(static_cast<JSOffset>(size * DEFAULT_GC_THRESHOLD)), 
      noGcOffset_(0) {
    std::memset(memory_.get(), 0, size);
}

JSOffset MemoryManager::allocate(size_t size) {
    size = align32(static_cast<JSOffset>(size));
    if (breakPoint_ + size > size_) {
        throw JSOutOfMemoryException();
    }
    JSOffset offset = breakPoint_;
    breakPoint_ += static_cast<JSOffset>(size);
    return offset;
}

JSValue MemoryManager::createString(const void* data, size_t length) {
    JSOffset totalSize = sizeof(JSOffset) + align32(static_cast<JSOffset>(length + 1));
    JSOffset offset = allocate(totalSize);
    
    // Store length in header (shifted left by 2, with type bits)
    JSOffset header = (static_cast<JSOffset>(length + 1) << 2) | static_cast<JSOffset>(ValueType::STRING);
    store(offset, header);
    
    // Copy string data
    if (data) {
        std::memcpy(&memory_[offset + sizeof(JSOffset)], data, length);
    }
    memory_[offset + sizeof(JSOffset) + length] = '\0'; // Null terminate
    
    return JSValue::string(offset);
}

JSValue MemoryManager::createObject(JSOffset parentOffset) {
    JSOffset totalSize = 2 * sizeof(JSOffset);
    JSOffset offset = allocate(totalSize);
    
    // Object header: first property offset (0) + type
    JSOffset header = static_cast<JSOffset>(ValueType::OBJECT);
    store(offset, header);
    store(offset + sizeof(JSOffset), parentOffset);
    
    return JSValue::object(offset);
}

std::string MemoryManager::getString(JSValue stringValue) const {
    if (!stringValue.isString()) {
        return "";
    }
    
    JSOffset offset = stringValue.asOffset();
    JSOffset header = load<JSOffset>(offset);
    size_t length = (header >> 2) - 1; // Remove type bits and null terminator
    
    const char* data = reinterpret_cast<const char*>(&memory_[offset + sizeof(JSOffset)]);
    return std::string(data, length);
}

size_t MemoryManager::getStringLength(JSValue stringValue) const {
    if (!stringValue.isString()) {
        return 0;
    }
    
    JSOffset offset = stringValue.asOffset();
    JSOffset header = load<JSOffset>(offset);
    return (header >> 2) - 1; // Remove type bits and null terminator
}

JSOffset MemoryManager::getEntitySize(JSOffset header) const {
    switch (header & 3) {
        case static_cast<JSOffset>(ValueType::OBJECT):
            return 2 * sizeof(JSOffset);
        case static_cast<JSOffset>(ValueType::PROPERTY):
            return 2 * sizeof(JSOffset) + sizeof(JSValue);
        case static_cast<JSOffset>(ValueType::STRING):
            return sizeof(JSOffset) + align32(header >> 2);
        default:
            return 0;
    }
}

void MemoryManager::runGC(JSValue rootScope) {
    if (noGcOffset_ == static_cast<JSOffset>(~0)) return; // GC disabled
    
    markAllEntitiesForDeletion();
    unmarkUsedEntities(rootScope);
    deleteMarkedEntities();
}

void MemoryManager::markAllEntitiesForDeletion() {
    for (JSOffset off = 0; off < breakPoint_; ) {
        JSOffset header = load<JSOffset>(off);
        JSOffset size = getEntitySize(header);
        
        // Mark entity for deletion
        store(off, header | GC_MARK);
        off += size;
    }
}

void MemoryManager::unmarkUsedEntities(JSValue rootScope) {
    // Unmark root scope and all reachable entities
    if (rootScope.isObject()) {
        unmarkEntity(rootScope.asOffset());
    }
    
    // Unmark entities protected from GC
    if (noGcOffset_ != 0) {
        unmarkEntity(noGcOffset_);
    }
}

void MemoryManager::deleteMarkedEntities() {
    for (JSOffset off = 0; off < breakPoint_; ) {
        JSOffset header = load<JSOffset>(off);
        JSOffset size = getEntitySize(header & ~GC_MARK);
        
        if (header & GC_MARK) {
            // Entity marked for deletion
            fixupOffsets(off, size);
            
            // Move memory to remove the entity
            std::memmove(&memory_[off], &memory_[off + size], breakPoint_ - off - size);
            breakPoint_ -= size;
            // Don't increment off, check same position again
        } else {
            off += size;
        }
    }
}

void MemoryManager::fixupOffsets(JSOffset start, JSOffset size) {
    // Fix up all offsets that point beyond the deleted entity
    for (JSOffset off = 0; off < breakPoint_; ) {
        JSOffset header = load<JSOffset>(off);
        JSOffset entitySize = getEntitySize(header & ~GC_MARK);
        
        if (header & GC_MARK) {
            off += entitySize;
            continue; // Skip entities marked for deletion
        }
        
        ValueType type = static_cast<ValueType>(header & 3);
        
        if (type == ValueType::OBJECT) {
            // Fix object's first property pointer
            JSOffset propOffset = load<JSOffset>(off + sizeof(JSOffset));
            if (propOffset > start) {
                store(off + sizeof(JSOffset), propOffset - size);
            }
        } else if (type == ValueType::PROPERTY) {
            // Fix property's next pointer
            JSOffset nextProp = header & ~(GC_MARK | 3U);
            if (nextProp > start) {
                store(off, (nextProp - size) | static_cast<JSOffset>(type));
            }
            
            // Fix property's key offset
            JSOffset keyOffset = load<JSOffset>(off + sizeof(JSOffset));
            if (keyOffset > start) {
                store(off + sizeof(JSOffset), keyOffset - size);
            }
            
            // Fix property's value if it's a memory entity
            JSValue value = load<JSValue>(off + 2 * sizeof(JSOffset));
            if ((value.getType() == ValueType::OBJECT || value.getType() == ValueType::STRING || 
                 value.getType() == ValueType::FUNCTION) && value.asOffset() > start) {
                JSValue newValue(value.getType(), value.asOffset() - size);
                store(off + 2 * sizeof(JSOffset), newValue);
            }
        }
        
        off += entitySize;
    }
}

JSOffset MemoryManager::unmarkEntity(JSOffset offset) {
    if (offset >= breakPoint_) return 0;
    
    JSOffset header = load<JSOffset>(offset);
    if (!(header & GC_MARK)) return header; // Already unmarked
    
    // Unmark this entity
    header &= ~GC_MARK;
    store(offset, header);
    
    ValueType type = static_cast<ValueType>(header & 3);
    
    if (type == ValueType::OBJECT) {
        // Unmark first property
        JSOffset propOffset = load<JSOffset>(offset + sizeof(JSOffset));
        if (propOffset != 0) {
            unmarkEntity(propOffset);
        }
    } else if (type == ValueType::PROPERTY) {
        // Unmark next property
        JSOffset nextProp = header & ~3U;
        if (nextProp != 0) {
            unmarkEntity(nextProp);
        }
        
        // Unmark key
        JSOffset keyOffset = load<JSOffset>(offset + sizeof(JSOffset));
        unmarkEntity(keyOffset);
        
        // Unmark value if it's a memory entity
        JSValue value = load<JSValue>(offset + 2 * sizeof(JSOffset));
        if (value.getType() == ValueType::OBJECT || value.getType() == ValueType::STRING || 
            value.getType() == ValueType::FUNCTION) {
            unmarkEntity(value.asOffset());
        }
    }
    
    return header;
}

// ============================================================================
// Lexer Implementation
// ============================================================================

void Lexer::skipWhitespace() {
    while (position_ < length_) {
        if (isSpace(code_[position_])) {
            position_++;
        } else if (position_ + 1 < length_ && code_[position_] == '/' && code_[position_ + 1] == '/') {
            // Single line comment
            position_ += 2;
            while (position_ < length_ && code_[position_] != '\n') {
                position_++;
            }
        } else if (position_ + 3 < length_ && code_[position_] == '/' && code_[position_ + 1] == '*') {
            // Multi-line comment
            position_ += 2;
            while (position_ + 1 < length_) {
                if (code_[position_] == '*' && code_[position_ + 1] == '/') {
                    position_ += 2;
                    break;
                }
                position_++;
            }
        } else {
            break;
        }
    }
}

TokenType Lexer::nextToken() {
    if (!consumed_) {
        return currentToken_;
    }
    
    skipWhitespace();
    
    if (position_ >= length_) {
        currentToken_ = TokenType::END_OF_FILE;
        consumed_ = false;
        return currentToken_;
    }
    
    tokenOffset_ = position_;
    char c = code_[position_];
    
    // Numbers
    if (isDigit(c) || (c == '.' && position_ + 1 < length_ && isDigit(code_[position_ + 1]))) {
        currentToken_ = parseNumber();
    }
    // Strings
    else if (c == '"' || c == '\'') {
        currentToken_ = parseString();
    }
    // Identifiers and keywords
    else if (isIdentifierStart(c)) {
        currentToken_ = parseIdentifier();
    }
    // Operators and punctuation
    else {
        currentToken_ = parseOperator();
    }
    
    tokenLength_ = position_ - tokenOffset_;
    consumed_ = false;
    return currentToken_;
}

TokenType Lexer::parseNumber() {
    size_t start = position_;
    
    // Handle decimal point at start
    if (code_[position_] == '.') {
        position_++;
    }
    
    // Parse integer part
    while (position_ < length_ && isDigit(code_[position_])) {
        position_++;
    }
    
    // Parse decimal part
    if (position_ < length_ && code_[position_] == '.' && start != position_ - 1) {
        position_++;
        while (position_ < length_ && isDigit(code_[position_])) {
            position_++;
        }
    }
    
    // Parse exponent
    if (position_ < length_ && (code_[position_] == 'e' || code_[position_] == 'E')) {
        position_++;
        if (position_ < length_ && (code_[position_] == '+' || code_[position_] == '-')) {
            position_++;
        }
        while (position_ < length_ && isDigit(code_[position_])) {
            position_++;
        }
    }
    
    // Convert to number
    std::string numStr(code_ + start, position_ - start);
    double value = std::strtod(numStr.c_str(), nullptr);
    tokenValue_ = JSValue::number(value);
    
    return TokenType::NUMBER;
}

TokenType Lexer::parseString() {
    char quote = code_[position_++];
    std::string result;
    
    while (position_ < length_ && code_[position_] != quote) {
        if (code_[position_] == '\\' && position_ + 1 < length_) {
            position_++;
            switch (code_[position_]) {
                case 'n': result += '\n'; break;
                case 't': result += '\t'; break;
                case 'r': result += '\r'; break;
                case '\\': result += '\\'; break;
                case '"': result += '"'; break;
                case '\'': result += '\''; break;
                default: result += code_[position_]; break;
            }
        } else {
            result += code_[position_];
        }
        position_++;
    }
    
    if (position_ < length_) {
        position_++; // Skip closing quote
    }
    
    // Store string content for later use
    tokenValue_ = JSValue::string(0); // Placeholder, will be created by engine
    
    return TokenType::STRING;
}

TokenType Lexer::parseIdentifier() {
    size_t start = position_;
    
    while (position_ < length_ && isIdentifierContinue(code_[position_])) {
        position_++;
    }
    
    std::string identifier(code_ + start, position_ - start);
    TokenType keywordToken = parseKeyword(identifier);
    
    if (keywordToken != TokenType::IDENTIFIER) {
        return keywordToken;
    }
    
    return TokenType::IDENTIFIER;
}

TokenType Lexer::parseKeyword(const std::string& identifier) {
    static const std::unordered_map<std::string, TokenType> keywords = {
        {"break", TokenType::BREAK},
        {"case", TokenType::CASE},
        {"catch", TokenType::CATCH},
        {"class", TokenType::CLASS},
        {"const", TokenType::CONST},
        {"continue", TokenType::CONTINUE},
        {"default", TokenType::DEFAULT},
        {"delete", TokenType::DELETE},
        {"do", TokenType::DO},
        {"else", TokenType::ELSE},
        {"false", TokenType::FALSE},
        {"finally", TokenType::FINALLY},
        {"for", TokenType::FOR},
        {"function", TokenType::FUNCTION},
        {"if", TokenType::IF},
        {"in", TokenType::IN},
        {"instanceof", TokenType::INSTANCEOF},
        {"let", TokenType::LET},
        {"new", TokenType::NEW},
        {"null", TokenType::NULL_TOKEN},
        {"return", TokenType::RETURN},
        {"switch", TokenType::SWITCH},
        {"this", TokenType::THIS},
        {"throw", TokenType::THROW},
        {"true", TokenType::TRUE},
        {"try", TokenType::TRY},
        {"typeof", TokenType::TYPEOF},
        {"undefined", TokenType::UNDEFINED},
        {"var", TokenType::VAR},
        {"void", TokenType::VOID},
        {"while", TokenType::WHILE},
        {"with", TokenType::WITH},
        {"yield", TokenType::YIELD}
    };
    
    auto it = keywords.find(identifier);
    return (it != keywords.end()) ? it->second : TokenType::IDENTIFIER;
}

TokenType Lexer::parseOperator() {
    char c = code_[position_++];
    
    switch (c) {
        case '(': return TokenType::LEFT_PAREN;
        case ')': return TokenType::RIGHT_PAREN;
        case '{': return TokenType::LEFT_BRACE;
        case '}': return TokenType::RIGHT_BRACE;
        case ';': return TokenType::SEMICOLON;
        case ',': return TokenType::COMMA;
        case '.': return TokenType::DOT;
        case ':': return TokenType::COLON;
        case '?': return TokenType::QUESTION;
        case '~': return TokenType::BITWISE_NOT;
        
        case '+': 
            if (position_ < length_ && code_[position_] == '+') {
                position_++;
                return TokenType::POST_INCREMENT;
            } else if (position_ < length_ && code_[position_] == '=') {
                position_++;
                return TokenType::PLUS_ASSIGN;
            }
            return TokenType::PLUS;
            
        case '-':
            if (position_ < length_ && code_[position_] == '-') {
                position_++;
                return TokenType::POST_DECREMENT;
            } else if (position_ < length_ && code_[position_] == '=') {
                position_++;
                return TokenType::MINUS_ASSIGN;
            }
            return TokenType::MINUS;
            
        case '*':
            if (position_ < length_ && code_[position_] == '*') {
                position_++;
                return TokenType::EXPONENT;
            } else if (position_ < length_ && code_[position_] == '=') {
                position_++;
                return TokenType::MULTIPLY_ASSIGN;
            }
            return TokenType::MULTIPLY;
            
        case '/':
            if (position_ < length_ && code_[position_] == '=') {
                position_++;
                return TokenType::DIVIDE_ASSIGN;
            }
            return TokenType::DIVIDE;
            
        case '%':
            if (position_ < length_ && code_[position_] == '=') {
                position_++;
                return TokenType::REMAINDER_ASSIGN;
            }
            return TokenType::REMAINDER;
            
        case '=':
            if (position_ < length_ && code_[position_] == '=') {
                position_++;
                return TokenType::EQUAL;
            }
            return TokenType::ASSIGN;
            
        case '!':
            if (position_ < length_ && code_[position_] == '=') {
                position_++;
                return TokenType::NOT_EQUAL;
            }
            return TokenType::NOT;
            
        case '<':
            if (position_ < length_ && code_[position_] == '<') {
                position_++;
                if (position_ < length_ && code_[position_] == '=') {
                    position_++;
                    return TokenType::SHIFT_LEFT_ASSIGN;
                }
                return TokenType::SHIFT_LEFT;
            } else if (position_ < length_ && code_[position_] == '=') {
                position_++;
                return TokenType::LESS_EQUAL;
            }
            return TokenType::LESS_THAN;
            
        case '>':
            if (position_ < length_ && code_[position_] == '>') {
                position_++;
                if (position_ < length_ && code_[position_] == '>') {
                    position_++;
                    if (position_ < length_ && code_[position_] == '=') {
                        position_++;
                        return TokenType::ZERO_FILL_RIGHT_SHIFT_ASSIGN;
                    }
                    return TokenType::ZERO_FILL_RIGHT_SHIFT;
                } else if (position_ < length_ && code_[position_] == '=') {
                    position_++;
                    return TokenType::SHIFT_RIGHT_ASSIGN;
                }
                return TokenType::SHIFT_RIGHT;
            } else if (position_ < length_ && code_[position_] == '=') {
                position_++;
                return TokenType::GREATER_EQUAL;
            }
            return TokenType::GREATER_THAN;
            
        case '&':
            if (position_ < length_ && code_[position_] == '&') {
                position_++;
                return TokenType::LOGICAL_AND;
            } else if (position_ < length_ && code_[position_] == '=') {
                position_++;
                return TokenType::BITWISE_AND_ASSIGN;
            }
            return TokenType::BITWISE_AND;
            
        case '|':
            if (position_ < length_ && code_[position_] == '|') {
                position_++;
                return TokenType::LOGICAL_OR;
            } else if (position_ < length_ && code_[position_] == '=') {
                position_++;
                return TokenType::BITWISE_OR_ASSIGN;
            }
            return TokenType::BITWISE_OR;
            
        case '^':
            if (position_ < length_ && code_[position_] == '=') {
                position_++;
                return TokenType::BITWISE_XOR_ASSIGN;
            }
            return TokenType::BITWISE_XOR;
            
        default:
            return TokenType::ERROR;
    }
}

// ============================================================================
// Scope Implementation
// ============================================================================

JSValue Scope::getVariable(JSEngine& engine, const std::string& name) {
    // Search through scope chain
    std::shared_ptr<Scope> currentScope = shared_from_this();
    
    while (currentScope) {
        JSValue propValue = engine.getObjectProperty(currentScope->scopeObject_, name);
        if (!propValue.isUndefined()) {
            return propValue;
        }
        
        // Move to parent scope
        currentScope = currentScope->parent_;
    }
    
    // Variable not found in scope chain
    return JSValue::undefined();
}

void Scope::setVariable(JSEngine& engine, const std::string& name, JSValue value) {
    // Set variable in current scope
    engine.setObjectProperty(scopeObject_, name, value);
}

bool Scope::hasVariable(JSEngine& engine, const std::string& name) {
    // Check if variable exists in scope chain
    std::shared_ptr<Scope> currentScope = shared_from_this();
    
    while (currentScope) {
        JSValue propValue = engine.getObjectProperty(currentScope->scopeObject_, name);
        if (!propValue.isUndefined()) {
            return true;
        }
        
        // Move to parent scope
        currentScope = currentScope->parent_;
    }
    
    return false;
}

// ============================================================================
// JSEngine Implementation
// ============================================================================

JSEngine::JSEngine(size_t memorySize) 
    : memory_(new MemoryManager(memorySize)),
      flags_(ExecutionFlags::NONE),
      cStackBase_(nullptr),
      maxCStackSize_(1024 * 1024), // 1MB default
      currentCStackSize_(0) {
    
    // Create global scope
    JSValue globalObj = memory_->createObject();
    globalScope_ = std::make_shared<Scope>(globalObj);
    currentScope_ = globalScope_;
    
    registerBuiltinFunctions();
}

void JSEngine::registerBuiltinFunctions() {
    // Register basic built-in functions
    registerNativeFunction("print", [](JSEngine& engine, const std::vector<JSValue>& args) -> JSValue {
        for (size_t i = 0; i < args.size(); ++i) {
            if (i > 0) std::printf(" ");
            std::printf("%s", engine.valueToString(args[i]).c_str());
        }
        std::printf("\n");
        return JSValue::undefined();
    });
    
    registerNativeFunction("typeof", [](JSEngine& engine, const std::vector<JSValue>& args) -> JSValue {
        if (args.empty()) {
            return engine.createString("undefined");
        }
        
        JSValue arg = args[0];
        switch (arg.getType()) {
            case ValueType::UNDEFINED: return engine.createString("undefined");
            case ValueType::NULL_VALUE: return engine.createString("object");
            case ValueType::BOOLEAN: return engine.createString("boolean");
            case ValueType::NUMBER: return engine.createString("number");
            case ValueType::STRING: return engine.createString("string");
            case ValueType::FUNCTION:
            case ValueType::NATIVE_FUNCTION: return engine.createString("function");
            default: return engine.createString("object");
        }
    });
}

JSValue JSEngine::evaluate(const std::string& code) {
    return evaluate(code.c_str(), code.length());
}

JSValue JSEngine::evaluate(const char* code, size_t length) {
    try {
        clearError();
        
        Lexer lexer(code, length);
        Parser parser(*this, lexer);
        
        return parser.parse();
    } catch (const JSException& e) {
        setError(e.what());
        return createError(e.what());
    } catch (const std::exception& e) {
        setError(std::string("Internal error: ") + e.what());
        return createError(e.what());
    }
}

JSValue JSEngine::createString(const std::string& str) {
    return memory_->createString(str.c_str(), str.length());
}

JSValue JSEngine::createString(const char* str, size_t length) {
    return memory_->createString(str, length);
}

JSValue JSEngine::createObject() {
    return memory_->createObject();
}

JSValue JSEngine::createArray() {
    // Arrays are objects with special properties
    JSValue array = createObject();
    setObjectProperty(array, "length", createNumber(0));
    return array;
}

void JSEngine::registerNativeFunction(const std::string& name, JSNativeFunction func) {
    nativeFunctions_[name] = std::move(func);
    // Also add to global scope
    setVariable(name, JSValue::nativeFunction(0)); // Placeholder offset
}

std::string JSEngine::valueToString(JSValue value) {
    switch (value.getType()) {
        case ValueType::UNDEFINED:
            return "undefined";
        case ValueType::NULL_VALUE:
            return "null";
        case ValueType::BOOLEAN:
            return value.asBoolean() ? "true" : "false";
        case ValueType::NUMBER: {
            double num = value.asNumber();
            if (std::floor(num) == num && num >= -2147483648.0 && num <= 2147483647.0) {
                return std::to_string(static_cast<int64_t>(num));
            } else {
                return std::to_string(num);
            }
        }
        case ValueType::STRING:
            return memory_->getString(value);
        case ValueType::OBJECT:
            return "[object Object]";
        case ValueType::FUNCTION:
        case ValueType::NATIVE_FUNCTION:
            return "[function Function]";
        case ValueType::ERROR:
            return "[Error]";
        default:
            return "[unknown]";
    }
}

bool JSEngine::isTruthy(JSValue value) {
    return value.isTruthy(*this);
}

JSValue JSEngine::getGlobalObject() {
    return globalScope_->getScopeObject();
}

JSValue JSEngine::getVariable(const std::string& name) {
    return currentScope_->getVariable(*this, name);
}

void JSEngine::setVariable(const std::string& name, JSValue value) {
    currentScope_->setVariable(*this, name, value);
}

JSValue JSEngine::getObjectProperty(JSValue object, const std::string& key) {
    if (!object.isObject()) {
        return JSValue::undefined();
    }
    
    // Search for property in the object
    JSOffset objOffset = object.asOffset();
    JSOffset propOffset = memory_->load<JSOffset>(objOffset) & ~3U; // Load first property offset
    
    while (propOffset < memory_->getUsedSize() && propOffset != 0) {
        // Load property key offset
        JSOffset keyOffset = memory_->load<JSOffset>(propOffset + sizeof(JSOffset));
        
        // Get key string
        JSValue keyValue = JSValue::string(keyOffset);
        std::string propKey = memory_->getString(keyValue);
        
        if (propKey == key) {
            // Found the property, return its value
            JSValue propValue = memory_->load<JSValue>(propOffset + 2 * sizeof(JSOffset));
            return propValue;
        }
        
        // Move to next property
        propOffset = memory_->load<JSOffset>(propOffset) & ~3U;
    }
    
    return JSValue::undefined();
}

void JSEngine::setObjectProperty(JSValue object, const std::string& key, JSValue value) {
    if (!object.isObject()) {
        return;
    }
    
    // Create key string
    JSValue keyValue = createString(key);
    JSOffset keyOffset = keyValue.asOffset();
    
    // Get object head offset
    JSOffset objOffset = object.asOffset();
    JSOffset currentFirstProp = memory_->load<JSOffset>(objOffset);
    
    // Create new property
    JSOffset propSize = 2 * sizeof(JSOffset) + sizeof(JSValue);
    JSOffset newPropOffset = memory_->allocate(propSize);
    
    // Set property type and link to previous first property
    JSOffset propHeader = (currentFirstProp & ~3U) | static_cast<JSOffset>(ValueType::PROPERTY);
    memory_->store(newPropOffset, propHeader);
    
    // Store key offset
    memory_->store(newPropOffset + sizeof(JSOffset), keyOffset);
    
    // Store value
    memory_->store(newPropOffset + 2 * sizeof(JSOffset), value);
    
    // Update object to point to new property as first property
    JSOffset newObjHeader = newPropOffset | static_cast<JSOffset>(ValueType::OBJECT);
    memory_->store(objOffset, newObjHeader);
}

void JSEngine::runGarbageCollection() {
    memory_->runGC(currentScope_->getScopeObject());
}

void JSEngine::setGCThreshold(size_t threshold) {
    memory_->setGCThreshold(static_cast<JSOffset>(threshold));
}

size_t JSEngine::getTotalMemory() const {
    return memory_->getTotalSize();
}

size_t JSEngine::getUsedMemory() const {
    return memory_->getUsedSize();
}

size_t JSEngine::getFreeMemory() const {
    return memory_->getFreeSize();
}

JSValue JSEngine::createError(const std::string& message) {
    JSValue errorStr = createString(message);
    return JSValue::error(errorStr.asOffset());
}

void JSEngine::setError(const std::string& message) {
    errorMessage_ = message;
}

void JSEngine::pushScope(JSValue scopeObject) {
    currentScope_ = std::make_shared<Scope>(scopeObject, currentScope_);
}

void JSEngine::popScope() {
    if (currentScope_->getParent()) {
        currentScope_ = currentScope_->getParent();
    }
}

// Operator execution methods
JSValue JSEngine::executeUnaryOperator(TokenType op, JSValue operand) {
    switch (op) {
        case TokenType::NOT:
            return JSValue::boolean(!isTruthy(operand));
        case TokenType::UNARY_MINUS:
            if (operand.isNumber()) {
                return JSValue::number(-operand.asNumber());
            }
            break;
        case TokenType::UNARY_PLUS:
            if (operand.isNumber()) {
                return operand;
            } else {
                return JSValue::number(toNumber(operand));
            }
        case TokenType::TYPEOF: {
            switch (operand.getType()) {
                case ValueType::UNDEFINED: return createString("undefined");
                case ValueType::NULL_VALUE: return createString("object");
                case ValueType::BOOLEAN: return createString("boolean");
                case ValueType::NUMBER: return createString("number");
                case ValueType::STRING: return createString("string");
                case ValueType::FUNCTION:
                case ValueType::NATIVE_FUNCTION: return createString("function");
                default: return createString("object");
            }
        }
        default:
            break;
    }
    return JSValue::undefined();
}

JSValue JSEngine::executeBinaryOperator(TokenType op, JSValue left, JSValue right) {
    switch (op) {
        case TokenType::PLUS:
            if (left.isString() || right.isString()) {
                return createString(toString(left) + toString(right));
            } else if (left.isNumber() && right.isNumber()) {
                return JSValue::number(left.asNumber() + right.asNumber());
            }
            break;
        case TokenType::MINUS:
            if (left.isNumber() && right.isNumber()) {
                return JSValue::number(left.asNumber() - right.asNumber());
            }
            break;
        case TokenType::MULTIPLY:
            if (left.isNumber() && right.isNumber()) {
                return JSValue::number(left.asNumber() * right.asNumber());
            }
            break;
        case TokenType::DIVIDE:
            if (left.isNumber() && right.isNumber()) {
                double rightVal = right.asNumber();
                if (rightVal != 0.0) {
                    return JSValue::number(left.asNumber() / rightVal);
                }
                return JSValue::number(std::numeric_limits<double>::infinity());
            }
            break;
        case TokenType::REMAINDER:
            if (left.isNumber() && right.isNumber()) {
                return JSValue::number(std::fmod(left.asNumber(), right.asNumber()));
            }
            break;
        case TokenType::EQUAL:
            return JSValue::boolean(toString(left) == toString(right));
        case TokenType::NOT_EQUAL:
            return JSValue::boolean(toString(left) != toString(right));
        case TokenType::LESS_THAN:
            if (left.isNumber() && right.isNumber()) {
                return JSValue::boolean(left.asNumber() < right.asNumber());
            }
            break;
        case TokenType::GREATER_THAN:
            if (left.isNumber() && right.isNumber()) {
                return JSValue::boolean(left.asNumber() > right.asNumber());
            }
            break;
        case TokenType::LESS_EQUAL:
            if (left.isNumber() && right.isNumber()) {
                return JSValue::boolean(left.asNumber() <= right.asNumber());
            }
            break;
        case TokenType::GREATER_EQUAL:
            if (left.isNumber() && right.isNumber()) {
                return JSValue::boolean(left.asNumber() >= right.asNumber());
            }
            break;
        case TokenType::LOGICAL_AND:
            return isTruthy(left) ? right : left;
        case TokenType::LOGICAL_OR:
            return isTruthy(left) ? left : right;
        default:
            break;
    }
    return JSValue::undefined();
}

// Type conversion implementations
double JSEngine::toNumber(JSValue value) {
    switch (value.getType()) {
        case ValueType::NUMBER:
            return value.asNumber();
        case ValueType::BOOLEAN:
            return value.asBoolean() ? 1.0 : 0.0;
        case ValueType::STRING: {
            std::string str = memory_->getString(value);
            if (str.empty()) return 0.0;
            try {
                return std::stod(str);
            } catch (...) {
                return std::numeric_limits<double>::quiet_NaN();
            }
        }
        case ValueType::UNDEFINED:
            return std::numeric_limits<double>::quiet_NaN();
        case ValueType::NULL_VALUE:
            return 0.0;
        default:
            return std::numeric_limits<double>::quiet_NaN();
    }
}

std::string JSEngine::toString(JSValue value) {
    return valueToString(value);
}

bool JSEngine::toBoolean(JSValue value) {
    return isTruthy(value);
}

// ============================================================================
// Parser Implementation
// ============================================================================

JSValue Parser::parse() {
    JSValue result = JSValue::undefined();
    
    while (lexer_.nextToken() != TokenType::END_OF_FILE) {
        result = parseStatement();
        if (engine_.hasError()) {
            break;
        }
    }
    
    return result;
}

JSValue Parser::parseStatement() {
    TokenType token = lexer_.getCurrentToken();
    
    switch (token) {
        case TokenType::VAR:
        case TokenType::LET:
        case TokenType::CONST:
            return parseVariableDeclaration();
        case TokenType::FUNCTION:
            return parseFunctionDeclaration();
        case TokenType::IF:
            return parseIfStatement();
        case TokenType::WHILE:
            return parseWhileStatement();
        case TokenType::FOR:
            return parseForStatement();
        case TokenType::RETURN:
            return parseReturnStatement();
        case TokenType::BREAK:
            return parseBreakStatement();
        case TokenType::CONTINUE:
            return parseContinueStatement();
        case TokenType::LEFT_BRACE:
            return parseBlockStatement();
        default:
            return parseExpressionStatement();
    }
}

JSValue Parser::parseExpression(Precedence minPrec) {
    JSValue left = parsePrimaryExpression();
    
    while (true) {
        TokenType op = lexer_.getCurrentToken();
        Precedence prec = getOperatorPrecedence(op);
        
        if (prec < minPrec) {
            break;
        }
        
        lexer_.setConsumed(true);
        lexer_.nextToken();
        
        if (op == TokenType::QUESTION) {
            left = parseTernaryExpression(left);
        } else {
            Precedence nextMinPrec = isRightAssociative(op) ? prec : static_cast<Precedence>(static_cast<int>(prec) + 1);
            JSValue right = parseExpression(nextMinPrec);
            left = parseBinaryExpression(left, op, prec);
            // Use right to avoid warning
            (void)right;
        }
    }
    
    return left;
}

JSValue Parser::parsePrimaryExpression() {
    TokenType token = lexer_.getCurrentToken();
    lexer_.setConsumed(true);
    
    switch (token) {
        case TokenType::UNDEFINED:
            return JSValue::undefined();
        case TokenType::NULL_TOKEN:
            return JSValue::null();
        case TokenType::TRUE:
            return JSValue::boolean(true);
        case TokenType::FALSE:
            return JSValue::boolean(false);
        case TokenType::NUMBER:
            return lexer_.getTokenValue();
        case TokenType::STRING: {
            // Extract string content from lexer
            size_t tokenStart = lexer_.getTokenOffset();
            size_t tokenLen = lexer_.getTokenLength();
            const char* code = lexer_.code_ + tokenStart;
            
            // Skip opening quote
            if (tokenLen >= 2 && (code[0] == '"' || code[0] == '\'')) {
                std::string content;
                for (size_t i = 1; i < tokenLen - 1; ++i) {
                    if (code[i] == '\\' && i + 1 < tokenLen - 1) {
                        switch (code[i + 1]) {
                            case 'n': content += '\n'; i++; break;
                            case 't': content += '\t'; i++; break;
                            case 'r': content += '\r'; i++; break;
                            case '\\': content += '\\'; i++; break;
                            case '"': content += '"'; i++; break;
                            case '\'': content += '\''; i++; break;
                            default: content += code[i]; break;
                        }
                    } else {
                        content += code[i];
                    }
                }
                return engine_.createString(content);
            }
            return engine_.createString("");
        }
        case TokenType::IDENTIFIER: {
            // Extract identifier name and resolve variable
            size_t tokenStart = lexer_.getTokenOffset();
            size_t tokenLen = lexer_.getTokenLength();
            std::string identifier(lexer_.code_ + tokenStart, tokenLen);
            return engine_.getVariable(identifier);
        }
        case TokenType::LEFT_PAREN: {
            lexer_.nextToken();
            JSValue expr = parseExpression();
            expectToken(TokenType::RIGHT_PAREN);
            return expr;
        }
        case TokenType::LEFT_BRACE:
            return parseObjectLiteral();
        default:
            throw JSParseException("Unexpected token in expression");
    }
}

JSValue Parser::parseVariableDeclaration() {
    lexer_.setConsumed(true);
    lexer_.nextToken();
    
    // Expect identifier
    if (lexer_.getCurrentToken() != TokenType::IDENTIFIER) {
        throw JSParseException("Expected identifier in variable declaration");
    }
    
    // Extract variable name
    size_t tokenStart = lexer_.getTokenOffset();
    size_t tokenLen = lexer_.getTokenLength();
    std::string varName(lexer_.code_ + tokenStart, tokenLen);
    
    lexer_.setConsumed(true);
    lexer_.nextToken();
    
    JSValue initialValue = JSValue::undefined();
    
    // Check for initializer
    if (lexer_.getCurrentToken() == TokenType::ASSIGN) {
        lexer_.setConsumed(true);
        lexer_.nextToken();
        initialValue = parseExpression();
    }
    
    // Set variable in current scope
    engine_.setVariable(varName, initialValue);
    
    return initialValue;
}

JSValue Parser::parseExpressionStatement() {
    JSValue expr = parseExpression();
    
    // Optional semicolon
    if (lexer_.getCurrentToken() == TokenType::SEMICOLON) {
        lexer_.setConsumed(true);
        lexer_.nextToken();
    }
    
    return expr;
}

JSValue Parser::parseObjectLiteral() {
    expectToken(TokenType::LEFT_BRACE);
    JSValue obj = engine_.createObject();
    
    while (lexer_.getCurrentToken() != TokenType::RIGHT_BRACE) {
        // TODO: Parse property key-value pairs
        break; // Placeholder
    }
    
    expectToken(TokenType::RIGHT_BRACE);
    return obj;
}

void Parser::expectToken(TokenType expected) {
    if (lexer_.getCurrentToken() != expected) {
        throw JSParseException("Expected different token");
    }
    lexer_.setConsumed(true);
    lexer_.nextToken();
}

Parser::Precedence Parser::getOperatorPrecedence(TokenType op) {
    switch (op) {
        case TokenType::ASSIGN:
        case TokenType::PLUS_ASSIGN:
        case TokenType::MINUS_ASSIGN:
            return Precedence::ASSIGNMENT;
        case TokenType::QUESTION:
            return Precedence::TERNARY;
        case TokenType::LOGICAL_OR:
            return Precedence::LOGICAL_OR;
        case TokenType::LOGICAL_AND:
            return Precedence::LOGICAL_AND;
        case TokenType::BITWISE_OR:
            return Precedence::BITWISE_OR;
        case TokenType::BITWISE_XOR:
            return Precedence::BITWISE_XOR;
        case TokenType::BITWISE_AND:
            return Precedence::BITWISE_AND;
        case TokenType::EQUAL:
        case TokenType::NOT_EQUAL:
            return Precedence::EQUALITY;
        case TokenType::LESS_THAN:
        case TokenType::LESS_EQUAL:
        case TokenType::GREATER_THAN:
        case TokenType::GREATER_EQUAL:
            return Precedence::RELATIONAL;
        case TokenType::SHIFT_LEFT:
        case TokenType::SHIFT_RIGHT:
        case TokenType::ZERO_FILL_RIGHT_SHIFT:
            return Precedence::SHIFT;
        case TokenType::PLUS:
        case TokenType::MINUS:
            return Precedence::ADDITIVE;
        case TokenType::MULTIPLY:
        case TokenType::DIVIDE:
        case TokenType::REMAINDER:
            return Precedence::MULTIPLICATIVE;
        case TokenType::EXPONENT:
            return Precedence::EXPONENTIATION;
        default:
            return Precedence::NONE;
    }
}

bool Parser::isRightAssociative(TokenType op) {
    return op == TokenType::ASSIGN || op == TokenType::EXPONENT;
}

// Binary and ternary expression implementations
JSValue Parser::parseBinaryExpression(JSValue left, TokenType op, Precedence /*prec*/) {
    // Execute binary operation based on operator type
    switch (op) {
        case TokenType::PLUS: {
            if (left.isString()) {
                // String concatenation
                std::string leftStr = engine_.valueToString(left);
                JSValue right = parseExpression(Precedence::ADDITIVE);
                std::string rightStr = engine_.valueToString(right);
                return engine_.createString(leftStr + rightStr);
            } else if (left.isNumber()) {
                // Numeric addition
                JSValue right = parseExpression(Precedence::ADDITIVE);
                if (right.isNumber()) {
                    return JSValue::number(left.asNumber() + right.asNumber());
                }
            }
            break;
        }
        case TokenType::MINUS:
            if (left.isNumber()) {
                JSValue right = parseExpression(Precedence::ADDITIVE);
                if (right.isNumber()) {
                    return JSValue::number(left.asNumber() - right.asNumber());
                }
            }
            break;
        case TokenType::MULTIPLY:
            if (left.isNumber()) {
                JSValue right = parseExpression(Precedence::MULTIPLICATIVE);
                if (right.isNumber()) {
                    return JSValue::number(left.asNumber() * right.asNumber());
                }
            }
            break;
        case TokenType::DIVIDE:
            if (left.isNumber()) {
                JSValue right = parseExpression(Precedence::MULTIPLICATIVE);
                if (right.isNumber()) {
                    double rightVal = right.asNumber();
                    if (rightVal != 0.0) {
                        return JSValue::number(left.asNumber() / rightVal);
                    }
                }
            }
            break;
        case TokenType::EQUAL:
            // Equality comparison
            return JSValue::boolean(engine_.valueToString(left) == engine_.valueToString(parseExpression(Precedence::EQUALITY)));
        case TokenType::NOT_EQUAL:
            // Inequality comparison
            return JSValue::boolean(engine_.valueToString(left) != engine_.valueToString(parseExpression(Precedence::EQUALITY)));
        case TokenType::LESS_THAN:
            if (left.isNumber()) {
                JSValue right = parseExpression(Precedence::RELATIONAL);
                if (right.isNumber()) {
                    return JSValue::boolean(left.asNumber() < right.asNumber());
                }
            }
            break;
        case TokenType::GREATER_THAN:
            if (left.isNumber()) {
                JSValue right = parseExpression(Precedence::RELATIONAL);
                if (right.isNumber()) {
                    return JSValue::boolean(left.asNumber() > right.asNumber());
                }
            }
            break;
        case TokenType::ASSIGN: {
            // Assignment operation
            JSValue right = parseExpression(Precedence::ASSIGNMENT);
            // For now, return the right value
            return right;
        }
        default:
            break;
    }
    return JSValue::undefined();
}

JSValue Parser::parseTernaryExpression(JSValue condition) {
    // Parse ternary operator: condition ? trueExpr : falseExpr
    expectToken(TokenType::QUESTION);
    JSValue trueExpr = parseExpression(Precedence::ASSIGNMENT);
    expectToken(TokenType::COLON);
    JSValue falseExpr = parseExpression(Precedence::TERNARY);
    
    // Evaluate condition and return appropriate expression
    return engine_.isTruthy(condition) ? trueExpr : falseExpr;
}

// Placeholder implementations for remaining Parser methods
JSValue Parser::parseUnaryExpression() { return JSValue::undefined(); }
JSValue Parser::parsePostfixExpression(JSValue /*left*/) { return JSValue::undefined(); }
JSValue Parser::parseCallExpression(JSValue /*function*/) { return JSValue::undefined(); }
JSValue Parser::parsePropertyAccess(JSValue /*object*/) { return JSValue::undefined(); }
JSValue Parser::parseArrayLiteral() { return JSValue::undefined(); }
JSValue Parser::parseStringLiteral() { return JSValue::undefined(); }
JSValue Parser::parseNumberLiteral() { return JSValue::undefined(); }
JSValue Parser::parseBlockStatement() { return JSValue::undefined(); }
JSValue Parser::parseFunctionDeclaration() { return JSValue::undefined(); }
JSValue Parser::parseIfStatement() { return JSValue::undefined(); }
JSValue Parser::parseWhileStatement() { return JSValue::undefined(); }
JSValue Parser::parseForStatement() { return JSValue::undefined(); }
JSValue Parser::parseReturnStatement() { return JSValue::undefined(); }
JSValue Parser::parseBreakStatement() { return JSValue::undefined(); }
JSValue Parser::parseContinueStatement() { return JSValue::undefined(); }
bool Parser::matchToken(TokenType /*token*/) { return false; }

// Placeholder implementations for remaining JSEngine methods
JSValue JSEngine::getProperty(JSValue /*object*/, const std::string& /*key*/) { return JSValue::undefined(); }
void JSEngine::setProperty(JSValue /*object*/, const std::string& /*key*/, JSValue /*value*/) {}
bool JSEngine::hasProperty(JSValue /*object*/, const std::string& /*key*/) { return false; }
JSValue JSEngine::callFunction(JSValue /*function*/, const std::vector<JSValue>& /*args*/) { return JSValue::undefined(); }
JSValue JSEngine::callNativeFunction(JSNativeFunction /*func*/, const std::vector<JSValue>& /*args*/) { return JSValue::undefined(); }

// Placeholder implementations for remaining MemoryManager methods
JSValue MemoryManager::createProperty(JSOffset /*nextProp*/, JSOffset /*keyOffset*/, JSValue /*value*/) { return JSValue::undefined(); }

} // namespace simplejs