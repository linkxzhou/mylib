#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <dirent.h>
#include <sys/stat.h>
#include <vector>
#include <algorithm>
#include "simplejs_v2.h"

using namespace simplejs;

int main() {
    std::cout << "SimpleJS v2.0 Test Program" << std::endl;
    std::cout << "Version: " << VERSION << std::endl;
    std::cout << "==========================" << std::endl;
    
    try {
        // Create JS engine with 64KB memory
        JSEngine engine(64 * 1024);
        
        std::cout << "\n1. Testing basic value creation:" << std::endl;
        
        // Test basic value creation
        JSValue undef = engine.createUndefined();
        JSValue null = engine.createNull();
        JSValue boolTrue = engine.createBoolean(true);
        JSValue boolFalse = engine.createBoolean(false);
        JSValue num = engine.createNumber(42.5);
        JSValue str = engine.createString("Hello, World!");
        
        std::cout << "undefined: " << engine.valueToString(undef) << std::endl;
        std::cout << "null: " << engine.valueToString(null) << std::endl;
        std::cout << "true: " << engine.valueToString(boolTrue) << std::endl;
        std::cout << "false: " << engine.valueToString(boolFalse) << std::endl;
        std::cout << "number: " << engine.valueToString(num) << std::endl;
        std::cout << "string: " << engine.valueToString(str) << std::endl;
        
        std::cout << "\n2. Testing type checking:" << std::endl;
        std::cout << "undef.isUndefined(): " << (undef.isUndefined() ? "true" : "false") << std::endl;
        std::cout << "null.isNull(): " << (null.isNull() ? "true" : "false") << std::endl;
        std::cout << "boolTrue.isBoolean(): " << (boolTrue.isBoolean() ? "true" : "false") << std::endl;
        std::cout << "num.isNumber(): " << (num.isNumber() ? "true" : "false") << std::endl;
        std::cout << "str.isString(): " << (str.isString() ? "true" : "false") << std::endl;
        
        std::cout << "\n3. Testing truthiness:" << std::endl;
        std::cout << "isTruthy(undefined): " << (engine.isTruthy(undef) ? "true" : "false") << std::endl;
        std::cout << "isTruthy(null): " << (engine.isTruthy(null) ? "true" : "false") << std::endl;
        std::cout << "isTruthy(true): " << (engine.isTruthy(boolTrue) ? "true" : "false") << std::endl;
        std::cout << "isTruthy(false): " << (engine.isTruthy(boolFalse) ? "true" : "false") << std::endl;
        std::cout << "isTruthy(42.5): " << (engine.isTruthy(num) ? "true" : "false") << std::endl;
        std::cout << "isTruthy(\"Hello\"): " << (engine.isTruthy(str) ? "true" : "false") << std::endl;
        
        std::cout << "\n4. Testing object creation:" << std::endl;
        JSValue obj = engine.createObject();
        JSValue arr = engine.createArray();
        std::cout << "object: " << engine.valueToString(obj) << std::endl;
        std::cout << "array: " << engine.valueToString(arr) << std::endl;
        std::cout << "obj.isObject(): " << (obj.isObject() ? "true" : "false") << std::endl;
        std::cout << "arr.isObject(): " << (arr.isObject() ? "true" : "false") << std::endl;
        
        std::cout << "\n5. Testing memory statistics:" << std::endl;
        std::cout << "Total memory: " << engine.getTotalMemory() << " bytes" << std::endl;
        std::cout << "Used memory: " << engine.getUsedMemory() << " bytes" << std::endl;
        std::cout << "Free memory: " << engine.getFreeMemory() << " bytes" << std::endl;
        
        std::cout << "\n6. Testing lexer:" << std::endl;
        const char* testCode = "let x = 42; // comment\nfunction test() { return true; }";
        Lexer lexer(testCode, strlen(testCode));
        
        std::cout << "Tokenizing: " << testCode << std::endl;
        TokenType token;
        int tokenCount = 0;
        while ((token = lexer.nextToken()) != TokenType::END_OF_FILE && tokenCount < 10) {
            std::cout << "Token " << tokenCount << ": " << static_cast<int>(token) << std::endl;
            lexer.setConsumed(true);
            tokenCount++;
        }
        
        std::cout << "\n7. Testing simple evaluation (basic):" << std::endl;
        try {
            JSValue result1 = engine.evaluate("42");
            std::cout << "eval(\"42\"): " << engine.valueToString(result1) << std::endl;
            
            JSValue result2 = engine.evaluate("true");
            std::cout << "eval(\"true\"): " << engine.valueToString(result2) << std::endl;
            
            JSValue result3 = engine.evaluate("\"hello\"");
            std::cout << "eval(\"\\\"hello\\\"\"): " << engine.valueToString(result3) << std::endl;
        } catch (const JSException& e) {
            std::cout << "Evaluation error: " << e.what() << std::endl;
        }
        
        std::cout << "\n✅ All basic tests completed successfully!" << std::endl;
        
        // 8. Testing JavaScript files from examples directory
        std::cout << "\n8. Testing JavaScript files from examples directory:" << std::endl;
        std::cout << "====================================================" << std::endl;
        
        const std::string examplesDir = "examples";
        std::vector<std::string> jsFiles;
        
        // Read directory and collect .js files
        DIR* dir = opendir(examplesDir.c_str());
        if (dir) {
            struct dirent* entry;
            while ((entry = readdir(dir)) != nullptr) {
                std::string filename = entry->d_name;
                if (filename.length() > 3 && filename.substr(filename.length() - 3) == ".js") {
                    jsFiles.push_back(filename);
                }
            }
            closedir(dir);
            
            // Sort files for consistent output
            std::sort(jsFiles.begin(), jsFiles.end());
            
            // Process each JavaScript file
            for (const auto& filename : jsFiles) {
                std::string filepath = examplesDir + "/" + filename;
                
                std::cout << "\n📄 Testing file: " << filename << std::endl;
                std::cout << "Path: " << filepath << std::endl;
                
                // Read file content
                std::ifstream file(filepath);
                if (file.is_open()) {
                    std::stringstream buffer;
                    buffer << file.rdbuf();
                    std::string content = buffer.str();
                    file.close();
                    
                    std::cout << "Content:\n" << content << std::endl;
                    std::cout << "---" << std::endl;
                    
                    // Execute JavaScript code
                    try {
                        JSValue result = engine.evaluate(content);
                        std::cout << "✅ Result: " << engine.valueToString(result) << std::endl;
                        std::cout << "   Type: " << (result.isNumber() ? "number" : 
                                                    result.isString() ? "string" :
                                                    result.isBoolean() ? "boolean" :
                                                    result.isObject() ? "object" :
                                                    result.isNull() ? "null" :
                                                    result.isUndefined() ? "undefined" : "unknown") << std::endl;
                        std::cout << "   Truthy: " << (engine.isTruthy(result) ? "true" : "false") << std::endl;
                    } catch (const JSException& e) {
                        std::cout << "❌ Execution error: " << e.what() << std::endl;
                    } catch (const std::exception& e) {
                        std::cout << "❌ Standard error: " << e.what() << std::endl;
                    }
                } else {
                    std::cout << "❌ Failed to open file: " << filepath << std::endl;
                }
                
                std::cout << std::endl;
            }
            
            std::cout << "📊 Summary: Tested " << jsFiles.size() << " JavaScript files" << std::endl;
        } else {
            std::cout << "❌ Failed to open examples directory: " << examplesDir << std::endl;
        }
        
    } catch (const JSException& e) {
        std::cerr << "❌ JS Exception: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "❌ Standard Exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Unknown exception occurred" << std::endl;
        return 1;
    }
    
    return 0;
}