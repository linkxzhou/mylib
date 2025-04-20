#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <dirent.h>
#include <sys/stat.h>
#include <errno.h>
#include "simplejs.h"

#define EXAMPLES_DIR "examples"
#define MAX_PATH_LEN 256
#define MAX_FILE_SIZE 4096

// Helper function to create a JS instance
struct js* create_js() {
    const size_t size = 1024 * 64; // 64 KB
    void* mem = malloc(size);
    return js_create(mem, size);
}

// Helper function to free a JS instance
void free_js(struct js* js) {
    free(js);
}

// Helper function to create example directory if it doesn't exist
void ensure_examples_dir() {
    struct stat st = {0};
    if (stat(EXAMPLES_DIR, &st) == -1) {
        #ifdef _WIN32
        int result = mkdir(EXAMPLES_DIR);
        #else
        int result = mkdir(EXAMPLES_DIR, 0755);
        #endif
        if (result != 0) {
            printf("Failed to create examples directory: %s\n", strerror(errno));
            exit(1);
        }
        printf("Created examples directory\n");
    }
}

// Helper function to read a file into memory
char* read_file(const char* path, size_t* size) {
    FILE* file = fopen(path, "r");
    if (!file) {
        printf("Failed to open file %s: %s\n", path, strerror(errno));
        return NULL;
    }
    
    // Get file size
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    rewind(file);
    
    if (file_size <= 0 || file_size > MAX_FILE_SIZE) {
        printf("Invalid file size for %s: %ld\n", path, file_size);
        fclose(file);
        return NULL;
    }
    
    // Allocate memory and read file
    char* buffer = (char*)malloc(file_size + 1);
    if (!buffer) {
        printf("Failed to allocate memory for file %s\n", path);
        fclose(file);
        return NULL;
    }
    
    size_t read_size = fread(buffer, 1, file_size, file);
    fclose(file);
    
    if (read_size != (size_t)file_size) {
        printf("Failed to read entire file %s\n", path);
        free(buffer);
        return NULL;
    }
    
    buffer[file_size] = '\0';
    if (size) *size = file_size;
    return buffer;
}

// Helper function to get JS value type as string
const char* js_type_str(int type) {
    switch (type) {
        case JS_UNDEF: return "undefined";
        case JS_NULL: return "null";
        case JS_TRUE: return "true";
        case JS_FALSE: return "false";
        case JS_STR: return "string";
        case JS_NUM: return "number";
        case JS_ERR: return "error";
        case JS_PRIV: return "private";
        default: return "unknown";
    }
}

// Main function to run tests from example files
int main() {    
    printf("\nRunning SimpleJS tests from example files...\n\n");
    
    DIR* dir = opendir(EXAMPLES_DIR);
    if (!dir) {
        printf("Failed to open examples directory: %s\n", strerror(errno));
        return 1;
    }
    
    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        // Skip directories and non-.js files
        if (entry->d_type == DT_DIR) continue;
        
        const char* filename = entry->d_name;
        const char* ext = strrchr(filename, '.');
        if (!ext || strcmp(ext, ".js") != 0) continue;
        
        // Construct file path
        char path[MAX_PATH_LEN];
        snprintf(path, sizeof(path), "%s/%s", EXAMPLES_DIR, filename);
        
        // Read file content
        size_t size = 0;
        char* content = read_file(path, &size);
        if (!content) continue;
        
        printf("Testing: %s\n", filename);
        printf("Content:\n%s\n", content);
        
        // Create JS instance and evaluate code
        struct js* js = create_js();
        
        // Set GC threshold for memory test
        if (strcmp(filename, "memory.js") == 0) {
            js_setgct(js, 1024 * 32);
        }
        
        jsval_t result = js_eval(js, content, size);
        
        // Print result
        printf("Result type: %s\n", js_type_str(js_type(result)));
        printf("Result: %s\n", js_str(js, result));
        
        // Get memory stats for memory test
        if (strcmp(filename, "memory.js") == 0) {
            size_t total, lwm, css;
            js_stats(js, &total, &lwm, &css);
            printf("Memory stats - Total: %zu, LWM: %zu, CSS: %zu\n", total, lwm, css);
        }
        
        printf("\n");
        
        // Clean up
        free_js(js);
        free(content);
    }
    
    closedir(dir);
    printf("All tests completed!\n");
    return 0;
}