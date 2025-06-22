#include <uv.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uv_write_t req;
    uv_buf_t buf;
} write_req_t;

void free_write_req(uv_write_t *req) {
    write_req_t *wr = (write_req_t*) req;
    free(wr->buf.base);
    free(wr);
}

void alloc_buffer(uv_handle_t *handle, size_t suggested_size, uv_buf_t *buf) {
    buf->base = (char*) malloc(suggested_size);
    buf->len = suggested_size;
}

void on_close(uv_handle_t* handle) {
    free(handle);
}

void send_http_response(uv_stream_t *client, const char* body) {
    char *response = (char*)malloc(2048);
    int content_length = strlen(body);
    
    snprintf(response, 2048,
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/plain\r\n"
        "Content-Length: %d\r\n"
        "Connection: keep-alive\r\n"
        "Server: libuv-http/1.0\r\n"
        "\r\n"
        "%s",
        content_length, body);
    
    write_req_t *req = (write_req_t*) malloc(sizeof(write_req_t));
    req->buf = uv_buf_init(response, strlen(response));
    uv_write((uv_write_t*) req, client, &req->buf, 1, [](uv_write_t *req, int status) {
        if (status) {
            fprintf(stderr, "Write error %s\n", uv_strerror(status));
        }
        free_write_req(req);
    });
}

void handle_http_request(uv_stream_t *client, const char* request) {
    char method[16], path[256], version[16];
    
    if (sscanf(request, "%15s %255s %15s", method, path, version) == 3) {
        char *response_body = (char*)malloc(1024);
        snprintf(response_body, 1024,
            "libuv HTTP Server Response\n"
            "Method: %s\n"
            "Path: %s\n"
            "Version: %s\n"
            "\n"
            "Request received successfully!",
            method, path, version);
        
        send_http_response(client, response_body);
        free(response_body);
    } else {
        const char* bad_request = 
            "HTTP/1.1 400 Bad Request\r\n"
            "Content-Type: text/plain\r\n"
            "Content-Length: 11\r\n"
            "Connection: close\r\n"
            "\r\n"
            "Bad Request";
        
        write_req_t *req = (write_req_t*) malloc(sizeof(write_req_t));
        req->buf = uv_buf_init((char*)bad_request, strlen(bad_request));
        uv_write((uv_write_t*) req, client, &req->buf, 1, [](uv_write_t *req, int status) {
            if (status) {
                fprintf(stderr, "Write error %s\n", uv_strerror(status));
            }
            write_req_t *wr = (write_req_t*) req;
            free(wr);
        });
    }
}

void echo_read(uv_stream_t *client, ssize_t nread, const uv_buf_t *buf) {
    if (nread > 0) {
        buf->base[nread] = '\0';
        
        if (strstr(buf->base, "\r\n\r\n") != NULL) {
            printf("Received HTTP request:\n%s\n", buf->base);
            handle_http_request(client, buf->base);
        }
    }
    
    if (nread < 0) {
        if (nread != UV_EOF) {
            fprintf(stderr, "Read error %s\n", uv_err_name(nread));
        }
        uv_close((uv_handle_t*) client, on_close);
    }
    
    free(buf->base);
}

void on_new_connection(uv_stream_t *server, int status) {
    if (status < 0) {
        fprintf(stderr, "New connection error %s\n", uv_strerror(status));
        return;
    }
    
    uv_tcp_t *client = (uv_tcp_t*) malloc(sizeof(uv_tcp_t));
    uv_tcp_init(uv_default_loop(), client);
    
    if (uv_accept(server, (uv_stream_t*) client) == 0) {
        uv_read_start((uv_stream_t*) client, alloc_buffer, echo_read);
    } else {
        uv_close((uv_handle_t*) client, on_close);
    }
}

int main() {
    uv_loop_t *loop = uv_default_loop();
    
    uv_tcp_t server;
    uv_tcp_init(loop, &server);
    
    struct sockaddr_in addr;
    uv_ip4_addr("0.0.0.0", 8080, &addr);
    
    uv_tcp_bind(&server, (const struct sockaddr*)&addr, 0);
    int r = uv_listen((uv_stream_t*) &server, 128, on_new_connection);
    if (r) {
        fprintf(stderr, "Listen error %s\n", uv_strerror(r));
        return 1;
    }
    
    printf("libuv HTTP server listening on port 8080\n");
    return uv_run(loop, UV_RUN_DEFAULT);
}