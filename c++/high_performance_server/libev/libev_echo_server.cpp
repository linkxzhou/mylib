#include <ev.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

struct client_watcher {
    ev_io io;
    int fd;
};

// HTTP 1.1 响应函数
static void send_http_response(int fd, const char* body) {
    char response[2048];
    int content_length = strlen(body);
    
    snprintf(response, sizeof(response),
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/plain\r\n"
        "Content-Length: %d\r\n"
        "Connection: keep-alive\r\n"
        "Server: libev-http/1.0\r\n"
        "\r\n"
        "%s",
        content_length, body);
    
    send(fd, response, strlen(response), 0);
}

// 解析HTTP请求并生成响应
static void handle_http_request(int fd, const char* request) {
    char method[16], path[256], version[16];
    
    // 简单解析HTTP请求行
    if (sscanf(request, "%15s %255s %15s", method, path, version) == 3) {
        char response_body[1024];
        snprintf(response_body, sizeof(response_body),
            "Echo Server Response\n"
            "Method: %s\n"
            "Path: %s\n"
            "Version: %s\n"
            "\n"
            "Request received successfully!",
            method, path, version);
        
        send_http_response(fd, response_body);
    } else {
        // 发送400 Bad Request
        const char* bad_request = 
            "HTTP/1.1 400 Bad Request\r\n"
            "Content-Type: text/plain\r\n"
            "Content-Length: 11\r\n"
            "Connection: close\r\n"
            "\r\n"
            "Bad Request";
        send(fd, bad_request, strlen(bad_request), 0);
    }
}

static void client_cb(EV_P_ ev_io *w, int revents) {
    struct client_watcher *client = (struct client_watcher*)w;
    char buffer[4096];
    
    if (revents & EV_READ) {
        ssize_t nread = recv(client->fd, buffer, sizeof(buffer) - 1, 0);
        if (nread <= 0) {
            if (nread == 0) {
                printf("Client disconnected\n");
            } else {
                perror("recv");
            }
            ev_io_stop(EV_A_ w);
            close(client->fd);
            free(client);
            return;
        }
        
        buffer[nread] = '\0';
        
        // 检查是否接收到完整的HTTP请求（以\r\n\r\n结尾）
        if (strstr(buffer, "\r\n\r\n") != NULL) {
            printf("Received HTTP request:\n%s\n", buffer);
            handle_http_request(client->fd, buffer);
        }
    }
}

static void accept_cb(EV_P_ ev_io *w, int revents) {
    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);
    int client_fd;
    
    client_fd = accept(w->fd, (struct sockaddr*)&client_addr, &client_len);
    if (client_fd < 0) {
        perror("accept");
        return;
    }
    
    // Set non-blocking
    fcntl(client_fd, F_SETFL, O_NONBLOCK);
    
    struct client_watcher *client = (struct client_watcher*)malloc(sizeof(struct client_watcher));
    client->fd = client_fd;
    
    ev_io_init(&client->io, client_cb, client_fd, EV_READ);
    ev_io_start(EV_A_ &client->io);
    
    printf("New client connected\n");
}

int main() {
    struct ev_loop *loop = EV_DEFAULT;
    int listen_fd;
    struct sockaddr_in listen_addr;
    ev_io accept_watcher;
    
    // Create socket
    listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        perror("socket");
        return 1;
    }
    
    int reuse = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    
    memset(&listen_addr, 0, sizeof(listen_addr));
    listen_addr.sin_family = AF_INET;
    listen_addr.sin_addr.s_addr = INADDR_ANY;
    listen_addr.sin_port = htons(8080);
    
    if (bind(listen_fd, (struct sockaddr*)&listen_addr, sizeof(listen_addr)) < 0) {
        perror("bind");
        return 1;
    }
    
    if (listen(listen_fd, 128) < 0) {
        perror("listen");
        return 1;
    }
    
    fcntl(listen_fd, F_SETFL, O_NONBLOCK);
    
    ev_io_init(&accept_watcher, accept_cb, listen_fd, EV_READ);
    ev_io_start(loop, &accept_watcher);
    
    printf("libev HTTP server listening on port 8080\n");
    ev_loop(loop, 0);
    
    return 0;
}