#include "co_routine.h"
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

struct task_t {
    stCoRoutine_t *co;
    int fd;
};

void *client_routine(void *arg) {
    co_enable_hook_sys();
    
    struct task_t *task = (struct task_t*)arg;
    int fd = task->fd;
    char buffer[1024];
    
    while (true) {
        int ret = read(fd, buffer, sizeof(buffer));
        if (ret <= 0) {
            if (ret < 0 && errno == EAGAIN) {
                continue;
            }
            break;
        }
        
        // Echo back
        write(fd, buffer, ret);
    }
    
    close(fd);
    return NULL;
}

void *accept_routine(void *arg) {
    co_enable_hook_sys();
    
    int listen_fd = *(int*)arg;
    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);
    
    while (true) {
        int client_fd = accept(listen_fd, (struct sockaddr*)&client_addr, &client_len);
        if (client_fd < 0) {
            if (errno == EAGAIN) {
                continue;
            }
            perror("accept");
            break;
        }
        
        printf("New client connected\n");
        
        struct task_t *task = (struct task_t*)malloc(sizeof(struct task_t));
        task->fd = client_fd;
        
        co_create(&task->co, NULL, client_routine, task);
        co_resume(task->co);
    }
    
    return NULL;
}

int main() {
    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        perror("socket");
        return 1;
    }
    
    int reuse = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(8080);
    
    if (bind(listen_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("bind");
        return 1;
    }
    
    if (listen(listen_fd, 128) < 0) {
        perror("listen");
        return 1;
    }
    
    fcntl(listen_fd, F_SETFL, O_NONBLOCK);
    
    stCoRoutine_t *accept_co;
    co_create(&accept_co, NULL, accept_routine, &listen_fd);
    co_resume(accept_co);
    
    printf("libco echo server listening on port 8080\n");
    co_eventloop(co_get_epoll_ct(), NULL, NULL);
    
    return 0;
}