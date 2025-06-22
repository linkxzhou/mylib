#include <event2/event.h>
#include <event2/buffer.h>
#include <event2/bufferevent.h>
#include <event2/listener.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>

static void send_http_response(struct bufferevent *bev, const char* body) {
    char response[2048];
    int content_length = strlen(body);
    
    snprintf(response, sizeof(response),
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/plain\r\n"
        "Content-Length: %d\r\n"
        "Connection: keep-alive\r\n"
        "Server: libevent-http/1.0\r\n"
        "\r\n"
        "%s",
        content_length, body);
    
    bufferevent_write(bev, response, strlen(response));
}

static void handle_http_request(struct bufferevent *bev, const char* request) {
    char method[16], path[256], version[16];
    
    if (sscanf(request, "%15s %255s %15s", method, path, version) == 3) {
        char response_body[1024];
        snprintf(response_body, sizeof(response_body),
            "libevent HTTP Server Response\n"
            "Method: %s\n"
            "Path: %s\n"
            "Version: %s\n"
            "\n"
            "Request received successfully!",
            method, path, version);
        
        send_http_response(bev, response_body);
    } else {
        const char* bad_request = 
            "HTTP/1.1 400 Bad Request\r\n"
            "Content-Type: text/plain\r\n"
            "Content-Length: 11\r\n"
            "Connection: close\r\n"
            "\r\n"
            "Bad Request";
        bufferevent_write(bev, bad_request, strlen(bad_request));
    }
}

static void echo_read_cb(struct bufferevent *bev, void *ctx) {
    struct evbuffer *input = bufferevent_get_input(bev);
    size_t len = evbuffer_get_length(input);
    
    if (len > 0) {
        char *data = (char*)malloc(len + 1);
        evbuffer_remove(input, data, len);
        data[len] = '\0';
        
        if (strstr(data, "\r\n\r\n") != NULL) {
            printf("Received HTTP request:\n%s\n", data);
            handle_http_request(bev, data);
        }
        
        free(data);
    }
}

static void echo_event_cb(struct bufferevent *bev, short events, void *ctx) {
    if (events & BEV_EVENT_ERROR) {
        perror("Error from bufferevent");
    }
    if (events & (BEV_EVENT_EOF | BEV_EVENT_ERROR)) {
        bufferevent_free(bev);
    }
}

static void accept_conn_cb(struct evconnlistener *listener,
                          evutil_socket_t fd, struct sockaddr *address,
                          int socklen, void *ctx) {
    struct event_base *base = evconnlistener_get_base(listener);
    struct bufferevent *bev = bufferevent_socket_new(base, fd, BEV_OPT_CLOSE_ON_FREE);
    
    bufferevent_setcb(bev, echo_read_cb, NULL, echo_event_cb, NULL);
    bufferevent_enable(bev, EV_READ | EV_WRITE);
}

static void accept_error_cb(struct evconnlistener *listener, void *ctx) {
    struct event_base *base = evconnlistener_get_base(listener);
    int err = EVUTIL_SOCKET_ERROR();
    fprintf(stderr, "Got an error %d (%s) on the listener. Shutting down.\n",
            err, evutil_socket_error_to_string(err));
    event_base_loopexit(base, NULL);
}

int main() {
    struct event_base *base;
    struct evconnlistener *listener;
    struct sockaddr_in sin;
    
    base = event_base_new();
    if (!base) {
        puts("Couldn't open event base");
        return 1;
    }
    
    memset(&sin, 0, sizeof(sin));
    sin.sin_family = AF_INET;
    sin.sin_addr.s_addr = htonl(INADDR_ANY);
    sin.sin_port = htons(8080);
    
    listener = evconnlistener_new_bind(base, accept_conn_cb, NULL,
                                      LEV_OPT_CLOSE_ON_FREE | LEV_OPT_REUSEABLE,
                                      -1, (struct sockaddr*)&sin, sizeof(sin));
    if (!listener) {
        perror("Couldn't create listener");
        return 1;
    }
    evconnlistener_set_error_cb(listener, accept_error_cb);
    
    printf("libevent HTTP server listening on port 8080\n");
    event_base_dispatch(base);
    
    return 0;
}