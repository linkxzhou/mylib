#include <mongoose.h>
#include <cstdio>
#include <cstring>

static void ev_handler(struct mg_connection *c, int ev, void *ev_data, void *fn_data) {
    if (ev == MG_EV_ACCEPT) {
        printf("New client connected\n");
    } else if (ev == MG_EV_HTTP_MSG) {
        struct mg_http_message *hm = (struct mg_http_message *) ev_data;
        
        char response_body[1024];
        snprintf(response_body, sizeof(response_body),
            "Mongoose HTTP Server Response\n"
            "Method: %.*s\n"
            "URI: %.*s\n"
            "\n"
            "Request received successfully!",
            (int)hm->method.len, hm->method.ptr,
            (int)hm->uri.len, hm->uri.ptr);
        
        mg_http_reply(c, 200, "Content-Type: text/plain\r\nServer: mongoose-http/1.0\r\n", "%s", response_body);
    } else if (ev == MG_EV_CLOSE) {
        printf("Client disconnected\n");
    }
}

int main() {
    struct mg_mgr mgr;
    struct mg_connection *c;
    
    mg_mgr_init(&mgr);
    c = mg_http_listen(&mgr, "http://0.0.0.0:8080", ev_handler, NULL);
    
    if (c == NULL) {
        printf("Failed to create HTTP listener\n");
        return 1;
    }
    
    printf("Mongoose HTTP server listening on port 8080\n");
    
    for (;;) {
        mg_mgr_poll(&mgr, 1000);
    }
    
    mg_mgr_free(&mgr);
    return 0;
}