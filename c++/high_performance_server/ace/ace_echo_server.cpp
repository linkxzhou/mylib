#include "ace/Reactor.h"
#include "ace/SOCK_Acceptor.h"
#include "ace/SOCK_Stream.h"
#include "ace/INET_Addr.h"
#include "ace/Event_Handler.h"
#include "ace/Log_Msg.h"
#include <cstring>
#include <cstdio>

class Echo_Handler : public ACE_Event_Handler {
public:
    Echo_Handler(ACE_SOCK_Stream *stream) : stream_(stream) {}
    
    virtual ~Echo_Handler() {
        delete stream_;
    }
    
    virtual ACE_HANDLE get_handle() const {
        return stream_->get_handle();
    }
    
    void send_http_response(const char* body) {
        char response[2048];
        int content_length = strlen(body);
        
        snprintf(response, sizeof(response),
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/plain\r\n"
            "Content-Length: %d\r\n"
            "Connection: keep-alive\r\n"
            "Server: ace-http/1.0\r\n"
            "\r\n"
            "%s",
            content_length, body);
        
        stream_->send_n(response, strlen(response));
    }
    
    void handle_http_request(const char* request) {
        char method[16], path[256], version[16];
        
        if (sscanf(request, "%15s %255s %15s", method, path, version) == 3) {
            char response_body[1024];
            snprintf(response_body, sizeof(response_body),
                "ACE HTTP Server Response\n"
                "Method: %s\n"
                "Path: %s\n"
                "Version: %s\n"
                "\n"
                "Request received successfully!",
                method, path, version);
            
            send_http_response(response_body);
        } else {
            const char* bad_request = 
                "HTTP/1.1 400 Bad Request\r\n"
                "Content-Type: text/plain\r\n"
                "Content-Length: 11\r\n"
                "Connection: close\r\n"
                "\r\n"
                "Bad Request";
            stream_->send_n(bad_request, strlen(bad_request));
        }
    }
    
    virtual int handle_input(ACE_HANDLE) {
        char buffer[4096];
        ssize_t bytes_received = stream_->recv(buffer, sizeof(buffer) - 1);
        
        if (bytes_received <= 0) {
            ACE_DEBUG((LM_DEBUG, "Client disconnected\n"));
            return -1;
        }
        
        buffer[bytes_received] = '\0';
        
        if (strstr(buffer, "\r\n\r\n") != NULL) {
            ACE_DEBUG((LM_DEBUG, "Received HTTP request:\n%s\n", buffer));
            handle_http_request(buffer);
        }
        
        return 0;
    }
    
    virtual int handle_close(ACE_HANDLE, ACE_Reactor_Mask) {
        delete this;
        return 0;
    }
    
private:
    ACE_SOCK_Stream *stream_;
};

class Accept_Handler : public ACE_Event_Handler {
public:
    Accept_Handler(const ACE_INET_Addr &addr) {
        if (acceptor_.open(addr, 1) == -1) {
            ACE_ERROR((LM_ERROR, "Failed to open acceptor\n"));
        }
    }
    
    virtual ACE_HANDLE get_handle() const {
        return acceptor_.get_handle();
    }
    
    virtual int handle_input(ACE_HANDLE) {
        ACE_SOCK_Stream *stream = new ACE_SOCK_Stream;
        ACE_INET_Addr client_addr;
        
        if (acceptor_.accept(*stream, &client_addr) == -1) {
            ACE_ERROR((LM_ERROR, "Failed to accept connection\n"));
            delete stream;
            return 0;
        }
        
        Echo_Handler *handler = new Echo_Handler(stream);
        if (ACE_Reactor::instance()->register_handler(handler, ACE_Event_Handler::READ_MASK) == -1) {
            ACE_ERROR((LM_ERROR, "Failed to register handler\n"));
            delete handler;
        }
        
        ACE_DEBUG((LM_DEBUG, "New client connected\n"));
        return 0;
    }
    
private:
    ACE_SOCK_Acceptor acceptor_;
};

int main() {
    ACE_INET_Addr server_addr(8080);
    Accept_Handler accept_handler(server_addr);
    
    if (ACE_Reactor::instance()->register_handler(&accept_handler, ACE_Event_Handler::ACCEPT_MASK) == -1) {
        ACE_ERROR_RETURN((LM_ERROR, "Failed to register accept handler\n"), 1);
    }
    
    ACE_DEBUG((LM_DEBUG, "ACE HTTP server listening on port 8080\n"));
    ACE_Reactor::instance()->run_reactor_event_loop();
    
    return 0;
}