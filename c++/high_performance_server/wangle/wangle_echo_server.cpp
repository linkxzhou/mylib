#include <wangle/bootstrap/ServerBootstrap.h>
#include <wangle/channel/AsyncSocketHandler.h>
#include <wangle/codec/ByteToMessageDecoder.h>
#include <wangle/channel/EventBaseHandler.h>
#include <folly/io/IOBuf.h>
#include <folly/io/Cursor.h>
#include <sstream>
#include <string>

using namespace wangle;
using namespace folly;

class HttpRequest {
public:
    std::string method;
    std::string path;
    std::string version;
    std::map<std::string, std::string> headers;
    std::string body;
};

class HttpDecoder : public ByteToMessageDecoder<HttpRequest> {
public:
    bool decode(Context* ctx, IOBufQueue& buf, HttpRequest& result, size_t&) override {
        auto data = buf.move();
        if (!data) {
            return false;
        }
        
        std::string request = data->moveToFbString().toStdString();
        
        // Check if we have a complete HTTP request
        size_t header_end = request.find("\r\n\r\n");
        if (header_end == std::string::npos) {
            // Put data back and wait for more
            buf.append(IOBuf::copyBuffer(request));
            return false;
        }
        
        // Parse HTTP request
        std::istringstream iss(request);
        iss >> result.method >> result.path >> result.version;
        
        // Parse headers
        std::string line;
        std::getline(iss, line); // consume the rest of the first line
        
        while (std::getline(iss, line) && line != "\r" && !line.empty()) {
            size_t colon_pos = line.find(':');
            if (colon_pos != std::string::npos) {
                std::string key = line.substr(0, colon_pos);
                std::string value = line.substr(colon_pos + 1);
                // Trim whitespace
                value.erase(0, value.find_first_not_of(" \t"));
                value.erase(value.find_last_not_of(" \t\r") + 1);
                result.headers[key] = value;
            }
        }
        
        // Parse body if present
        auto content_length_it = result.headers.find("Content-Length");
        if (content_length_it != result.headers.end()) {
            int content_length = std::stoi(content_length_it->second);
            size_t body_start = header_end + 4;
            if (request.length() >= body_start + content_length) {
                result.body = request.substr(body_start, content_length);
            }
        }
        
        return true;
    }
};

class HttpHandler : public HandlerAdapter<HttpRequest> {
public:
    void read(Context* ctx, HttpRequest req) override {
        std::cout << "Received HTTP request: " << req.method << " " << req.path << std::endl;
        
        // Create response content
        std::ostringstream content;
        content << "Echo HTTP Request:\n";
        content << "Method: " << req.method << "\n";
        content << "Path: " << req.path << "\n";
        content << "Version: " << req.version << "\n";
        content << "Headers:\n";
        for (const auto& header : req.headers) {
            content << header.first << ": " << header.second << "\n";
        }
        if (!req.body.empty()) {
            content << "Body:\n" << req.body << "\n";
        }
        
        // Create HTTP response
        std::ostringstream response;
        response << "HTTP/1.1 200 OK\r\n";
        response << "Content-Type: text/plain\r\n";
        response << "Content-Length: " << content.str().length() << "\r\n";
        response << "Connection: keep-alive\r\n";
        response << "Server: Wangle-HTTP-Server/1.0\r\n";
        response << "\r\n";
        response << content.str();
        
        // Send response as IOBuf
        auto response_buf = IOBuf::copyBuffer(response.str());
        ctx->getTransport()->writeChain(nullptr, std::move(response_buf));
    }
    
    void readException(Context* ctx, exception_wrapper e) override {
        std::cout << "Client disconnected: " << e.what() << std::endl;
        
        // Send HTTP 400 Bad Request
        std::string bad_response = "HTTP/1.1 400 Bad Request\r\n"
                                  "Content-Type: text/plain\r\n"
                                  "Content-Length: 11\r\n"
                                  "Connection: close\r\n"
                                  "\r\n"
                                  "Bad Request";
        
        auto response_buf = IOBuf::copyBuffer(bad_response);
        ctx->getTransport()->writeChain(nullptr, std::move(response_buf));
        close(ctx);
    }
};

class HttpPipelineFactory : public PipelineFactory<DefaultPipeline> {
public:
    DefaultPipeline::Ptr newPipeline(std::shared_ptr<AsyncTransportWrapper> sock) override {
        auto pipeline = DefaultPipeline::create();
        pipeline->addBack(AsyncSocketHandler(sock));
        pipeline->addBack(EventBaseHandler());
        pipeline->addBack(std::make_shared<HttpDecoder>());
        pipeline->addBack(std::make_shared<HttpHandler>());
        pipeline->finalize();
        return pipeline;
    }
};

int main() {
    ServerBootstrap<DefaultPipeline> server;
    server.childPipeline(std::make_shared<HttpPipelineFactory>());
    server.bind(8080);
    
    std::cout << "wangle HTTP/1.1 server listening on port 8080" << std::endl;
    server.waitForStop();
    
    return 0;
}