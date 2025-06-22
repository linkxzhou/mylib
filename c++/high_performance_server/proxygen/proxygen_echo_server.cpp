#include <proxygen/httpserver/HTTPServer.h>
#include <proxygen/httpserver/RequestHandler.h>
#include <proxygen/httpserver/RequestHandlerFactory.h>
#include <proxygen/httpserver/ResponseBuilder.h>
#include <folly/Memory.h>
#include <folly/io/async/EventBaseManager.h>
#include <sstream>

using namespace proxygen;
using namespace folly;

class EchoHandler : public RequestHandler {
public:
    void onRequest(std::unique_ptr<HTTPMessage> headers) noexcept override {
        // Store the request method and path
        method_ = headers->getMethodString();
        path_ = headers->getPath();
        version_ = headers->getVersionString();
        
        // Store all headers
        headers->getHeaders().forEach([this](const std::string& name, const std::string& value) {
            headers_ += name + ": " + value + "\n";
        });
    }
    
    void onBody(std::unique_ptr<IOBuf> body) noexcept override {
        if (body_) {
            body_->prependChain(std::move(body));
        } else {
            body_ = std::move(body);
        }
    }
    
    void onEOM() noexcept override {
        std::ostringstream response;
        response << "Echo HTTP/1.1 Request:\n";
        response << "Method: " << method_ << "\n";
        response << "Path: " << path_ << "\n";
        response << "Version: " << version_ << "\n";
        response << "Headers:\n" << headers_;
        
        if (body_) {
            response << "Body: " << body_->moveToFbString().toStdString() << "\n";
        }
        
        ResponseBuilder(downstream_)
            .status(200, "OK")
            .header("Content-Type", "text/plain")
            .header("Server", "Proxygen-Echo-Server/1.0")
            .header("Connection", "keep-alive")
            .body(response.str())
            .sendWithEOM();
    }
    
    void onUpgrade(UpgradeProtocol protocol) noexcept override {}
    
    void requestComplete() noexcept override {
        delete this;
    }
    
    void onError(ProxygenError err) noexcept override {
        ResponseBuilder(downstream_)
            .status(400, "Bad Request")
            .header("Content-Type", "text/plain")
            .header("Connection", "close")
            .body("Bad Request")
            .sendWithEOM();
        delete this;
    }
    
private:
    std::string method_;
    std::string path_;
    std::string version_;
    std::string headers_;
    std::unique_ptr<IOBuf> body_;
};

class EchoHandlerFactory : public RequestHandlerFactory {
public:
    void onServerStart(folly::EventBase* evb) noexcept override {}
    
    void onServerStop() noexcept override {}
    
    RequestHandler* onRequest(RequestHandler* handler, HTTPMessage* msg) noexcept override {
        return new EchoHandler();
    }
};

int main() {
    HTTPServerOptions options;
    options.threads = static_cast<size_t>(sysconf(_SC_NPROCESSORS_ONLN));
    options.idleTimeout = std::chrono::milliseconds(60000);
    options.shutdownOn = {SIGINT, SIGTERM};
    options.enableContentCompression = false;
    options.handlerFactories = RequestHandlerChain()
        .addThen<EchoHandlerFactory>()
        .build();
    
    std::vector<HTTPServer::IPConfig> IPs = {
        {SocketAddress("0.0.0.0", 8080, true), HTTPServer::Protocol::HTTP}
    };
    
    HTTPServer server(std::move(options));
    server.bind(IPs);
    
    std::thread t([&] () {
        server.start();
    });
    
    std::cout << "proxygen HTTP/1.1 echo server listening on port 8080" << std::endl;
    t.join();
    
    return 0;
}