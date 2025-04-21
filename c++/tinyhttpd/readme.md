## tinyhttpd++
### 介绍
tinyhttpd++ 是重写 tinyhttpd 的 C++ 版本，实现最基本的 HTTP 服务器功能：
- 多线程
- 文件读写
- 简单的 CGI 协议
- 4xx, 5xx 状态码

### 目录结构
```
├── httpd.h         # HTTP 服务器核心头文件
├── httpd.cc        # HTTP 服务器实现
├── Makefile        # 构建脚本
├── htdocs/         # 网站根目录
└── readme.md       # 项目说明
```

### 构建与运行

#### 构建 (Release 默认)
```sh
make
./tinyhttpd
```

#### 构建 (Debug 带调试信息)
```sh
make debug
./tinyhttpd
```

#### 清理构建产物
```sh
make clean
```

### 主要用法
- 默认监听 8080 或 8081 端口（可在 main 函数中修改）
- 访问 http://localhost:8080/ 或 http://localhost:8081/ 查看效果
- 网页文件请放在 `htdocs/` 目录下
- 支持简单 CGI 脚本（如 .cgi 文件）
- 启动服务：`./tinyhttpd port` // 指定端口启动，否则是 8081

### Makefile 说明
- 支持 debug/release 构建
- 自动检测头文件依赖
- 支持多源文件扩展

### 常见问题
- **端口被占用**：请检查端口是否已被其他进程占用，或修改 main 函数端口号。
- **找不到 htdocs 目录**：请确保 `htdocs/` 目录存在且有可访问的文件。

### 工作流程
（1）服务器启动，在指定端口或随机选取端口绑定httpd服务。
```
    Httpd httpd;
    LOG("startup port:%d", 8080);
    httpd.startup(8080);
```
（2）收到一个HTTP请求时（其实就是listen的端口accpet的时候），派生一个线程运行accept_request函数。  
```
    if (pthread_create(&pthread , NULL, accept_request, s) != 0)
    {
        ERROR("pthread_create");
    }
```
（3）取出HTTP请求中的method(GET或POST)和url，包括url携带的参数。  
```
    HttpdSocketPtr socket = (HttpdSocketPtr)arg;
    socket->parseMethod();
    if (!socket->isPOST() && !socket->isGET())
    {
        socket->error501();
        return NULL;
    }
    // 解析URL
    socket->parseUrl();
    // 解析header
    socket->parseHeader();
```
（4）格式化url到path数组，表示浏览器请求的服务器文件路径，在tinyhttpd中服务器文件是在htdocs文件夹下，当url以/结尾，或url是个目录，则默认在path中加上index.html，表示访问主页。  
（5）如果文件路径合法，对于无参数的GET请求，直接输出服务器文件到浏览器，其他情况（带参数GET，POST方式，url为可执行文件），则调用 excute_cgi 函数执行cgi脚本。  
```
    if ((st.st_mode & S_IFMT) == S_IFDIR)
    {
        strcat(path, "/index.html");
    }
    LOG("cgi:%d, path:%s", socket->cgi(), path);
    // 不采用cgi
    if (!(socket->cgi()))
    {
        socket->serveFile(path);
    }
    else
    {
        socket->executeCGI(path);
    }
```
（6）读取整个HTTP请求并丢弃，如果是POST则找出Content-Length。把HTTP 200状态码写到套接字。  
（7）建立两个管道，cgi_input和cgi_output, 并fork一个进程。  
（8）在子进程中，把STDOUT重定向到cgi_output的写入端，把STDIN重定向到cgi_input的读取端，关闭cgi_input的写入端和cgi_output的读取端，设置request_method的环境变量，GET设置查询的环境变量，POST设置content_length的环境变量，这些环境变量都是为了给cgi脚本调用，接着用 execl运行cgi程序。  
```
    snprintf(meth_env, sizeof(meth_env) - 1, "REQUEST_METHOD=%s", m_method_.c_str());
    putenv(meth_env);
    if (isGET()) 
    {
        snprintf(query_env, sizeof(query_env) - 1, "QUERY_STRING=%s", m_query_.c_str());
        putenv(query_env);
    }
    else 
    {
        snprintf(length_env, sizeof(length_env) - 1, "CONTENT_LENGTH=%d", content_len);
        putenv(length_env);
    }
    execl(path, path, NULL);
```
（9）在父进程中，关闭cgi_input的读取端和cgi_output的写入端，如果POST的话，把POST数据写入cgi_input，已被重定向到STDIN，读取 cgi_output的管道输出到客户端，该管道输入是STDOUT，接着关闭所有管道，等待子进程结束。 

### 最后
（1）为了后续改造方便，可以尝试重写HttpdSocket的虚函数：  
```
    virtual void error501();

    virtual void error404();

    virtual void error400();

    virtual void error500();

    virtual void serveFile(const char *path);

    virtual void executeCGI(const char *path);
```
（2）以下内容来自tinyhttpd源作者:   
This software is copyright 1999 by J. David Blackstone. Permission is granted to redistribute and modify this software under the terms of the GNU General Public License, available at http://www.gnu.org/ .

If you use this software or examine the code, I would appreciate knowing and would be overjoyed to hear about it at jdavidb@sourceforge.net .

This software is not production quality. It comes with no warranty of any kind, not even an implied warranty of fitness for a particular purpose. I am not responsible for the damage that will likely result if you use this software on your computer system.

I wrote this webserver for an assignment in my networking class in 1999. We were told that at a bare minimum the server had to serve pages, and told that we would get extra credit for doing "extras." Perl had introduced me to a whole lot of UNIX functionality (I learned sockets and fork from Perl!), and O'Reilly's lion book on UNIX system calls plus O'Reilly's books on CGI and writing web clients in Perl got me thinking and I realized I could make my webserver support CGI with little trouble.

Now, if you're a member of the Apache core group, you might not be impressed. But my professor was blown over. Try the color.cgi sample script and type in "chartreuse." Made me seem smarter than I am, at any rate. :)

Apache it's not. But I do hope that this program is a good educational tool for those interested in http/socket programming, as well as UNIX system calls. (There's some textbook uses of pipes, environment variables, forks, and so on.)

One last thing: if you look at my webserver or (are you out of mind?!?) use it, I would just be overjoyed to hear about it. Please email me. I probably won't really be releasing major updates, but if I help you learn something, I'd love to know!

Happy hacking!

J. David Blackstone