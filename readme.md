# glog-mini介绍

**（1）glog-mini是为了方便使用glog库的mini版本，改造于glog-version0.3.4**  

**（2）使用方法执行make，将在目录下生产glog.a的静态库**   

**（3）引入头文件**    

```c++
#include "base/commandlineflags.h"
#include "base/config.h"
#include "glog/logging.h"
```

**编译时候，使用同时引入glog.a库即可**     

**（4）使用样例参考：**    

`unittest` 下面的 `demangle_unittest.cpp`   
编译：`cd unittest && g++ -o unittest demangle_unittest.cpp -I./ -I../ ../glog/glog.a -g -lpthread -ldl -lrt`  
执行：`./unittest`  

**（5）修改bazel编译方式：**    

编译命令：   
```
bazel build //glog:glog
```

测试命令：  
```
bazel test //unittest:demangle_unittest
```

# glog的使用方法

## 日志格式   

日志文件名称格式：<program name>.<hostname>.<user name>.log.<severity level>.<date>.<time>.<pid>   
例如：`hello_world.example.com.hamaji.log.INFO.20161120-222411.10474` 

# 日志级别  

```  
FLAGS_log_dir       日志输出目录  
FLAGS_v             自定义VLOG(m)时，m值小于此处设置值的语句才有输出  
FLAGS_max_log_size  每个日志文件最大大小（MB级别）  
FLAGS_minloglevel   输出日志的最小级别，即高于等于该级别的日志都将输出  
```

# 初始化  

```
1、google::InitGoogleLogging(argv[0]);  // 初始化  
2、google::SetLogDestination(google::GLOG_INFO, "./test");  
- 第一个参数为日志级别  
- 第二个参数表示输出目录及日志文件名前缀  
3、google::SetStderrLogging(google::INFO); // 输出到标准输出的时候大于 INFO 级别的都输出；等同于 FLAGS_stderrthreshold=google::INFO; 

// 可以设置的FLAGS
FLAGS_logbufsecs=0;　　	// 实时输出日志  
FLAGS_max_log_size=100; // 最大日志大小（MB）  
FLAGS_log_dir="./test"; // 设置日志生成目录  
```

## LOG_XX   

满足一定条件下输出日志，例如：  
```c++
LOG_IF(INFO, num_cookies > 10) << "Got lots of cookies"; 
``` 

## 条件输出   

```c++
LOG_IF(INFO, num_cookies > 10) << "Got lots of cookies"; //当条件满足时输出日志  
LOG_EVERY_N(INFO, 10) << "Got the " << google::COUNTER << "th cookie";　//google::COUNTER   记录该语句被执行次数，从1开始，在第一次运行输出日志之后，每隔 10 次再输出一次日志信息  
LOG_IF_EVERY_N(INFO, (size > 1024), 10) << "Got the " << google::COUNTER << "th big cookie";　　//上述两者的结合，不过要注意，是先每隔10次去判断条件是否满足，如果滞则输出日志；而不是当满足某条件的情况下，每隔10次输出一次日志信息  
LOG_FIRST_N(INFO, 20) << "Got the " << google::COUNTER << "th cookie"; //当此语句执行的前 20 次都输出日志，然后不再输出  
```

## 日志类型  

```c++
LOG    		//内置日志  
VLOG    	//自定义日志  
DLOG    	//DEBUG模式可输出的日志  
DVLOG   	//DEBUG模式可输出的自定义日志  
SYSLOG  	//系统日志，同时通过 syslog() 函数写入到 /var/log/message 文件  
PLOG    	//perror风格日志，设置errno状态并输出到日志中  
RAW_LOG     //线程安全的日志，需要#include <glog/raw_logging.h>  
```

## CHECK宏  

- 当通过该宏指定的条件不成立的时候，程序会中止，并且记录对应的日志信息;  
- 功能类似于ASSERT，区别是CHECK宏不受NDEBUG约束，在release版中同样有效;  
样例：  
```c++
CHECK(true) << "TRUE is output.";
CHECK(false) << "TRUE is output.";
```

# 使用样例

```c++
google::InitGoogleLogging(program);

//设置级别高于 google::INFO 的日志同时输出到屏幕 
google::SetStderrLogging(google::INFO); 

//设置输出到屏幕的日志显示相应颜色         
FLAGS_colorlogtostderr=true;  

//设置 google::ERROR 级别的日志存储路径和文件名前缀                              
google::SetLogDestination(google::ERROR,"log/error_"); 

//设置 google::INFO 级别的日志存储路径和文件名前缀 
google::SetLogDestination(google::INFO,LOGDIR"/INFO_");    

//设置 google::WARNING 级别的日志存储路径和文件名前缀
google::SetLogDestination(google::WARNING,LOGDIR"/WARNING_");

//设置 google::ERROR 级别的日志存储路径和文件名前缀
google::SetLogDestination(google::ERROR,LOGDIR"/ERROR_");   

//缓冲日志输出，默认为30秒，此处改为立即输出
FLAGS_logbufsecs = 0;         

//最大日志大小为 100MB                              
FLAGS_max_log_size = 100;     

//当磁盘被写满时，停止日志输出                         
FLAGS_stop_logging_if_full_disk = true; 

//设置文件名扩展，如平台？或其它需要区分的信息                    
google::SetLogFilenameExtension("91_");

//捕捉 core dumped                     
google::InstallFailureSignalHandler();    
  
//默认捕捉 SIGSEGV 信号信息输出会输出到                 
google::InstallFailureWriter(&SignalHandle);                
```