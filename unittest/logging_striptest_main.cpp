// Copyright (c) 2007, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Author: Sergey Ioffe

// The common part of the striplog tests.

#include <stdio.h>
#include <string>
#include <iosfwd>

#include "commandlineflags.h"
#include "config.h"
#include "logging.h"

DECLARE_bool(logtostderr);
GLOG_DEFINE_bool(check_mode, false, "Prints 'opt' or 'dbg'");

using std::string;
using namespace GOOGLE_NAMESPACE;

int CheckNoReturn(bool b) {
  string s;
  if (b) {
    LOG(FATAL) << "Fatal";
  } else {
    return 0;
  }
}

struct A { };
std::ostream &operator<<(std::ostream &str, const A&) {return str;}

int main(int, char* argv[]) {
  //FLAGS_logtostderr = true;
  InitGoogleLogging(argv[0]);
  FLAGS_log_dir = "."; // 设置当前目录
  FLAGS_stderrthreshold=google::INFO; // 设置日志级别
  //SetLogDestination(google::GLOG_INFO,"./20161120");
  if (FLAGS_check_mode) {
    printf("%s\n", DEBUG_MODE ? "dbg" : "opt");
    return 0;
  }
  LOG(INFO) << "TESTMESSAGE INFO";
  LOG(WARNING) << 2 << "something" << "TESTMESSAGE WARNING"
               << 1 << 'c' << A() << std::endl;
  LOG(ERROR) << "TESTMESSAGE ERROR";
  bool flag = true;
  (flag ? LOG(INFO) : LOG(ERROR)) << "TESTMESSAGE COND";
  //LOG(FATAL) << "TESTMESSAGE FATAL";
  //LOG(FATAL) << "ANOTHRE TESTMESSAGE FATAL";
  // 测试条件输出
  for (int i = 1; i <= 100;i++)
  {
      LOG_IF(INFO,i==100)<<"LOG_IF(INFO,i==100)  google::COUNTER="<<google::COUNTER<<"  i="<<i;
      LOG_EVERY_N(INFO,10)<<"LOG_EVERY_N(INFO,10)  google::COUNTER="<<google::COUNTER<<"  i="<<i;
      LOG_IF_EVERY_N(WARNING,(i>50),10)<<"LOG_IF_EVERY_N(INFO,(i>50),10)  google::COUNTER="<<google::COUNTER<<"  i="<<i;
      LOG_FIRST_N(ERROR,5)<<"LOG_FIRST_N(INFO,5)  google::COUNTER="<<google::COUNTER<<"  i="<<i;
  }

  CHECK(true)<< "TRUE is output.";
  CHECK(false)<< "TRUE is output.";

  return 0;
}
