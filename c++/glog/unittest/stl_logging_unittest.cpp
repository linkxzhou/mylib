// Copyright (c) 2003, Google Inc.
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

#include "base/config.h"
#include "glog/logging.h"
#include "glog/stl_logging.h"

#ifdef HAVE_USING_OPERATOR

#include <functional>
#include <iostream>
#include <map>
#include <ostream>
#include <string>
#include <vector>
#include <unordered_set>

#ifdef __GNUC__
// C++0x isn't enabled by default in GCC and libc++ does not have
// non-standard ext/* and tr1/unordered_*.
#if defined(_LIBCPP_VERSION)
#ifndef GLOG_STL_LOGGING_FOR_UNORDERED
#define GLOG_STL_LOGGING_FOR_UNORDERED
#endif
#else
#ifndef GLOG_STL_LOGGING_FOR_EXT_HASH
#define GLOG_STL_LOGGING_FOR_EXT_HASH
#endif
#ifndef GLOG_STL_LOGGING_FOR_EXT_SLIST
#define GLOG_STL_LOGGING_FOR_EXT_SLIST
#endif
#ifndef GLOG_STL_LOGGING_FOR_TR1_UNORDERED
#define GLOG_STL_LOGGING_FOR_TR1_UNORDERED
#endif
#endif
#endif

#include <gtest/gtest.h>

using namespace std;
#ifdef GLOG_STL_LOGGING_FOR_EXT_HASH
using namespace __gnu_cxx;
#endif

struct user_hash
{
  size_t operator()(int x) const { return x; }
};

void TestSTLLogging()
{
  {
    // Test a sequence.
    vector<int> v;
    v.push_back(10);
    v.push_back(20);
    v.push_back(30);
    ostringstream ss;
    ss << v;
    EXPECT_EQ(ss.str(), "10 20 30");
    vector<int> copied_v(v);
    CHECK_EQ(v, copied_v); // This must compile.
  }

  {
    // Test a long sequence.
    vector<int> v;
    string expected;
    for (int i = 0; i < 100; i++)
    {
      v.push_back(i);
      if (i > 0)
        expected += ' ';
      char buf[256];
      sprintf(buf, "%d", i);
      expected += buf;
    }
    v.push_back(100);
    expected += " ...";
    ostringstream ss;
    ss << v;
    CHECK_EQ(ss.str(), expected.c_str());
  }
}

int main(int, char **)
{
  TestSTLLogging();
  std::cout << "PASS\n";
  return 0;
}

#else

#include <iostream>

int main(int, char **)
{
  std::cout << "We don't support stl_logging for this compiler.\n"
            << "(we need compiler support of 'using ::operator<<' "
            << "for this feature.)\n";
  return 0;
}

#endif // HAVE_USING_OPERATOR
