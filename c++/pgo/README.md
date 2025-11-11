# C++ PGO (Profile-Guided Optimization) 性能优化示例

本项目演示在 C++ 中使用 Profile-Guided Optimization (PGO) 来优化程序性能，并针对 macOS/Clang 的 LLVM PGO 流程进行了完整验证，亦给出 GCC 的等价做法。目录内已包含：`main.cpp`（工作负载示例）、`Makefile`（PGO 自动化）、`pgo_optimize.sh`（一键流程，macOS/Clang）。

## 什么是 PGO？

Profile-Guided Optimization (PGO) 是一种“以真实运行数据为依据”的编译优化方式：先用插桩或采样收集程序在代表性输入下的执行画像（热点路径、分支概率、调用频次等），再用该画像指导编译器做更贴近实际负载的优化决策（代码布局、内联、分支预测、循环向量化与调度、寄存器分配等）。

### PGO 的两阶段流程
- 训练构建与采样：生成带插桩的二进制并运行，收集画像（Clang: `*.profraw`；GCC: `*.gcda`）。
- 应用构建：用画像文件（Clang: `default.profdata`；GCC: 直接读取 `*.gcda`）指导优化，输出 PGO 优化二进制。

## 快速开始

### Mac/Clang编译

```bash
# 1) 普通构建
clang++ -std=c++17 -march=native -o main main.cpp

# 2) 插桩构建
clang++ -std=c++17 -march=native -fprofile-instr-generate -o main_pgo_gen main.cpp

# 3) 运行收集 raw profile（可用通配符生成多文件）
LLVM_PROFILE_FILE="default_%p.profraw" ./main_pgo_gen >/dev/null

# 4) 合并画像
llvm-profdata merge default_*.profraw -o default.profdata

# 5) 应用构建（使用画像）
clang++ -std=c++17 -march=native -fprofile-instr-use=default.profdata -o main_pgo main.cpp

# 6) 性能对比
/usr/bin/time -p ./main >/dev/null
/usr/bin/time -p ./main_pgo >/dev/null
```

### Linux/GCC 编译
```bash
# 生成阶段：带 profile 插桩
g++ -std=c++17 -O3 -march=native -fprofile-generate -o main_gen main.cpp
./main_gen >/dev/null   # 运行后生成 *.gcda 数据

# 应用阶段：使用 profile 做优化
g++ -std=c++17 -O3 -march=native -fprofile-use -fprofile-correction -o main_pgo main.cpp
```

## 比较汇编代码（C++/Clang）

有两种常用方式：
- 比较机器码反汇编（更贴近指令级差异）
```bash
# 生成并对比反汇编
clang++ -std=c++17 -O3 -march=native -o main main.cpp
clang++ -std=c++17 -O3 -march=native -fprofile-instr-use=default.profdata -o main_pgo main.cpp

# 使用 llvm-objdump 反汇编
llvm-objdump -d ./main     > no_pgo.asm
llvm-objdump -d ./main_pgo > with_pgo.asm

diff -u no_pgo.asm with_pgo.asm | less
```
- 直接生成汇编（可读性好，但与最终指令排布可能有差异）
```bash
clang++ -std=c++17 -O3 -march=native -S -o no_pgo.s main.cpp
clang++ -std=c++17 -O3 -march=native -fprofile-instr-use=default.profdata -S -o with_pgo.s main.cpp

diff -u no_pgo.s with_pgo.s | less
```

观察要点：
- 热路径上的基本块顺序与布局变化（跳转更少、顺序更线性）。
- 分支预测相关指令序列与概率性调整（条件跳转的顺序与距离）。
- 内联导致的调用消失与常量传播、循环优化触发。
- 向量化指令（如 `vmovapd/vmulps` 等）是否更容易被触发。

## PGO 原理与优化机制深入分析（C++/GCC/Clang）

PGO 的核心流程分为两阶段：
- 训练构建与采样（GCC: `-fprofile-generate`；Clang: `-fprofile-instr-generate` 产生 `*.profraw`）
- 应用构建（GCC: `-fprofile-use -fprofile-correction`；Clang: `-fprofile-instr-use=default.profdata`）

训练阶段收集的数据（不同编译器实现细节略有差异）：
- 基本块/边覆盖与频次：热点识别。
- 分支概率与错预测：`if/else`、`switch` 的实际走向与错率。
- 调用图与内联机会：函数调用次数与栈深度分布。
- 值/目标分析：间接调用目标、虚调用落点等（Clang 更常见）。

应用阶段编译器做出的优化：
- 代码布局（热/冷分区）提升 I-Cache 与预测命中。
- 分支优化用真实概率指导 `likely/unlikely`。
- 热点内联与去虚化（Indirect Call Promotion）。
- 循环展开、向量化与指令调度结合访存模式与迭代分布。
- 热路径寄存器分配更激进，减少溢出与访存。
- 与 LTO/ThinLTO 协同提升跨 TU 的优化深度。

## 为什么本示例中 PGO 会更快（结合示例代码）

本示例的核心热点在 <mcfile name="main.cpp" path="/Volumes/my/github/mylib/c++/pgo/main.cpp"></mcfile> 的函数 <mcsymbol name="branchySum" filename="main.cpp" path="/Volumes/my/github/mylib/c++/pgo/main.cpp" startline="29" type="function"></mcsymbol> 上：
- 热分支占比高：在每次迭代中，约 `hot_ratio%` 的路径走向“热分支”，执行小函数 <mcsymbol name="fast_op" filename="main.cpp" path="/Volumes/my/github/mylib/c++/pgo/main.cpp" startline="21" type="function"></mcsymbol>；其余走向“冷分支”，执行较重的 <mcsymbol name="slow_op" filename="main.cpp" path="/Volumes/my/github/mylib/c++/pgo/main.cpp" startline="23" type="function"></mcsymbol>。
- 轻量级随机源：示例使用 xorshift（而非 `std::mt19937`）产生数据，避免随机数生成开销掩盖热点分支，从而让 PGO 更专注于真实热点优化。

PGO 如何利用这些信息变快：
- 基本块布局（代码放置）
  - 训练阶段记录每个分支的真实概率与执行频次；应用阶段将“热路径”基本块线性放置在一起，减少指令缓存（I-Cache）与分支跳转带来的开销。
  - 在本示例中，fast 路径更可能被顺序执行，减少冷块干扰与跳转开销。
- 分支优化与预测
  - 编译器用画像数据替代启发式，重写条件分支的安排与预测提示；实际高概率的 if 分支被视为 `likely`，错预测率下降。
  - 对 `if ((x % 100) < hot_ratio)` 的真实概率建模，使热分支更“直线”，冷分支更“远离”。
- 热点函数内联与联动优化
  - 热路径上的小函数更倾向被内联，消除调用开销，并触发常量传播、强度削减、寄存器分配优化等后续机会。
  - 示例里的 <mcsymbol name="fast_op" filename="main.cpp" path="/Volumes/my/github/mylib/c++/pgo/main.cpp" startline="21" type="function"></mcsymbol> 作为热路径微函数，内联后循环体更紧凑；而 <mcsymbol name="slow_op" filename="main.cpp" path="/Volumes/my/github/mylib/c++/pgo/main.cpp" startline="23" type="function"></mcsymbol> 被放入冷区，减少 I-Cache 污染。
- 热路径寄存器分配更激进
  - 在热点循环体中，PGO 指导编译器把寄存器资源优先分配给高频值，减少溢出与访存（尤其在长循环中更显著）。
- 与 LTO 协同
  - 示例默认开启 `-flto`，PGO 的布局与分支概率配合跨单元优化，进一步改善内联与调度机会（取决于编译器版本）。

验证与参数建议（可复制粘贴）
- 调整体量与热比例，强化信号：
```bash
make clean && make compare RUN_ROUNDS=3 TRAIN_N=15000000 TRAIN_HOT=99 TEST_N=15000000 TEST_HOT=99
```
- 或使用脚本（统一轮数/规模/热比例）：
```bash
bash pgo_optimize.sh 3 15000000 99
```
- 如仍不明显：继续增大 `N` 或把 `hot_ratio` 提到 `99`，确保训练与测试使用一致参数；保证 `llvm-profdata` 可用（macOS 可 `xcode-select --install` 或 `brew install llvm` 并将 `$(brew --prefix llvm)/bin` 入 PATH）。

观察点（如何判断 PGO 起效）
- `real` 时间下降（通常 5–20%，硬件/版本相关）；PGO 二进制大小可能略增。
- 生成 `no_pgo.s` 与 `with_pgo.s`（或反汇编）对比：
  - 热路径基本块更连续、跳转更少；
  - 调用消失（fast_op 被内联），冷函数 slow_op 远离热段；
  - 条件跳转顺序与距离变化，更贴近实际概率。

与原理对应的示例代码片段说明
- 热分支建模：
```cpp
if ((x % 100) < hot_ratio) { // 热分支
    sum += fast_op(x);
} else {                     // 冷分支
    sum += slow_op(x);
}
```
- 轻量级随机源（减少噪声）：
```cpp
seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
int x = static_cast<int>(seed & 0x3fffffff);
```
