## PyTorch 的这个设计（Dataset -> DataLoader）
该设计体现了软件工程中经典的“解耦”(Decoupling) 和 “责任分离” (Separation of Concerns) 原则。

**Dataset**负责取数据,定义了“是什么”（数据本身）.核心是:
1. __getitem__(index)。给它一个索引 5，它就给你第 5 张图。它根本不知道什么是 Batch，什么是 Shuffle。
2. 特点：它通常是静态的、顺序无关的。

**Dataloader**负责运数据, 定义了“怎么用”（如何喂给模型）,核心是:
1. Batching（打包）：把 16 张图打包成一个 (16, 3, 32, 32) 的大张量
2. Shuffling（洗牌）：每轮训练开始前把数据打乱
3. Multiprocessing（多进程）：启动多个num_workers去Dataset里取数据

### 为什么这样设计?
如果把这两者合在一起，会有什么灾难？

**复用性差**

假设你想换一种洗牌方式，或者想改变打包逻辑（比如文本变长打包），你就得去改数据读取的代码。现在，你只需要换一个 DataLoader 的参数，Dataset 完全不用动。

**代码乱**

数据读取逻辑（比如解析 JSON、解码 JPEG）通常很复杂。如果混入多线程逻辑，代码会变得极其难以维护。