# VimDojo

这是一个面向 Vim 新手的本地优先开源刷题平台，体验类似 LeetCode：题库 + 难度分级 + 引导式提示 + 文本结果判定。无需服务器，离线可用。

## 项目定位
- 通过刷题让新手掌握常见 Vim 用法
- 强调效率与 Vim 的“思维方式”
- 提供题库、难度分级与自由练习区
- 学习进度仅保存在本地（无账号系统）

## 目标
- 让完全新手快速上手常用 Vim 操作
- 用引导式题目建立正确的编辑习惯
- 提供可持续扩展的题库与社区贡献方式

## 约束
- 本地运行：完全在用户电脑上运行（支持离线）
- 跨平台：macOS 与 Windows 均可使用
- 无后端服务

## 仓库结构
- `docs/` 产品规范与范围说明
- `content/` YAML 内容（章节、题目、成就）
- `design/` 信息架构、流程、线框
- `meta/` 项目文案与常见问题
- `src/` 前端源码
- `scripts/` 内容构建脚本
- `index.html` 入口页面
- `vite.config.cjs` 构建配置
- `package.json` 前端依赖与脚本

## 内容格式
内容使用 YAML 维护，详见 `docs/content-spec.md`。

## 技术选型
详见 `docs/tech-stack.md`。

## 路线图
详见 `docs/roadmap.md`。

## 许可
Apache-2.0，详见 `LICENSE`。
