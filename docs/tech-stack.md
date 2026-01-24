# 技术选型（前端本地版）

本项目以“浏览器直接打开、完全离线、本地存储”为前提，选择最轻量的前端方案。

## 结论
- 框架：原生 JavaScript + Vite
- 编辑器：CodeMirror 6 + Vim 键位插件
- 内容：YAML 作为源文件，构建时转换为 JSON
- 进度：localStorage（支持导出/导入）
- 部署：静态构建产物，离线可用

## 选型理由
- Vite 构建简单、产物可离线打开（配合 `base: "./"`）。
- 原生 JS 学习成本低，适合不熟悉前端的维护者。
- 运行时不读文件，避免本地浏览器的文件访问限制。
- localStorage 足够支撑 MVP 的本地进度需求。

## 内容构建流程
1) 维护 `content/` 下的 YAML。
2) 执行 `node scripts/build-content.mjs` 生成 `src/generated/content.json`。
3) Vite 打包输出 `dist/`。

## 本地进度导入/导出
- 导出：生成 JSON 文件下载到本地。
- 导入：读取 JSON 文件并覆盖本地进度。
