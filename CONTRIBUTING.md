# 贡献指南

感谢你的帮助！本项目以内容贡献为主。

## 可以贡献的内容
- 新题目（YAML）
- 章节说明
- 成就设计
- 文档完善

## 添加题目
1) 在 `content/problems/` 新建 `<problem-id>.yaml`。
2) 按 `docs/content-spec.md` 的结构填写。
3) 题目描述清晰、适合新手，并提供有用的 `hint`。
4) 确保 `targetText` 与目标一致（精确匹配）。

## 风格建议
- 简洁、清晰、面向新手
- 避免含糊目标
- 尽量拆成可学习的小步骤

## 提交前检查
- `id` 唯一
- `chapter` 有效
- `difficulty` 取值为 `easy`/`medium`/`hard`
- `targetText` 正确

## 许可
提交即表示同意以 Apache-2.0 授权你的贡献。
