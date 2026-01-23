# 内容规范（YAML）

本项目使用 YAML 管理学习内容，便于社区协作与内容审核。

## 章节
**文件：** `content/chapters/index.yaml`

结构：
```yaml
chapters:
  - id: movement
    title: 移动
    description: 光标移动与导航。
    order: 1
```

规则：
- `id` 唯一且小写。
- `order` 用于展示顺序。

## 题目
**文件：** `content/problems/<problem-id>.yaml`

结构：
```yaml
id: move-001
title: 移动到行尾
chapter: movement
difficulty: easy # easy | medium | hard
prompt: 在第一行末尾插入 "*"。
hint: 建议使用 0 / $ / w / i
inputText: |
  快速的棕色狐狸
  跳过了懒狗
# targetText 用于判定，提交后才展示。
targetText: |
  快速的棕色狐狸*
  跳过了懒狗

tags:
  - movement
```

规则：
- `id` 在全库唯一。
- `chapter` 必须匹配 `chapters[].id`。
- `difficulty` 取值：`easy`、`medium`、`hard`。
- `targetText` 不在提交前展示。

## 成就
**文件：** `content/achievements/index.yaml`

结构：
```yaml
achievements:
  - id: first-clear
    title: 首次通关
    description: 完成第一道题。
    condition: 完成题目数 >= 1
    iconKey: badge-1
```

规则：
- `id` 唯一。
- `condition` 为可读规则，具体实现后续再定义。

## 标签（建议）
- `movement`（移动）
- `editing`（编辑）
- `text-objects`（文本对象）
- `search-replace`（搜索替换）
- `macros`（宏）
- `mixed`（综合）
