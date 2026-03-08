# Multilingual README Design

## Goal
Add Traditional Chinese (繁體中文) version of README with one-click language switching.

## Structure
- `README.md` — English (existing, add language switcher at top)
- `README.zh-TW.md` — Traditional Chinese (new file, full translation)

## Language Switcher
Line 1 of both files, before the `#` title:

**English version:**
```markdown
🌐 Language: **English** | [繁體中文](README.zh-TW.md)
```

**Chinese version:**
```markdown
🌐 語言: [English](README.md) | **繁體中文**
```

Current language is bold (not linked). Other language is a clickable link.

## Translation Rules
- **Translate:** Prose, section headers, table labels, descriptions
- **Keep English:** Code blocks, file paths, CLI commands, ASCII pipeline diagram
- **Hybrid terms:** English technical terms with Chinese in-context, e.g. "批次正規化（Batch Normalization）"
- **Proper nouns unchanged:** Shift-GCN, MediaPipe, NTU RGB+D, PyTorch, CUDA, SGD, Nesterov

## Scope
- Two files modified/created: `README.md`, `README.zh-TW.md`
- No code, config, or structural changes
- `README.zh-TW.md` is a complete translation, not a summary
