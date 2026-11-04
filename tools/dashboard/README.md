# 实验看板

本地只读看板：实时显示每次运行的状态、日志、实验台账与 checkpoint，**不修改任何实验代码**。

- 地址：**http://127.0.0.1:8765/**
- 只绑 `127.0.0.1`，不对外网开放
- 前端每 2 秒刷新状态、每 1.5 秒追一次日志

## 页面内容

| 区块 | 数据来源 | 说明 |
|---|---|---|
| 运行状态 | `logs/runs.jsonl` | 当前在跑什么、已运行多久、退出码 |
| 实验台账 | `results/experiments.csv` | 所有 arm 的 P / R / Micro-F1 / 三类 F1 / CI |
| 运行记录 | `docs/runs/*.md` | 每次实验的完整记录文件 |
| 实时日志 | `logs/*.log` | 选文件 + 自动滚到底 |
| Checkpoint | `D:\pt_ner_work\checkpoints\*\*\train_config.json` | 步数、batch、lr、耗时、大小、最近 loss |

## 启动

```powershell
cd "C:\Users\liush\Desktop\NLP Final Report\Portuguese_ErrorNER"
& "C:\Users\liush\Desktop\NLP Ass#1\.venv\Scripts\python.exe" tools\dashboard\server.py --port 8765
```

只用标准库，不需要装依赖。

## 怎么让运行进入看板

### 方式 A：自己的终端（完整 PowerShell）

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run.ps1 `
    -Name train_E1_seed13 `
    -Exe "C:\Users\liush\Desktop\NLP Ass#1\.venv\Scripts\python.exe" `
    -Args "src\train\train_arm.py","--arm","E1","--data","data/curated/real_300.jsonl","--seed","13","--out","D:/pt_ner_work/checkpoints/E1/seed13"
```

包装器会把 stdout+stderr 写进 `logs/<时间戳>_<名字>.log`，并在 `logs/runs.jsonl` 登记运行状态。

### 方式 B：沙箱内联（AI 会话里执行时用）

沙箱是 **Windows PowerShell 5.1 + ConstrainedLanguage**，`pwsh` 不在 PATH、`.ps1` 不能被执行（`DotSourceNotSupported`）。所以会话内直接内联：

```powershell
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"; $name = "train_E1_seed13"
$file = "${stamp}_$name.log"; $log = "logs\$file"
$start = (Get-Date).ToString("o")
# 1) 登记开始
([ordered]@{id="${stamp}_$name";name=$name;file=$file;cmd="...";start=$start;end=$null;exit=$null;status="running"} |
    ConvertTo-Json -Compress) | Add-Content "logs\runs.jsonl" -Encoding utf8
# 2) 日志头（必须用 Tee 写，见下方编码说明）
@("=== $name ===","start : $start","---") | Tee-Object -FilePath $log | Out-Null
# 3) 跑命令并实时 tee
& $py src\train\train_arm.py --arm E1 ... *>&1 | Tee-Object -FilePath $log -Append
$code = $LASTEXITCODE
# 4) 收尾
@("---","exit  : $code") | Tee-Object -FilePath $log -Append | Out-Null
([ordered]@{id="${stamp}_$name";name=$name;file=$file;cmd="...";start=$start;end=(Get-Date).ToString("o");exit=$code;status=$(if($code -eq 0){"done"}else{"failed"})} |
    ConvertTo-Json -Compress) | Add-Content "logs\runs.jsonl" -Encoding utf8
```

## 两个必须记住的坑

**① 编码：整个日志文件必须用同一种编码。**
Windows PowerShell 5.1 的 `Tee-Object` 只能写 **UTF-16LE + BOM**，而 `Out-File -Encoding utf8` 写 **UTF-8 + BOM**。两者混在同一个文件里会得到 `t i c k` 这种字符间夹空字节的乱码。
**结论：日志头和正文一律用 `Tee-Object` 写**，不要用 `Out-File` 写头。看板服务端按 BOM 自动识别编码，两种都能读。

**② 沙箱里不能执行 `.ps1`。**
ConstrainedLanguage 禁止运行脚本文件，也禁止 `[scriptblock]::Create` / `Invoke-Expression`。`scripts/run.ps1` 是给你自己终端用的，会话内必须用上面的内联写法。

## 目录

```
tools/dashboard/
├── server.py            # 标准库 HTTP 服务（127.0.0.1:8765）
└── static/
    ├── index.html
    ├── app.js
    └── style.css
```

## 与 Git 的关系

`logs/` 已在 `.gitignore` 中（运行产物，不进版本库）；`tools/dashboard/` 是源码，应当提交。
