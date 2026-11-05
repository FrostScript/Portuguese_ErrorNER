# 模板版
五个 prompts 文件仅保留占位符，不内嵌种子、标签清单或具体样本量。不附预填副本。
数据独立放在 data/seeds_20.json，标签独立放在 data/label_schema.json。后者为可修改工作规范，不要求组员另交文件。

需要 Python 3，无第三方依赖。解压后在目录中执行：
```
python prepare_prompt.py e2_generation --batch e2_001 --n 20 --out e2_request.txt
```
只有执行时才读取并注入数据。将生成的请求发送模型，回复保存为 e2_raw.jsonl：
```
python convert_validate.py e2_raw.jsonl e2_checked.jsonl
```
E3依次执行，每步实际发送模型并保存回复后再进行下一步：
```
python prepare_prompt.py e3_guidance --out step1.txt
python prepare_prompt.py e3_refiner --initial initial_guidance.json --out step2.txt
python prepare_prompt.py e3_generation --refined refined_guidance.json --batch e3_001 --n 20 --out step3.txt
python prepare_prompt.py e3_calibrator --refined refined_guidance.json --generated e3_raw.jsonl --out step4.txt
python convert_validate.py e3_calibrated.jsonl e3_checked.jsonl
```
四步回复分别保存为 initial_guidance.json、refined_guidance.json、e3_raw.jsonl、e3_calibrated.jsonl。
可用 --schema 路径、--seeds 路径替换外部配置。--n 是每批生成数量，由调用时指定。
两组共用相同种子、模型和基础校验；先各试20条，最终各保留300条，种子不混入训练。批次ID不能重复。跨批合并原始或校准JSONL再统一转换，以检查重复。校验器检查结构和偏移，不保证语义与漏标正确，需人工抽检；当前校验器按本项目PER/ORG/LOC校验。E3为四阶段项目适配草案，不声称严格复现论文。自然葡语若需额外预处理，两组须统一并重新计算偏移。
