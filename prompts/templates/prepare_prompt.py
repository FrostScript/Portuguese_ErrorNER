import argparse,json
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('stage',choices=['e2_generation','e2_negative','e3_guidance','e3_refiner','e3_generation','e3_negative','e3_calibrator','e3_calibrator_negative']); p.add_argument('--batch',default='batch001'); p.add_argument('--n',type=int,default=20); p.add_argument('--initial'); p.add_argument('--refined'); p.add_argument('--generated'); p.add_argument('--schema',default=str(Path(__file__).resolve().parent/'data/label_schema.json')); p.add_argument('--seeds',default=str(Path(__file__).resolve().parent/'data/seeds_20.json')); p.add_argument('--out',default='ready_prompt.txt'); a=p.parse_args()
r=Path(__file__).resolve().parent
seeds=json.loads(Path(a.seeds).read_text(encoding='utf-8-sig'))
s=(r/'prompts'/f'{a.stage}.txt').read_text(encoding='utf-8')
values={'LABEL_SCHEMA':Path(a.schema).read_text(encoding='utf-8-sig'),'SEED_EXAMPLES':json.dumps([{'text':x['text'],'entities':x['entities']} for x in seeds],ensure_ascii=False),'N':str(a.n),'BATCH_ID':a.batch}
for key,path in [('INITIAL_GUIDANCE',a.initial),('REFINED_GUIDANCE',a.refined),('GENERATED_BATCH',a.generated)]:
 if '{'+key+'}' in s:
  if not path: p.error('缺少参数 '+key)
  values[key]=Path(path).read_text(encoding='utf-8-sig')
for k,v in values.items(): s=s.replace('{'+k+'}',v)
Path(a.out).write_text(s,encoding='utf-8'); print('已生成',a.out)
