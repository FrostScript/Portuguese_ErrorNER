import argparse,json
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('input'); p.add_argument('output'); a=p.parse_args()
rows=[]; ids=set(); texts=set(); rejected=0
for line_no,line in enumerate(Path(a.input).read_text(encoding='utf-8-sig').splitlines(),1):
 if not line.strip(): continue
 try:
  x=json.loads(line)
  if 'status' in x:
   assert x['status'] in ['keep','revise','reject'],'非法status'
   if x['status']=='reject': rejected+=1; continue
   assert x['sample']['id']==x['id'],'校准ID不一致'
   x=x['sample']
  assert isinstance(x['id'],str) and x['id'] and x['id'] not in ids,'ID为空或重复'
  t=x['text']; assert isinstance(t,str) and t.strip() and t not in texts,'文本为空或重复'
  assert isinstance(x['entities'],list) and x['entities'],'没有实体'
  es=[]
  for e in x['entities']:
   v=e['text']; label=e['label']; n=e['occurrence']
   assert isinstance(v,str) and v and label in ['PER','ORG','LOC'],'实体或标签不合法'
   assert type(n)==int and n>0,'occurrence不合法'
   pos=-len(v)
   for _ in range(n):
    pos=t.find(v,pos+len(v)); assert pos>=0,'找不到指定实体出现位置'
   es.append({'start':pos,'end':pos+len(v),'text':v,'label':label})
  es.sort(key=lambda e:e['start'])
  assert all(es[i]['end']<=es[i+1]['start'] for i in range(len(es)-1)),'实体重叠或重复'
  rows.append({'id':x['id'],'text':t,'entities':es}); ids.add(x['id']); texts.add(t)
 except Exception as ex: raise SystemExit(f'第{line_no}行失败：{ex}；未写出结果')
out=Path(a.output)
assert out.resolve()!=Path(a.input).resolve(),'不得覆盖输入文件'
assert not out.exists(),'输出文件已存在，请换文件名'
out.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8')
print(f'通过 {len(rows)} 条；校准拒绝 {rejected} 条。仅验证结构和偏移，不保证语义完整正确。')
