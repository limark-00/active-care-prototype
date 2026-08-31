from __future__ import annotations
import argparse,json,random,re,time,time as _time
from datetime import datetime,timezone
from pathlib import Path
from .common import TARGETS,dump_json,evaluate,load_jsonl

def split_segments(text):
 lines=[x.strip() for x in text.splitlines() if x.strip()]
 events=[]; context=[]
 for line in lines:
  if re.match(r'^词条\d+（',line): events.append(re.sub(r'^词条\d+','词条',line))
  elif re.match(r'^事件\d+(?:（[^）]+）)?[：:]',line): events.append(re.sub(r'^事件\d+','事件',line))
  elif not line.startswith('请判断'): context.append(line)
 return ['\n'.join(context),*events]

def choose_device(torch,name):
 if name!='auto': return torch.device(name)
 return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def build_model(torch,input_dim,d_model=128,targets=TARGETS):
 nn=torch.nn
 class Model(nn.Module):
  def __init__(self):
   super().__init__(); self.proj=nn.Sequential(nn.Linear(input_dim,d_model),nn.LayerNorm(d_model),nn.GELU(),nn.Dropout(.15)); self.types=nn.Embedding(2,d_model)
   layer=nn.TransformerEncoderLayer(d_model,4,256,.15,batch_first=True,norm_first=True,activation='gelu'); self.encoder=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
   self.heads=nn.ModuleDict({k:nn.Linear(d_model,len(v)) for k,v in targets.items()})
  def forward(self,x,mask):
   types=torch.ones(mask.shape,dtype=torch.long,device=x.device); types[:,0]=0
   h=self.proj(x)+self.types(types); h=self.encoder(h,src_key_padding_mask=~mask)
   return {k:head(h[:,0]) for k,head in self.heads.items()}
 return Model()

def main():
 p=argparse.ArgumentParser(); root=Path(__file__).parents[1]; p.add_argument('--data-dir',type=Path,default=root/'data/v2'); p.add_argument('--output-root',type=Path,default=root/'outputs'); p.add_argument('--model-name',default='hfl/chinese-macbert-base'); p.add_argument('--device',default='auto'); p.add_argument('--epochs',type=int,default=24); p.add_argument('--batch-size',type=int,default=128); p.add_argument('--encode-batch-size',type=int,default=32); a=p.parse_args()
 import numpy as np,torch
 total_started=time.time()
 from transformers import AutoModel,AutoTokenizer
 from tqdm import tqdm
 random.seed(20260830); np.random.seed(20260830); torch.manual_seed(20260830)
 device=choose_device(torch,a.device); run=a.output_root/f"event-transformer-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"; run.mkdir(parents=True)
 names=('train','validation','test','ood_test','natural_test'); splits={n:load_jsonl(a.data_dir/f'{n}.jsonl') for n in names}
 segment_lists={}; vocab={}; texts=[]
 for name,rows in splits.items():
  current=[]
  for row in rows:
   ids=[]
   for segment in split_segments(row['input_text']):
    if segment not in vocab: vocab[segment]=len(texts); texts.append(segment)
    ids.append(vocab[segment])
   current.append(ids)
  segment_lists[name]=current
 print(f'Encoding {len(texts)} unique text segments on {device}...')
 encode_started=time.time()
 tokenizer=AutoTokenizer.from_pretrained(a.model_name,trust_remote_code=False); lm=AutoModel.from_pretrained(a.model_name,trust_remote_code=False).to(device).eval()
 vectors=[]
 with torch.inference_mode():
  for start in tqdm(range(0,len(texts),a.encode_batch_size)):
   batch=tokenizer(texts[start:start+a.encode_batch_size],padding=True,truncation=True,max_length=256,return_tensors='pt'); batch={k:v.to(device) for k,v in batch.items()}
   out=lm(**batch).last_hidden_state; mask=batch['attention_mask'].unsqueeze(-1); pooled=(out*mask).sum(1)/mask.sum(1).clamp_min(1); vectors.append(pooled.cpu())
 table=torch.cat(vectors).float(); encode_seconds=time.time()-encode_started; model_commit=getattr(lm.config,'_commit_hash',None); del lm
 dump_json(run/'encoder.json',{'model_name':a.model_name,'commit':model_commit,'unique_segments':len(texts),'device':str(device)})
 max_slots=max(len(x) for values in segment_lists.values() for x in values)
 label_maps={k:{str(v):i for i,v in enumerate(vals)} for k,vals in TARGETS.items()}
 def pack(name):
  rows=splits[name]; idx=torch.zeros((len(rows),max_slots),dtype=torch.long); mask=torch.zeros((len(rows),max_slots),dtype=torch.bool)
  for i,values in enumerate(segment_lists[name]): idx[i,:len(values)]=torch.tensor(values); mask[i,:len(values)]=True
  ys={k:torch.tensor([label_maps[k][str(r[k])] for r in rows]) for k in TARGETS}; return idx,mask,ys
 packed={n:pack(n) for n in names}; model=build_model(torch,table.shape[1]).to(device); opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=.02)
 ce=torch.nn.CrossEntropyLoss(); best=-1.; stale=0; log=(run/'train.jsonl').open('w')
 def predict(name):
  model.eval(); idx,mask,_=packed[name]; collected={k:[] for k in TARGETS}
  with torch.inference_mode():
   for s in range(0,len(idx),a.batch_size):
    ids=idx[s:s+a.batch_size]; m=mask[s:s+a.batch_size].to(device); x=table[ids].to(device); out=model(x,m)
    for k,vals in TARGETS.items(): collected[k]+= [vals[i] for i in out[k].argmax(1).cpu().tolist()]
  return collected
 training_started=time.time()
 for epoch in range(1,a.epochs+1):
  model.train(); idx,mask,ys=packed['train']; order=torch.randperm(len(idx)); total=0.
  for s in range(0,len(order),a.batch_size):
   take=order[s:s+a.batch_size]; ids=idx[take]; m=mask[take].to(device); x=table[ids].to(device); out=model(x,m); loss=sum(ce(out[k],ys[k][take].to(device)) for k in TARGETS); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); total+=loss.item()*len(take)
  pred=predict('validation'); metrics=evaluate(splits['validation'],pred); score=metrics['mean_macro_f1']; entry={'epoch':epoch,'train_loss':total/len(idx),'validation_mean_macro_f1':score,'elapsed_training_seconds':time.time()-training_started}; log.write(json.dumps(entry)+'\n'); log.flush(); print(entry)
  if score>best+1e-4: best=score; stale=0; torch.save({'state_dict':model.state_dict(),'input_dim':table.shape[1],'d_model':128,'targets':TARGETS,'encoder':a.model_name},run/'best_model.pt')
  else: stale+=1
  if stale>=5: break
 log.close(); saved=torch.load(run/'best_model.pt',map_location=device,weights_only=True); model.load_state_dict(saved['state_dict'])
 reports={}
 for name in names:
  reports[name]=evaluate(splits[name],predict(name)); dump_json(run/f'metrics-{name}.json',reports[name])
 dump_json(run/'run.json',{'model':'frozen-macbert-event-transformer','device':str(device),'best_validation_score':best,'epochs_completed':epoch,'total_seconds':time.time()-total_started,'encoding_seconds':encode_seconds,'training_seconds':time.time()-training_started,'reports':{k:v['mean_macro_f1'] for k,v in reports.items()}})
 print(json.dumps({'run_dir':str(run),'scores':{k:v['mean_macro_f1'] for k,v in reports.items()}},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
