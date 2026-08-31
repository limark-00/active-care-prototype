from __future__ import annotations
import argparse,json,time
from datetime import datetime,timezone
from pathlib import Path
from .common import TARGETS,dump_json,evaluate,load_jsonl

def main():
 p=argparse.ArgumentParser(); p.add_argument('--data-dir',type=Path,default=Path(__file__).parents[1]/'data/v2'); p.add_argument('--output-root',type=Path,default=Path(__file__).parents[1]/'outputs'); a=p.parse_args()
 import numpy as np
 from joblib import dump
 from sklearn.feature_extraction.text import TfidfVectorizer
 from sklearn.linear_model import SGDClassifier
 run=a.output_root/f"baseline-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"; run.mkdir(parents=True)
 splits={n:load_jsonl(a.data_dir/f'{n}.jsonl') for n in ('train','validation','test','ood_test','natural_test')}
 vectorizer=TfidfVectorizer(analyzer='char',ngram_range=(2,5),min_df=2,max_features=120000,sublinear_tf=True,dtype=np.float32)
 started=time.time(); x_train=vectorizer.fit_transform(r['input_text'] for r in splits['train'])
 models={}
 for target in TARGETS:
  model=SGDClassifier(loss='log_loss',class_weight='balanced',max_iter=2000,tol=1e-4,random_state=20260830,early_stopping=True,validation_fraction=.1,n_iter_no_change=8)
  model.fit(x_train,[r[target] for r in splits['train']]); models[target]=model
 dump({'vectorizer':vectorizer,'models':models,'targets':TARGETS},run/'model.joblib')
 reports={}
 for name,rows in splits.items():
  x=vectorizer.transform(r['input_text'] for r in rows)
  pred={t:models[t].predict(x).tolist() for t in TARGETS}; reports[name]=evaluate(rows,pred); dump_json(run/f'metrics-{name}.json',reports[name])
 dump_json(run/'run.json',{'model':'tfidf-sgd-logistic','seconds':time.time()-started,'features':len(vectorizer.vocabulary_),'reports':{k:v['mean_macro_f1'] for k,v in reports.items()}})
 print(json.dumps({'run_dir':str(run),'scores':{k:v['mean_macro_f1'] for k,v in reports.items()}},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
