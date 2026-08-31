from __future__ import annotations
import json
from pathlib import Path

TARGETS={
 'intervention_level':['I0','I1','I2','I3','I4'],
 'alert_mode':['NONE','PAGE_WARNING','URGENT_HELP'],
 'manual_review':[False,True],
 'abstain':[False,True],
}
ORDERED_TARGETS={
 'intervention_level':['I0','I1','I2','I3','I4'],
 'derived_risk_level':['L0','L1','L2','L3','L4'],
}

def load_jsonl(path):
 return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def dump_json(path,value):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def evaluate(rows,predictions,targets=TARGETS):
 from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,f1_score
 report={}
 for target,labels in targets.items():
  true=[r[target] for r in rows]; pred=predictions[target]
  observed=[label for label in labels if label in set(true)]
  report[target]={
   'accuracy':accuracy_score(true,pred),
   'macro_f1':f1_score(true,pred,labels=labels,average='macro',zero_division=0),
   'observed_macro_f1':f1_score(true,pred,labels=observed,average='macro',zero_division=0),
   'all_labels_present':len(observed)==len(labels),
   'labels':labels,
   'confusion_matrix':confusion_matrix(true,pred,labels=labels).tolist(),
   'classification_report':classification_report(true,pred,labels=labels,output_dict=True,zero_division=0),
  }
  if target in ORDERED_TARGETS:
   order={value:index for index,value in enumerate(ORDERED_TARGETS[target])}
   gaps=[order[str(p)]-order[str(t)] for t,p in zip(true,pred)]
   report[target]['ordinal']={
    'mean_absolute_error':sum(abs(gap) for gap in gaps)/len(gaps),
    'within_one_level_accuracy':sum(abs(gap)<=1 for gap in gaps)/len(gaps),
    'under_prediction_rate':sum(gap<0 for gap in gaps)/len(gaps),
    'over_prediction_rate':sum(gap>0 for gap in gaps)/len(gaps),
    'severe_under_prediction_rate':sum(gap<=-2 for gap in gaps)/len(gaps),
    'severe_over_prediction_rate':sum(gap>=2 for gap in gaps)/len(gaps),
   }
 report['mean_macro_f1']=sum(report[t]['macro_f1'] for t in targets)/len(targets)
 report['mean_observed_macro_f1']=sum(report[t]['observed_macro_f1'] for t in targets)/len(targets)
 pairs={}
 for row in rows:
  if row.get('counterfactual_pair_id') and row.get('variant_id')==1:
   pairs.setdefault(row['counterfactual_pair_id'],[]).append(row)
 exact=changed=total=0
 by_id={r['sample_id']:i for i,r in enumerate(rows) if r.get('sample_id')}
 for items in pairs.values():
  if len(items)!=2: continue
  total+=1; idx=[by_id[x['sample_id']] for x in items]
  p=[predictions['intervention_level'][i] for i in idx]
  changed+=p[0]!=p[1]
  exact+=p==[x['intervention_level'] for x in items]
 report['counterfactual']={'pair_count':total,'prediction_changed_rate':changed/total if total else None,'pair_exact_accuracy':exact/total if total else None}
 return report
