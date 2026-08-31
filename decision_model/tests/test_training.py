import unittest
from decision_model.training.train_event_transformer import split_segments
from decision_model.training.train_partial_macbert import freeze_except_last_layers
from decision_model.training.common import evaluate


class TrainingUtilitiesTests(unittest.TestCase):
 def test_split_segments_separates_context_and_canonical_events(self):
  text='当前场景：厨房。\n词条1（L2，摄像头事件）：持续停留。\n词条2（L1，系统状态事件）：输入过期。\n患者能力：未知。\n请判断下一步。'
  parts=split_segments(text)
  self.assertEqual(len(parts),3)
  self.assertIn('当前场景',parts[0]); self.assertNotIn('请判断',parts[0])
  self.assertTrue(parts[1].startswith('词条（L2'))
  self.assertTrue(parts[2].startswith('词条（L1'))

 def test_split_segments_accepts_v3_events_without_risk_codes(self):
  text='地点：厨房。\n事件1（摄像头）：老人一直靠近灶台。\n事件2：水壶还在加热\n能力情况：能够理解提醒。\n请给出下一步。'
  parts=split_segments(text)
  self.assertEqual(len(parts),3)
  self.assertTrue(parts[1].startswith('事件（摄像头）'))
  self.assertTrue(parts[2].startswith('事件：'))

 def test_only_last_requested_encoder_layers_are_unfrozen(self):
  import torch

  class Encoder(torch.nn.Module):
   def __init__(self):
    super().__init__(); self.layer=torch.nn.ModuleList([torch.nn.Linear(2,2) for _ in range(4)])

  class LanguageModel(torch.nn.Module):
   def __init__(self):
    super().__init__(); self.embeddings=torch.nn.Embedding(3,2); self.encoder=Encoder()

  model=LanguageModel(); info=freeze_except_last_layers(model,2)
  self.assertEqual(info['unfrozen_layer_count'],2)
  self.assertFalse(any(p.requires_grad for p in model.embeddings.parameters()))
  self.assertFalse(any(p.requires_grad for layer in model.encoder.layer[:2] for p in layer.parameters()))
  self.assertTrue(all(p.requires_grad for layer in model.encoder.layer[2:] for p in layer.parameters()))

 def test_ordered_metrics_report_underprediction(self):
  rows=[
   {'intervention_level':'I3'},
   {'intervention_level':'I4'},
   {'intervention_level':'I1'},
  ]
  metrics=evaluate(rows,{'intervention_level':['I2','I2','I3']},targets={'intervention_level':['I0','I1','I2','I3','I4']})
  ordinal=metrics['intervention_level']['ordinal']
  self.assertAlmostEqual(ordinal['under_prediction_rate'],2/3)
  self.assertAlmostEqual(ordinal['over_prediction_rate'],1/3)
  self.assertAlmostEqual(ordinal['severe_under_prediction_rate'],1/3)
if __name__=='__main__': unittest.main()
