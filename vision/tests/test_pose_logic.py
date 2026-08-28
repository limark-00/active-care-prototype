from copy import deepcopy
import pytest
from pose_logic import PoseTracker, features


def person(hands=False, horizontal=False):
    coordinates = [(0.5, 0.16)] * 17
    for i, point in {5:(.44,.25),6:(.56,.25),7:(.42,.39),8:(.58,.39),9:(.4,.5),10:(.6,.5),11:(.45,.5),12:(.55,.5),13:(.45,.72),14:(.55,.72),15:(.45,.93),16:(.55,.93)}.items():
        coordinates[i] = point
    if hands:
        coordinates[9], coordinates[10] = (.4,.1), (.6,.1)
    box = [.35,.1,.65,.96]
    if horizontal:
        box = [.1,.74,.95,.96]
        for i, point in {5:(.25,.83),6:(.25,.89),11:(.55,.83),12:(.55,.89),13:(.73,.83),14:(.73,.89),15:(.9,.83),16:(.9,.89)}.items():
            coordinates[i] = point
    return {'bbox': box, 'confidence': .9, 'keypoints': [{'x':x,'y':y,'confidence':.9} for x,y in coordinates]}


def codes(output):
    return {a['code'] for a in output[0]['actions']}


def test_stable_standing_and_hands_need_multiple_frames():
    tracker = PoseTracker()
    assert not codes(tracker.update([person(True)], 640, 480, 0))
    tracker.update([person(True)], 640, 480, .3)
    result = tracker.update([person(True)], 640, 480, .7)
    assert {'hands_up', 'standing'} <= codes(result)
    assert result[0]['id'] == 1


def test_low_confidence_pose_is_unknown_not_standing():
    tracker = PoseTracker()
    for now in (0, .3, .7):
        tracker.update([person(True)], 640, 480, now)
    partial = person(True)
    partial['keypoints'][11]['confidence'] = .1
    result = tracker.update([partial], 640, 480, 1)
    assert not result[0]['pose_reliable']
    assert result[0]['actions'] == []


def test_missing_detection_breaks_identity_and_continuity():
    tracker = PoseTracker()
    tracker.update([person(True)], 640, 480, 0)
    tracker.update([], 640, 480, .3)
    result = tracker.update([person(True)], 640, 480, .7)
    assert result[0]['id'] == 2
    assert result[0]['actions'] == []


def test_long_gap_and_resolution_change_reset_evidence():
    tracker = PoseTracker()
    tracker.update([person(True)], 640, 480, 0)
    result = tracker.update([person(True)], 640, 480, 10)
    assert not result[0]['actions']
    first_id = result[0]['id']
    result = tracker.update([person(True)], 320, 240, 10.3)
    assert result[0]['id'] != first_id


def test_lying_at_start_is_not_a_fall():
    tracker = PoseTracker()
    for now in (0, .3, .7, 1.1, 1.5, 2):
        result = tracker.update([person(horizontal=True)], 640, 480, now)
        assert 'fall_candidate' not in codes(result)
    assert 'horizontal' in codes(result)


def test_fall_requires_standing_downward_transition_and_hold_then_emits_once():
    tracker = PoseTracker()
    for now in (0, .3, .7):
        tracker.update([person()], 640, 480, now)
    for now in (1, 1.3, 1.7, 2):
        result = tracker.update([person(horizontal=True)], 640, 480, now)
        assert 'fall_candidate' not in codes(result)
    result = tracker.update([person(horizontal=True)], 640, 480, 2.3)
    assert 'fall_candidate' in codes(result)
    assert 'fall_candidate' not in codes(tracker.update([person(horizontal=True)], 640, 480, 2.6))


def test_sideways_lean_without_descent_does_not_trigger_fall():
    tracker = PoseTracker()
    for now in (0,.3,.7):
        tracker.update([person()], 640, 480, now)
    lying = person(horizontal=True)
    lying['bbox'][1] -= .4
    lying['bbox'][3] -= .4
    for p in lying['keypoints']:
        p['y'] = max(0, p['y']-.4)
    for now in (1,1.3,1.7,2,2.3):
        assert 'fall_candidate' not in codes(tracker.update([lying], 640, 480, now))


def test_a_later_fall_can_be_reported_after_stable_upright_recovery():
    tracker = PoseTracker()
    reports = []
    for offset in (0, 3):
        for step in (0, .3, .7):
            tracker.update([person()], 640, 480, offset + step)
        for step in (1, 1.3, 1.7, 2, 2.3):
            result = tracker.update([person(horizontal=True)], 640, 480, offset + step)
            if 'fall_candidate' in codes(result):
                reports.append(result[0]['id'])
    assert reports == [1, 1]


def test_multi_person_tracks_do_not_share_hands():
    a, b = person(True), person()
    b['bbox'] = [x + .31 if i % 2 == 0 else x for i,x in enumerate(b['bbox'])]
    for p in b['keypoints']:
        p['x'] += .31
    tracker = PoseTracker()
    for now in (0,.3,.7):
        result = tracker.update([a,b], 640, 480, now)
    assert 'hands_up' in {a['code'] for a in result[0]['actions']}
    assert 'hands_up' not in {a['code'] for a in result[1]['actions']}
    assert result[0]['id'] != result[1]['id']


def test_seated_is_only_a_candidate_and_requires_leg_points():
    p = person()
    for side in ((11,13,15),(12,14,16)):
        p['keypoints'][side[1]] = {'x': .75, 'y': .5, 'confidence': .9}
        p['keypoints'][side[2]] = {'x': .75, 'y': .8, 'confidence': .9}
    assert features(p,640,480)['posture'] == 'seated_candidate'
    for i in (13,14,15,16):
        p['keypoints'][i]['confidence'] = .1
    assert features(p,640,480)['posture'] is None


def test_bad_time_rejected_and_input_not_mutated():
    p = person()
    before = deepcopy(p)
    tracker = PoseTracker()
    tracker.update([p], 640, 480, 1)
    assert p == before
    with pytest.raises(ValueError):
        tracker.update([p], 640, 480, 1)
