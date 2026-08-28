"""Deterministic pose heuristics, NOT a trained action classifier or fall detector."""
from dataclasses import dataclass, field
from math import acos, degrees, hypot

CONFIDENCE = 0.45
MAX_GAP = 1.5
HOLD_SECONDS = 0.6
LABELS = {
    'standing': '站姿候选', 'seated_candidate': '屈膝 / 坐姿候选',
    'leaning': '躯干倾斜', 'horizontal': '横卧姿态候选',
    'one_hand_up': '单手举起', 'hands_up': '双手举起',
    'moving': '画面内明显位移', 'fall_candidate': '疑似跌倒 · 请人工核查',
}


def midpoint(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def angle(a, b, c):
    u, v = (a[0] - b[0], a[1] - b[1]), (c[0] - b[0], c[1] - b[1])
    denominator = hypot(*u) * hypot(*v)
    if denominator < 1e-6:
        return 0.0
    return degrees(acos(max(-1, min(1, (u[0] * v[0] + u[1] * v[1]) / denominator))))


def features(person, width, height):
    points = person['keypoints']

    def point(index):
        p = points[index]
        return (p['x'] * width, p['y'] * height) if p['confidence'] >= CONFIDENCE else None

    shoulders, hips = [point(i) for i in (5, 6)], [point(i) for i in (11, 12)]
    if any(p is None for p in shoulders + hips):
        return {'posture': None, 'hands': None, 'hip': None, 'body_height': 0}
    shoulder, hip = midpoint(*shoulders), midpoint(*hips)
    torso = hypot(hip[0] - shoulder[0], hip[1] - shoulder[1])
    body_height = max(1, (person['bbox'][3] - person['bbox'][1]) * height)
    if torso < max(12, body_height * 0.12):
        return {'posture': None, 'hands': None, 'hip': None, 'body_height': 0}
    tilt = degrees(acos(max(-1, min(1, (hip[1] - shoulder[1]) / torso))))
    raised = [point(i) is not None and point(i)[1] < shoulders[j][1] - torso * 0.22 for j, i in enumerate((9, 10))]
    hands = 'hands_up' if all(raised) else 'one_hand_up' if any(raised) else None
    knee_angles = []
    for side in ((11, 13, 15), (12, 14, 16)):
        values = [point(i) for i in side]
        if all(p is not None for p in values):
            knee_angles.append(angle(*values))
    box_width = (person['bbox'][2] - person['bbox'][0]) * width
    posture = None
    if 65 <= tilt <= 115 and box_width > body_height * 1.05:
        posture = 'horizontal'
    elif 35 <= tilt < 65:
        posture = 'leaning'
    elif tilt < 30 and knee_angles:
        if min(knee_angles) > 155:
            posture = 'standing'
        elif max(knee_angles) < 135:
            posture = 'seated_candidate'
    return {'posture': posture, 'hands': hands, 'hip': hip, 'body_height': body_height}


def iou(a, b):
    intersect = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = lambda box: max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    return intersect / max(1e-8, area(a) + area(b) - intersect)


@dataclass
class Track:
    id: int
    bbox: list
    last_seen: float
    holds: dict = field(default_factory=dict)
    positions: list = field(default_factory=list)
    upright: tuple | None = None
    fall_since: float | None = None
    fall_reported: bool = False

    def actions(self, person, width, height, now):
        f = features(person, width, height)
        reliable = f['hip'] is not None
        codes = {v for v in (f['posture'], f['hands']) if v}
        actions = []
        self.holds = {code: value for code, value in self.holds.items() if code in codes}
        for code in sorted(codes):
            start, count = self.holds.get(code, (now, 0))
            self.holds[code] = (start, count + 1)
            if count + 1 >= 3 and now - start >= HOLD_SECONDS:
                actions.append({'code': code, 'label': LABELS[code], 'evidence': '关键点几何关系连续保持 ≥0.6 秒且至少 3 帧；规则判断。'})
        if not reliable:
            self.positions.clear()
            self.upright = None
            self.fall_since = None
            return [], False
        self.positions = [(at, p) for at, p in self.positions if now - at <= 1.2]
        self.positions.append((now, f['hip']))
        if self.positions and now - self.positions[0][0] >= 0.6:
            first = self.positions[0][1]
            if hypot(f['hip'][0] - first[0], f['hip'][1] - first[1]) > f['body_height'] * 0.18:
                actions.append({'code': 'moving', 'label': LABELS['moving'], 'evidence': '髋部中点在画面内发生相对位移；不等于行走，也不是米制距离。'})
        # Require previously stable standing, a rapid downward displacement, then sustained horizontal pose.
        if any(a['code'] == 'standing' for a in actions):
            self.upright = (now, f['hip'][1], f['body_height'])
            self.fall_since = None
            self.fall_reported = False
        elif f['posture'] == 'horizontal':
            if self.fall_since is None and self.upright:
                at, hip_y, old_height = self.upright
                if now - at <= 1.2 and f['hip'][1] - hip_y >= 0.30 * old_height:
                    self.fall_since = now
            if self.fall_since is not None and now - self.fall_since >= 1.2 and not self.fall_reported:
                actions.append({'code': 'fall_candidate', 'label': LABELS['fall_candidate'], 'evidence': '稳定站姿后快速向下位移，随后横卧候选持续 ≥1.2 秒；可能误报或漏报。'})
                self.fall_reported = True
        else:
            self.fall_since = None
        return actions, reliable


class PoseTracker:
    """Short-lived geometric association, no face recognition or identity inference."""
    def __init__(self):
        self.tracks = {}
        self.sequence = 0
        self.last_time = None
        self.dimensions = None

    def update(self, persons, width, height, now):
        if self.last_time is not None and now <= self.last_time:
            raise ValueError('Frame times must increase')
        if self.last_time is not None and (now - self.last_time > MAX_GAP or self.dimensions != (width, height)):
            self.tracks.clear()
        self.last_time, self.dimensions = now, (width, height)
        self.tracks = {k: t for k, t in self.tracks.items() if now - t.last_seen <= MAX_GAP}
        candidates = []
        for index, person in enumerate(persons):
            box = person['bbox']
            center = midpoint(box[:2], box[2:])
            scores = []
            for track in self.tracks.values():
                # A missed detection resets temporal evidence rather than inventing continuity.
                if track.last_seen != self.previous_frame_time:
                    continue
                old_center = midpoint(track.bbox[:2], track.bbox[2:])
                scale = max(0.08, hypot(track.bbox[2] - track.bbox[0], track.bbox[3] - track.bbox[1]))
                distance = hypot(center[0] - old_center[0], center[1] - old_center[1]) / scale
                overlap = iou(box, track.bbox)
                if overlap >= 0.15 or distance < 0.45:
                    scores.append((overlap + max(0, 1 - distance) * 0.4, index, track.id))
            scores.sort(reverse=True)
            # Ambiguous crossing: prefer a new ID to mixing two people's action history.
            if scores and (len(scores) == 1 or scores[0][0] - scores[1][0] >= 0.12):
                candidates.append(scores[0])
        assignments, used = {}, set()
        for _, index, track_id in sorted(candidates, reverse=True):
            if track_id not in used:
                assignments[index] = track_id
                used.add(track_id)
        output = []
        current_tracks = {}
        for index, person in enumerate(persons):
            if index in assignments:
                track = self.tracks[assignments[index]]
            else:
                self.sequence += 1
                track = Track(self.sequence, person['bbox'], now)
            actions, reliable = track.actions(person, width, height, now)
            track.bbox, track.last_seen = person['bbox'], now
            current_tracks[track.id] = track
            output.append({**person, 'id': track.id, 'actions': actions, 'pose_reliable': reliable})
        self.tracks = current_tracks
        self.previous_frame_time = now
        return output

    previous_frame_time = None
