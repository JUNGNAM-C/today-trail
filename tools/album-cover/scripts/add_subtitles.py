# 릴스 영상에 자막을 넣는다. 배경이 밝으면 남색·어두우면 크림색 글자로 자동 전환한다.
# 자막 파일 형식(두 가지):
#   ① 한 줄에 한 문장 → 영상 길이에 맞춰 균등하게 나눔(긴 문장은 쉼표에서 두 줄로)
#   ② 문장 줄(들) 다음에 "5초"처럼 시간 줄 → 적힌 시간만큼 보임, 문장 줄바꿈도 적힌 그대로
# 끝 제목: --title-gothic/--title-myeongjo 글꼴을 주면 자막이 끝난 뒤 남은 시간 동안 "오늘의(고딕) 트레일(명조)"이 나타났다 사라진다.
# 사용: python add_subtitles.py <입력.mp4> <자막.txt> <출력.mp4> [--cy 382] [--title-gothic 고딕.otf --title-myeongjo 명조.otf]
import sys, os, re, argparse, subprocess, tempfile, shutil, cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
ap = argparse.ArgumentParser()
ap.add_argument('src'); ap.add_argument('txt'); ap.add_argument('dst')
ap.add_argument('--cy', type=int, default=382)             # 자막 가운데 높이(액자 위 한 줄 띄움)
ap.add_argument('--title-gothic'); ap.add_argument('--title-myeongjo')
args = ap.parse_args()
SRC, TXT, DST = args.src, args.txt, args.dst
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')
FONT = ImageFont.truetype(os.path.join(ROOT, 'fonts', 'RIDIBatang.woff2'), 46)
LINE_H, MAX_W = 68, 880                     # 줄 간격, 한 줄 최대 폭
CY = args.cy
TITLE = bool(args.title_gothic and args.title_myeongjo)
FADE = 0.25
NAVY, CREAM = np.array([36, 52, 78], np.float32), np.array([245, 238, 222], np.float32)

cap = cv2.VideoCapture(SRC); fps = cap.get(cv2.CAP_PROP_FPS); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
dur = n / fps

# 자막 읽기: (줄 목록, 시간) 묶음
raw = [l.rstrip() for l in open(TXT, encoding='utf-8')]
TIME = re.compile(r'^\s*(\d+(?:\.\d+)?)\s*초\s*$')
blocks, cur = [], []
for l in raw:
    m = TIME.match(l)
    if m:
        if cur: blocks.append((cur, float(m.group(1)))); cur = []
    elif l.strip(): cur.append(l.strip())
if blocks:
    if cur: blocks.append((cur, None))
    timed = True
else:                                       # 시간 표시가 없으면 한 줄 = 한 문장, 균등 분배
    blocks = [([l.strip()], None) for l in raw if l.strip()]; timed = False
if not timed or any(b[1] is None for b in blocks):
    known = sum(b[1] for b in blocks if b[1]); rest = [b for b in blocks if b[1] is None]
    share = max(dur - known, 0) / max(len(rest), 1)
    blocks = [(ls, t if t else share) for ls, t in blocks]
starts = np.cumsum([0] + [t for _, t in blocks])
print(f'{n} frames, {dur:.2f}s video, subtitles {starts[-1]:.2f}s:', ', '.join(f'{t:g}s' for _, t in blocks))

def wrap(text):
    d = ImageDraw.Draw(Image.new('L', (10, 10)))
    if d.textlength(text, font=FONT) > MAX_W:              # 두 줄이면 가운데에 가까운 쉼표에서 끊는다
        cuts = [i + 1 for i, c in enumerate(text) if c == ',']
        cuts = [c for c in cuts if d.textlength(text[:c], font=FONT) <= MAX_W and d.textlength(text[c:].strip(), font=FONT) <= MAX_W]
        if cuts:
            c = min(cuts, key=lambda c: abs(c - len(text) / 2))
            return [text[:c].strip(), text[c:].strip()]
    out, cur = [], ''
    for w in text.split(' '):
        t = (cur + ' ' + w).strip()
        if d.textlength(t, font=FONT) <= MAX_W or not cur: cur = t
        else: out.append(cur); cur = w
    out.append(cur)
    return out
def text_mask(given, W, H):
    m = Image.new('L', (W, H), 0); d = ImageDraw.Draw(m)
    ls = [w for g in given for w in wrap(g)] if len(given) > 1 else wrap(given[0])   # 적힌 줄바꿈을 따르고, 넘치는 줄만 다시 나눔
    y0 = CY - LINE_H * len(ls) / 2
    for k, l in enumerate(ls):
        w = d.textlength(l, font=FONT)
        d.text(((W - w) / 2, y0 + k * LINE_H + (LINE_H - 46) / 2 - 4), l, font=FONT, fill=255)
    return np.array(m).astype(np.float32) / 255.0
def title_mask(W, H):
    m = Image.new('L', (W, H), 0); d = ImageDraw.Draw(m)
    fg, fm = ImageFont.truetype(args.title_gothic, 64), ImageFont.truetype(args.title_myeongjo, 64)
    a, sp, b = '오늘의', ' ', '트레일'
    wa, ws, wb = d.textlength(a, font=fg), d.textlength(sp, font=fg) * 1.2, d.textlength(b, font=fm)
    x = (W - (wa + ws + wb)) / 2; y = CY - 40
    d.text((x, y), a, font=fg, fill=255); d.text((x + wa + ws, y), b, font=fm, fill=255)
    return np.array(m).astype(np.float32) / 255.0
T_END = starts[-1]                                         # 자막이 끝나는 시각 = 제목이 나타나는 시각
MASKS = None; TMASK = None

tmp = tempfile.mkdtemp(prefix='subs_')
try:
    for i in range(n):
        ok, f = cap.read()
        if not ok: break
        H, W = f.shape[:2]
        if MASKS is None:
            MASKS = [text_mask(ls, W, H) for ls, _ in blocks]
            if TITLE: TMASK = title_mask(W, H)
        t = i / fps
        if TITLE and t >= T_END:                                      # 끝 제목: 0.5초에 걸쳐 나타나 머물다 0.6초에 걸쳐 사라짐
            a = min(1.0, (t - T_END) / 0.5, (dur - t) / 0.6)
            m = TMASK * max(a, 0.0)
        else:
            k = min(int(np.searchsorted(starts, t, side='right')) - 1, len(blocks) - 1)
            lt, seg = t - starts[k], blocks[k][1]
            a = min(1.0, lt / FADE, (seg - lt) / FADE)                # 문장마다 서서히 나타났다 사라짐
            if k == len(blocks) - 1 and not TITLE: a = min(1.0, lt / FADE)   # 끝 제목이 없으면 마지막 문장은 끝까지 남김
            m = MASKS[k] * max(a, 0.0)
        rgb = f[..., ::-1].astype(np.float32)
        # 자막 뒤 배경(글자 없는 가장자리)과의 대비가 더 큰 쪽 글자색을 고르고, 바뀔 때는 몇 프레임에 걸쳐 부드럽게
        bg = rgb[CY - 80:CY + 80, 30:90].reshape(-1, 3).mean(0)
        def rl(c): c = (c / 255.0) ** 2.2; return float(c @ [0.2126, 0.7152, 0.0722])
        lb = rl(bg); cn = (lb + 0.05) / (rl(NAVY) + 0.05); cc = (rl(CREAM) + 0.05) / (lb + 0.05)
        target = 1.0 if cn > cc else 0.0
        S_ = target if i == 0 else S_ + (target - S_) * 0.2
        col = CREAM * (1 - S_) + NAVY * S_
        out = rgb * (1 - m[..., None]) + col * m[..., None]
        cv2.imwrite(os.path.join(tmp, f'f_{i:04d}.jpg'), np.clip(out, 0, 255).astype(np.uint8)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 93])
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-framerate', f'{fps:g}', '-i', os.path.join(tmp, 'f_%04d.jpg'),
                    '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', DST], check=True)
finally:
    shutil.rmtree(tmp, ignore_errors=True)
print('saved', DST)
