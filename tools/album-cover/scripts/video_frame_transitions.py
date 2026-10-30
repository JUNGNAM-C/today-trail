# 앨범 커버 3장 사이 액자 그림 전환(1080x1920 릴스): 구름 덮고 걷기 / 해 고리 퍼지기 / 인물 확대·축소
# 타이밍 근거: ../TRANSITIONS.md
# 사용: python video_frame_transitions.py <커버1.png> <커버2.png> <커버3.png> <출력폴더> [sheet|frames]
import sys, os, math, cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
C1, C2, C3, OUT = sys.argv[1:5]
MODE = sys.argv[5] if len(sys.argv) > 5 else 'sheet'
os.makedirs(OUT, exist_ok=True)
FONTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fonts')
FPS = 30; W, H = 1080, 1920; S = 840; FX, FY = 120, 540
def hx(h): return np.array([int(h[i:i+2], 16) for i in (1, 3, 5)], np.float32)
def lerp(a, b, t): return a + (b - a) * t
def clamp01(x): return max(0.0, min(1.0, x))
def smooth(x): x = clamp01(x); return x * x * (3 - 2 * x)

# ---------- 커버 불러오기: 액자 그림, 위·아래 띠 색, 띠 경계, 인물 마스크 ----------
class Cover:
    def __init__(s, path):
        im = np.array(Image.open(path).convert('RGB')).astype(np.float32)
        s.pic = im[190:1030, 120:960].copy()
        s.top, s.bot = im[60, 20].copy(), im[1300, 20].copy()
        col = im[190:1030, 20]
        s.split = int(np.argmax(np.abs(col - s.bot).sum(1) < np.abs(col - s.top).sum(1)))
        lum = s.pic @ np.array([0.299, 0.587, 0.114], np.float32)
        n, lab, st, _ = cv2.connectedComponentsWithStats((lum < 45).astype(np.uint8))
        s.mask = (lab == 1 + np.argmax(st[1:, 4])).astype(np.float32)
        s.rcol = s.pic[s.mask > 0].mean(0)
        dist = cv2.distanceTransform(s.mask.astype(np.uint8), cv2.DIST_L2, 5)
        y, x = np.unravel_index(np.argmax(dist), dist.shape)
        s.core, s.thick = (float(x), float(y)), float(dist.max())
        s.msoft = cv2.GaussianBlur(s.mask, (0, 0), 1.2)
        # 실루엣 바깥 붓터치가 사람 색을 집어 오지 않도록, 사람을 지운 그림
        s.edist = cv2.distanceTransform(s.mask.astype(np.uint8), cv2.DIST_L2, 5)
        # 하늘 층이 가로 띠라서, 가린 곳은 같은 줄의 왼쪽·오른쪽 색을 가로로 이어 메운다(큰 실루엣도 띠가 그대로 이어짐)
        hole = cv2.dilate(s.mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        s.norun = s.pic.copy(); xs = np.arange(S)
        for y in range(S):
            hr = hole[y]
            if hr.any() and (~hr).any():
                for ch in range(3): s.norun[y, hr, ch] = np.interp(xs[hr], xs[~hr], s.pic[y, ~hr, ch])
        hp = s.pic - cv2.GaussianBlur(s.pic, (0, 0), 5)               # 이어 메운 곳에도 잔 붓결을 옆에서 빌려 온다
        hp[cv2.dilate(hole.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0] = 0   # 실루엣 윤곽선은 빌려 오지 않게
        det = np.zeros_like(hp)
        for sh in (60, -60, 140, -140):                                 # 여러 방향에서 빌려 와 빈 곳이 없게
            r_ = np.roll(hp, sh, axis=1); det = np.where(np.abs(det).sum(2, keepdims=True) > 0, det, r_)
        s.norun[hole] += det[hole] * 0.8
A, B, Cc = Cover(C1), Cover(C2), Cover(C3)

# ---------- 바깥 배경 + 글자 ----------
txt = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(txt)
def spaced(text, y, size, fill, fontpath, fit=None, gap=0):
    f = ImageFont.truetype(fontpath, size); ws = [d.textlength(c, font=f) for c in text]
    if fit: gap = (fit - sum(ws)) / (len(text) - 1)
    x = (W - (sum(ws) + gap * (len(text) - 1))) / 2
    for c, w in zip(text, ws): d.text((x, y), c, font=f, fill=fill); x += w + gap
spaced('KILIAN JORNET', FY - 100, 44, (245, 238, 222, 255), os.path.join(FONTS, 'WantedSans-Bold.ttf'), fit=840)
spaced('RICH ROLL PODCAST', 1690, 26, (36, 52, 78, 255), os.path.join(FONTS, 'WantedSans-Medium.ttf'), gap=12)
txt = np.array(txt).astype(np.float32); TA = txt[..., 3:] / 255.0

def compose(pic, top, bot, split):
    cv = np.empty((H, W, 3), np.float32)
    sy = int(round(FY + split)); cv[:sy] = top; cv[sy:] = bot
    cv[FY:FY + S, FX:FX + S] = pic
    cv = cv * (1 - TA) + txt[..., :3] * TA
    return np.clip(cv, 0, 255).astype(np.uint8)
def bg_mix(a, b, t): return lerp(a.top, b.top, t), lerp(a.bot, b.bot, t), lerp(a.split, b.split, t)

# ---------- 1) 구름으로 덮고 걷기 ----------
# 구름 둑: 위가 뭉게뭉게한 크림색 덩어리. 시드 3개를 3프레임마다 바꿔 끓듯이(손그림 느낌) 보이게.
CL = [hx('#8E8193'), hx('#A897A6'), hx('#C6AFB3'), hx('#E2C6BC'), hx('#F3D8C6'), hx('#FCECDD')]
BANK_H = 1700; EDGE = 260; BW = S + 1400; BX = 900   # 둑을 액자보다 넓게: 대각선으로 빠질 때 둑 끝이 보이지 않도록
def make_bank(seed):
    r = np.random.RandomState(seed)
    img = np.tile(CL[3], (BANK_H, BW, 1)); al = np.zeros((BANK_H, BW), np.float32)
    al[EDGE + 60:] = 1
    pts = [(r.randint(EDGE - 40, BANK_H), r.randint(-80, BW + 80)) for _ in range(2400)]
    pts.sort()
    for y, x in pts:
        w = int(r.uniform(60, 150)); h = int(w * r.uniform(0.42, 0.6)); k = r.randint(1, 4)
        cv2.ellipse(al, (x, y), (w, h), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
        cv2.ellipse(img, (x, y), (w, h), 0, 0, 360, [float(c) for c in CL[k]], -1, cv2.LINE_AA)
        for j in range(r.randint(2, 4)):            # 위로 솟은 봉우리, 윗머리만 밝게
            bx = int(x + (j - 1) * w * 0.55 + r.uniform(-12, 12)); rr = int(w * r.uniform(0.35, 0.5)); by = y - h // 2
            cv2.ellipse(al, (bx, by), (rr, int(rr * 0.8)), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
            cv2.ellipse(img, (bx, by), (rr, int(rr * 0.8)), 0, 0, 360, [float(c) for c in CL[k]], -1, cv2.LINE_AA)
            cv2.ellipse(img, (bx - rr // 6, by - rr // 5), (int(rr * 0.82), int(rr * 0.5)), 0, 180, 360, [float(c) for c in CL[min(k + 2, 5)]], -1, cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (0, 0), 1.0); al = cv2.GaussianBlur(al, (0, 0), 1.2)
    return img, np.clip(al, 0, 1)
BANKS = [make_bank(s) for s in (3, 5, 8)]
BRK = cv2.GaussianBlur(np.random.RandomState(4).randn(BANK_H, BW).astype(np.float32), (0, 0), 38)
BRK = (BRK - BRK.min()) / (BRK.max() - BRK.min())        # 걷힐 때 구멍이 먼저 뚫리는 순서
def shift(img, dx, dy, fill):
    M = np.float32([[1, 0, dx - BX], [0, 1, dy]])
    return cv2.warpAffine(img, M, (S, S), borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
T_COVER, T_HOLD, T_REVEAL = 1.4, 0.8, 1.2
CLOUD_LEN = T_COVER + T_HOLD + T_REVEAL
def cloud_frame(t, a, b):
    f = int(t * FPS); bank, bal = BANKS[(f // 3) % 3]
    if t < T_COVER:                                   # 덮기: 감속(아래에서 빠르게 솟다가 느려짐)
        e = 1 - (1 - t / T_COVER) ** 2.4
        dy = lerp(S + 40, -EDGE - 120, e)
        L = shift(bank, 0, dy, (0, 0, 0)); M = shift(bal, 0, dy, 0)
        pic = a.pic; mix = 0.0
    elif t < T_COVER + T_HOLD:                        # 덮인 채 머물기(그동안 바깥 띠가 다음 그림 색으로)
        u = (t - T_COVER) / T_HOLD
        dy = -EDGE - 120 - u * 40
        L = shift(bank, 0, dy, (0, 0, 0)); M = shift(bal, 0, dy, 0)
        pic = b.pic; mix = smooth(u)
    else:                                             # 걷기: 가속(느리게 시작해 끝에서 탁 열림), 대각선으로 빠짐
        u = (t - T_COVER - T_HOLD) / T_REVEAL; e = u ** 2.4
        dx, dy = e * 520, -EDGE - 160 - e * 700
        L = shift(bank, dx, dy, (0, 0, 0)); M = shift(bal, dx, dy, 0)
        thr = shift(BRK, dx, dy, 1.0)
        M = M * np.clip((thr - (e * 1.15 - 0.12)) / 0.06, 0, 1)
        pic = b.pic; mix = 1.0
    pic = pic * (1 - M[..., None]) + L * M[..., None]
    return compose(pic, *bg_mix(a, b, mix))

# ---------- 2) 해에서 고리가 퍼지며 다음 그림이 열림 ----------
SUN = (780 * S / 1000, 544 * S / 1000); SUNR = 46 * S / 1000
yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
RD = np.hypot(xx - SUN[0], yy - SUN[1]); TH = np.arctan2(yy - SUN[1], xx - SUN[0])
RMAX = float(RD.max()) + 30
T_SWELL, T_RING, T_RHOLD = 0.35, 1.05, 0.5
RING_LEN = T_SWELL + T_RING + T_RHOLD
RINGC = [hx('#FFF3DC'), hx('#F9D2A6'), hx('#F6B98A')]
def ring_frame(t, a, b):
    t = math.floor(t * FPS / 2) * 2 / FPS                 # 고리는 2프레임마다 갱신(손그림 간격)
    pic = a.pic.copy(); mix = 0.0
    if t < T_SWELL:                                         # 해가 한 번 부풀며 밝아짐
        u = t / T_SWELL; r0 = SUNR * (1 + 0.45 * math.sin(math.pi * u * 0.5))
        sunm = np.clip((r0 - RD) / 1.5, 0, 1)
        pic = pic * (1 - sunm[..., None]) + RINGC[0] * sunm[..., None]
        halo = np.clip((r0 * 1.6 - RD) / 2, 0, 1) * (1 - sunm) * 0.55 * u
        pic = pic * (1 - halo[..., None]) + RINGC[1] * halo[..., None]
    else:                                                   # 고리: 처음 0.4초에 빠르게, 그 뒤 느려지며 끝까지(물결 곡선)
        u = clamp01((t - T_SWELL) / T_RING); p = 1 - (1 - u) ** 3
        core = lerp(SUNR * 1.45, RMAX, p); mix = smooth(p)
        inside = np.clip((core - RD) / 1.5, 0, 1)
        pic = pic * (1 - inside[..., None]) + b.pic * inside[..., None]
        edge = np.clip(1 - np.abs(RD - core - 7) / 7, 0, 1)            # 열리는 가장자리의 밝은 빛
        pic = pic * (1 - edge[..., None]) + RINGC[0] * edge[..., None]
        for k in range(3):                                   # 앞서 퍼지는 고리 3겹: 바깥일수록 가늘고 끊김
            uk = clamp01((t - T_SWELL - 0.1 * (k + 1)) / T_RING); pk = 1 - (1 - uk) ** 3
            rk = core + (k + 1) * lerp(34, 70, pk) * (1 - 0.3 * k)
            wk = (7 - 2 * k) * (1 - 0.55 * pk) + 0.8
            brk = np.sin(TH * (5 + 3 * k) + k * 1.7 + pk * 3) * 0.5 + np.sin(TH * (11 + k) - k) * 0.5
            vis = (brk > lerp(-1.2, 0.6, pk) * (0.5 + 0.25 * k)).astype(np.float32) * (1 - pk * 0.6)
            rm = np.clip(1 - np.abs(RD - rk) / wk, 0, 1) * vis * (1 - inside)
            pic = pic * (1 - rm[..., None]) + RINGC[k] * rm[..., None]
    return compose(pic, *bg_mix(a, b, mix))

# ---------- 3) 인물 확대 → 실루엣 색으로 가득 → 다음 그림 인물에서 축소 ----------
T_IN, T_DARK, T_OUT = 1.1, 0.25, 1.1
ZOOM_LEN = T_IN + T_DARK + T_OUT
def zoom_view(cv, z, c):
    cx = c[0] + (xx - S / 2) / z; cy = c[1] + (yy - S / 2) / z
    bg = cv2.remap(cv.pic, cx, cy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    m = cv2.remap(cv.msoft, cx, cy, cv2.INTER_LINEAR, borderValue=0)
    m = np.clip((m - 0.5) * 2.2 * min(z, 6) + 0.5, 0, 1)                 # 확대해도 윤곽은 또렷하게
    return bg * (1 - m[..., None]) + cv.rcol * m[..., None]
def zmax(cv): return S / (cv.thick * 2) * 1.6

# 확대할수록 붓터치가 드러나게: 그림 좌표에 고정된 붓터치 격자를 확대 배율에 맞는 촘촘함(2배 단계)으로 골라
# 화면에서 늘 25~50px 크기로 보이게 그린다. 사람 안쪽 붓터치는 실루엣 안에만, 바깥 붓터치는 바깥에만.
def h01(i, j, s): return np.mod(np.sin(i * 12.9898 + j * 78.233 + s * 37.719) * 43758.5453, 1.0)

# 사람이 붓터치로 흩어지거나(sc: 0→1) 붓터치가 모여 사람이 되는(sc: 1→0) 확대 화면.
# 흩어질 때는 실루엣 가장자리 붓터치부터 떨어져 나가 바깥으로 흘러가며 옅어지고, 모일 때는 그 반대로 가운데부터 채워진다.
def zoom_scatter(cv, z, c, sc, gather=False):
    cx = c[0] + (xx - S / 2) / z; cy = c[1] + (yy - S / 2) / z
    base = cv2.remap(cv.norun, cx, cy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)   # 사람 없는 배경
    m = cv2.remap(cv.msoft, cx, cy, cv2.INTER_LINEAR, borderValue=0)
    m = np.clip((m - 0.5) * 2.2 * min(z, 6) + 0.5, 0, 1)
    k = 2 ** max(0, int(math.floor(math.log2(max(z, 1) / 1.5))))
    dw, dh = DW / k, DH / k
    x0, x1 = c[0] - S / (2 * z) - 3 * dw, c[0] + S / (2 * z) + 3 * dw
    y0, y1 = c[1] - S / (2 * z) - 3 * dh, c[1] + S / (2 * z) + 3 * dh
    J = np.arange(int(math.floor(y0 / dh)), int(math.ceil(y1 / dh)) + 1)
    I = np.arange(int(math.floor(x0 / dw)) - 1, int(math.ceil(x1 / dw)) + 1)
    JJ, II = np.meshgrid(J, I, indexing='ij')
    sx_ = (II + 0.5 * (JJ % 2) + (h01(II, JJ, k) - 0.5) * 0.5) * dw
    sy_ = (JJ + (h01(II, JJ, k + 7) - 0.5) * 0.4) * dh
    px, py = (sx_ - c[0]) * z + S / 2, (sy_ - c[1]) * z + S / 2
    sxi, syi = np.clip(sx_, 0, S - 1).astype(int), np.clip(sy_, 0, S - 1).astype(int)
    var = (h01(II, JJ, k + 3) - 0.5)[..., None]
    isr = cv.msoft[syi, sxi] > 0.5
    col = cv.norun[syi, sxi] * (1 + var * 0.12)
    rc = cv.rcol + var * np.array([26, 22, 34], np.float32) + 6
    col = np.where(isr[..., None], rc, col)
    ang = np.where(isr, (h01(II, JJ, k + 5) - 0.5) * 70, (h01(II, JJ, k + 9) - 0.5) * 16)
    # 흩어지는 순서: 가장자리(얇은 곳)일수록 먼저, 같은 깊이 안에서는 무작위
    depth = np.clip(cv.edist[syi, sxi] / max(cv.thick, 1), 0, 1)
    th = np.clip(depth * 0.7 + h01(II, JJ, k + 11) * 0.3, 0, 1)
    p = np.clip((sc - th * 0.75) / 0.25, 0, 1) * isr          # 이 붓터치가 떠난 정도(0=제자리, 1=다 떠남)
    if gather: p = p ** 0.8
    dirx = np.cos(np.radians(ang + (h01(II, JJ, k + 13) - 0.5) * 120)); diry = np.sin(np.radians(ang + (h01(II, JJ, k + 13) - 0.5) * 120)) - 0.4
    fly = (40 + 90 * h01(II, JJ, k + 15)) * (p ** 1.5) * (1 if not gather else -1)
    qx, qy = px + dirx * fly, py + diry * fly
    ax = (max(int(dw * z * 0.62 * 1.2), 2), max(int(dh * z * 0.62 * 1.3), 1))
    # 아직 제자리인 사람 붓터치가 덮는 영역 = 남아 있는 실루엣
    present = np.zeros((S, S), np.float32)
    ovb = base.copy(); ovr = base.copy()
    al = clamp01((z - 1.25) / 2.2) * 0.9
    for (x_, y_, qx_, qy_, cl, an, r_, pp) in zip(px.ravel(), py.ravel(), qx.ravel(), qy.ravel(), col.reshape(-1, 3), ang.ravel(), isr.ravel(), p.ravel()):
        if x_ < -3 * ax[0] or x_ > S + 3 * ax[0] or y_ < -3 * ax[0] or y_ > S + 3 * ax[0]: continue
        if not r_:
            if al > 0.02: cv2.ellipse(ovb, (int(x_), int(y_)), ax, float(an), 0, 360, [float(v) for v in cl], -1, cv2.LINE_AA)
        elif pp < 0.02:
            cv2.ellipse(present, (int(x_), int(y_)), (int(ax[0] * 1.15), int(ax[1] * 1.6)), float(an), 0, 360, 1.0, -1, cv2.LINE_AA)
            cv2.ellipse(ovr, (int(x_), int(y_)), ax, float(an), 0, 360, [float(v) for v in cl], -1, cv2.LINE_AA)
    img = base * (1 - al) + ovb * al                            # 배경: 가까이 갈수록 붓터치
    mr = m * np.clip(present, 0, 1)                             # 남은 실루엣(제자리 붓터치가 있는 곳만)
    img = img * (1 - mr[..., None]) + ovr * mr[..., None]
    # 날아가는(또는 날아오는) 붓터치는 실루엣에 갇히지 않고 맨 위에, 떠날수록 옅게
    for lv in range(3):
        ov = img.copy(); mm = np.zeros((S, S), np.float32)
        for (qx_, qy_, cl, an, r_, pp) in zip(qx.ravel(), qy.ravel(), col.reshape(-1, 3), ang.ravel(), isr.ravel(), p.ravel()):
            if not r_ or pp < 0.02 or pp > 0.98 or min(int((1 - pp) * 3), 2) != lv: continue
            sh = 1 - 0.4 * pp
            cv2.ellipse(ov, (int(qx_), int(qy_)), (max(int(ax[0] * sh), 1), max(int(ax[1] * sh), 1)), float(an), 0, 360, [float(v) for v in cl], -1, cv2.LINE_AA)
            cv2.ellipse(mm, (int(qx_), int(qy_)), (max(int(ax[0] * sh), 1), max(int(ax[1] * sh), 1)), float(an), 0, 360, 1.0, -1, cv2.LINE_AA)
        a_ = mm * (lv + 0.5) / 3
        img = img * (1 - a_[..., None]) + ov * a_[..., None]
    return img
# 붓터치 칸(화면 기준, 큼직하게): 가운데 장면에서 붓터치 하나씩 다음 그림 색으로 바뀌는 순서
_g = np.random.RandomState(21)
_wx = cv2.GaussianBlur(_g.randn(S, S).astype(np.float32), (0, 0), 4); _wx /= _wx.std()
_wy = cv2.GaussianBlur(_g.randn(S, S).astype(np.float32), (0, 0), 4); _wy /= _wy.std()
_qy = np.clip(yy + _wy * 5, 0, S - 1).astype(int); _qx = (xx + _wx * 12 + ((_qy // 26) % 2) * 34).astype(int)
SCELL = h01(_qx // 68, _qy // 26, 3.0).astype(np.float32)

def stroke_overlay(cv, z, c, base, m, recolor=None):
    k = 2 ** max(0, int(math.floor(math.log2(max(z, 1) / 1.5))))   # 붓터치 격자 단계
    dw, dh = DW / k, DH / k
    x0, x1 = c[0] - S / (2 * z) - dw, c[0] + S / (2 * z) + dw
    y0, y1 = c[1] - S / (2 * z) - dh, c[1] + S / (2 * z) + dh
    J = np.arange(int(math.floor(y0 / dh)), int(math.ceil(y1 / dh)) + 1)
    I = np.arange(int(math.floor(x0 / dw)) - 1, int(math.ceil(x1 / dw)) + 1)
    JJ, II = np.meshgrid(J, I, indexing='ij')
    sx_ = (II + 0.5 * (JJ % 2) + (h01(II, JJ, k) - 0.5) * 0.5) * dw          # 그림 좌표(엇갈린 벽돌 + 흔들림)
    sy_ = (JJ + (h01(II, JJ, k + 7) - 0.5) * 0.4) * dh
    px, py = (sx_ - c[0]) * z + S / 2, (sy_ - c[1]) * z + S / 2               # 화면 좌표
    sxi, syi = np.clip(sx_, 0, S - 1).astype(int), np.clip(sy_, 0, S - 1).astype(int)
    col = cv.norun[syi, sxi].copy(); isr = cv.msoft[syi, sxi] > 0.5
    var = (h01(II, JJ, k + 3) - 0.5)[..., None]
    col[isr] = cv.rcol + var[isr] * np.array([26, 22, 34], np.float32) + 6   # 사람 안쪽: 같은 계열 색을 조금씩 다르게
    col[~isr] = col[~isr] * (1 + var[~isr] * 0.12)
    ang = np.where(isr, (h01(II, JJ, k + 5) - 0.5) * 70, (h01(II, JJ, k + 9) - 0.5) * 16)
    if recolor is not None:                                  # 꽉 찬 순간: 붓터치가 하나씩 다음 인물 색으로
        sw = recolor(h01(II, JJ, 99))
        col[isr] = col[isr] * (1 - sw[isr][:, None]) + (recolor.to + var[isr] * np.array([26, 22, 34], np.float32) + 6) * sw[isr][:, None]
        px = px + np.cos(np.radians(ang)) * 18 * sw; py = py + np.sin(np.radians(ang)) * 18 * sw
    ax = (max(int(dw * z * 0.62 * 1.2), 2), max(int(dh * z * 0.62 * 1.3), 1))
    ovr, ovo = base.copy(), base.copy()
    for (pxi, pyi, cl, an, r_) in zip(px.ravel(), py.ravel(), col.reshape(-1, 3), ang.ravel(), isr.ravel()):
        if pxi < -ax[0] or pxi > S + ax[0] or pyi < -ax[0] or pyi > S + ax[0]: continue
        cv2.ellipse(ovr if r_ else ovo, (int(pxi), int(pyi)), ax, float(an), 0, 360, [float(v) for v in cl], -1, cv2.LINE_AA)
    return ovo * (1 - m[..., None]) + ovr * m[..., None]
def zoom_view_strokes(cv, z, c, recolor=None):
    cx = c[0] + (xx - S / 2) / z; cy = c[1] + (yy - S / 2) / z
    bg = cv2.remap(cv.pic, cx, cy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    m = cv2.remap(cv.msoft, cx, cy, cv2.INTER_LINEAR, borderValue=0)
    m = np.clip((m - 0.5) * 2.2 * min(z, 6) + 0.5, 0, 1)
    flat = bg * (1 - m[..., None]) + cv.rcol * m[..., None]
    al = clamp01((z - 1.25) / 2.2) * 0.9                     # 배율 1일 땐 원래 그림, 가까이 갈수록 붓터치
    if al <= 0.02 and recolor is None: return flat
    st = stroke_overlay(cv, z, c, flat, m, recolor)
    return flat * (1 - al) + st * al if recolor is None else st
T_IN, T_MID, T_OUT = 1.3, 0.45, 1.3
ZOOM_LEN = T_IN + T_MID + T_OUT
ZM = 5.0                                                   # 사람이 다 흩어진 뒤 머무는 배율(배경 붓터치가 크게 보이는 정도)
def zoom_frame(t, a, b):
    t = math.floor(t * FPS / 2) * 2 / FPS                   # 2프레임마다 갱신(붓터치가 끓듯이)
    if t < T_IN:                                            # 확대하며 사람이 가장자리부터 붓터치로 흩어짐
        u = t / T_IN; e = u ** 1.8
        z = math.exp(math.log(ZM) * e); k = smooth(min(1, u * 1.3))
        c = (lerp(S / 2, a.core[0], k), lerp(S / 2, a.core[1], k))
        sc = smooth((u - 0.3) / 0.7)
        return compose(zoom_scatter(a, z, c, sc), *bg_mix(a, b, 0.0))
    if t < T_IN + T_MID:                                    # 붓터치만 남은 화면이 칸마다 하나씩 다음 그림 색으로
        u = (t - T_IN) / T_MID
        fa = zoom_scatter(a, ZM, a.core, 1.0); fb = zoom_scatter(b, ZM, b.core, 1.0, gather=True)
        sw = cv2.GaussianBlur(np.clip((u * 1.45 - SCELL * 0.45 - 0.1) / 0.12, 0, 1), (0, 0), 1.5)[..., None]
        return compose(fa * (1 - sw) + fb * sw, *bg_mix(a, b, smooth(u)))
    u = (t - T_IN - T_MID) / T_OUT; e = 1 - (1 - u) ** 2.0  # 붓터치가 모여 다음 사람이 되며 축소
    z = math.exp(math.log(ZM) * (1 - e)); k = smooth(min(1, u * 1.3))
    c = (lerp(b.core[0], S / 2, k), lerp(b.core[1], S / 2, k))
    sc = 1 - smooth(u / 0.65)
    return compose(zoom_scatter(b, z, c, sc, gather=True), *bg_mix(a, b, 1.0))

# ---------- 4) 붓터치로 흩어졌다 모이기 (러빙 빈센트 예고편 참고) ----------
# 개체(사람 → 땅)가 붓터치 조각으로 흩어지며 하늘 결을 따라 흘러가고, 하늘이 아래 층부터 다음 그림 색으로
# 바뀐 뒤, 다음 그림의 땅 → 사람이 붓터치로 모여 나타난다.
from scipy.ndimage import median_filter
def objects_of(cv, npz=None):
    if npz is not None:                                    # 1번: 그림을 만들 때 저장한 개체 마스크
        L = np.load(npz)
        rs = lambda m: cv2.resize(m.astype(np.uint8), (S, S), interpolation=cv2.INTER_NEAREST) > 0
        run, grd = rs(L['runner']), rs(L['rock']) | rs(L['cloud'])
    else:                                                  # 2·3번: 완성 그림에서 하늘 색에 없는 덩어리를 땅으로
        run = cv.mask > 0; pic = cv.pic; top = max(cv.split - 25, 50)
        sky = pic[:top][~run[:top]].reshape(-1, 3)
        sky = sky[np.random.RandomState(0).choice(len(sky), 20000)]
        _, _, cen = cv2.kmeans(sky, 12, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 3, cv2.KMEANS_PP_CENTERS)
        dist = np.min(np.linalg.norm(cv2.GaussianBlur(pic, (0, 0), 2)[:, :, None, :] - cen[None, None], axis=3), 2)
        G = ((dist > 24) & ~run).astype(np.uint8)
        G = cv2.morphologyEx(cv2.morphologyEx(G, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)) > 0
        ridge = np.full(S, S, np.float32); y35 = int(S * 0.35)
        for x in range(S):
            for y0 in np.where(G[y35:, x])[0]:
                if G[y35 + y0:y35 + y0 + 12, x].mean() > 0.6: ridge[x] = y35 + y0; break
        ridge = median_filter(ridge, 9, mode='nearest')
        grd = (yy >= ridge[None, :]) & ~run
    # 개체를 걷어 낸 하늘: 땅 위 하늘 색을 가로로 넓게 흐려 아래로 잇고(해 같은 점은 지워짐),
    # 그 위에 바로 위 하늘의 잔 붓결만 얹는다
    pic = cv.pic
    topy = np.where(grd.any(0), grd.argmax(0), S).astype(np.float32)
    if (topy < S).any():                                     # 땅이 안 잡힌 열은 이웃 값으로, 들쭉날쭉한 윗선은 매끈하게
        ok = topy < S; topy[~ok] = np.interp(np.where(~ok)[0], np.where(ok)[0], topy[ok])
        topy = np.minimum(topy, median_filter(topy, 41, mode='nearest'))
    topy = topy.astype(int)
    tmed = int(np.median(topy[topy < S])) if (topy < S).any() else S
    ref = np.median(pic[max(tmed - 30, 0):max(tmed - 8, 1)], 0)
    ref = cv2.GaussianBlur(ref[None], (0, 0), sigmaX=60, sigmaY=0.1)[0]
    hp = pic - cv2.GaussianBlur(pic, (0, 0), 6)
    plate = pic.copy()
    for x in range(S):
        y0 = topy[x]
        if y0 >= S: continue
        src = np.clip(y0 - 6 - (np.arange(S - y0) % 34), 0, S - 1)
        plate[y0:, x] = ref[x] + hp[src, x] * 0.8
    edge = np.clip((yy - topy[None, :] + 12) / 12, 0, 1)[..., None] * (yy >= topy[None, :] - 12)[..., None]
    plate = pic * (1 - edge) + plate * edge                  # 이음매를 살짝 섞는다
    # 사람이 하늘 속에 떠 있는 부분은 주변 하늘로 메운다
    hole = cv2.dilate(run.astype(np.uint8), np.ones((7, 7), np.uint8))
    plate = cv2.inpaint(np.clip(plate, 0, 255).astype(np.uint8), hole, 8, cv2.INPAINT_TELEA).astype(np.float32)
    return run, grd, plate
ORA = objects_of(A, os.path.join(os.path.dirname(C1), 'layers.npz')); ORB = objects_of(B)

# 붓터치 조각: 가로로 길쭉한 칸(하늘 결 방향)으로 나눈다
DW, DH = 22, 8
grs = np.random.RandomState(12)
offx = (np.arange(S) // DH % 2) * (DW // 2)                 # 벽돌처럼 엇갈리게
# 칸 경계가 네모나게 보이지 않도록 좌표를 출렁이게 해서 붓터치 모양의 칸으로
wxn = cv2.GaussianBlur(grs.randn(S, S).astype(np.float32), (0, 0), 3); wxn /= wxn.std()
wyn = cv2.GaussianBlur(grs.randn(S, S).astype(np.float32), (0, 0), 3); wyn /= wyn.std()
qy = np.clip(yy + wyn * 2.6, 0, S - 1).astype(int); qx = np.clip(xx + wxn * 6 + offx[qy], 0, S + DW).astype(int)
dab_id = (qy // DH) * (S // DW + 2) + (qx // DW)
ND = int(dab_id.max()) + 1
jit = grs.rand(ND).astype(np.float32)
def dab_mean(m): return np.bincount(dab_id.ravel(), weights=m.ravel().astype(np.float32), minlength=ND) / np.maximum(np.bincount(dab_id.ravel(), minlength=ND), 1)
cyd = dab_mean(yy) / S; cxd = dab_mean(xx) / S
runA, grdA, plA = ORA[0], ORA[1], ORA[2]; runB, grdB, plB = ORB[0], ORB[1], ORB[2]
fA_run, fA_grd = dab_mean(runA) > 0.3, dab_mean(grdA) > 0.3
fB_run, fB_grd = dab_mean(runB) > 0.3, dab_mean(grdB) > 0.3
# 단계별 시각(초): ① A 개체가 흩어짐 ② 하늘이 B 색으로 ③ B 개체가 모임
T1 = np.where(fA_run, 0.05 + 0.4 * jit, np.where(fA_grd, 0.3 + 0.35 * cyd + 0.3 * jit, 9))  # 사람 → 땅(위에서 아래로)
T2 = 0.75 + 0.95 * (1 - cyd) + 0.25 * jit + 0.1 * np.sin(cxd * 7 + cyd * 3)          # 하늘: 아래 층부터 위로
TX = np.where(fA_grd, T1, T2)                              # A 땅 자리는 흩어지는 순간 바로 B 쪽이 드러남
TR = np.where(fA_run, T1, 99)                               # A 사람 자리는 먼저 A 하늘로 메워졌다가 하늘과 함께 바뀜
# B 땅은 A 땅이 흩어진 바로 뒤에 모여 빈 하늘이 오래 보이지 않게, B 사람은 하늘이 다 바뀐 뒤에
T3 = np.where(fB_grd, np.maximum(0.75 + 0.35 * cyd + 0.3 * jit, np.where(fA_run | fA_grd, T1, T2) + 0.3), np.where(fB_run, 2.1 + 0.35 * jit, 99))
STROKE_LEN = 2.9; DUR = 0.42                              # 조각 하나가 날아가거나 내려앉는 데 걸리는 시간
FLOW = np.stack([np.cos(0.25 * np.sin(cyd * 9)), -0.18 + 0.1 * np.sin(cxd * 5)], 1)   # 하늘 결을 따라 오른쪽 위로
def stroke_frame(t, a, b):
    t = math.floor(t * FPS / 2) * 2 / FPS                   # 2프레임마다 갱신
    sx = cv2.GaussianBlur((t >= TX)[dab_id].astype(np.float32), (0, 0), 1.2)[..., None]
    s3 = cv2.GaussianBlur((t >= T3)[dab_id].astype(np.float32), (0, 0), 1.2)[..., None]
    sr = cv2.GaussianBlur((t >= TR)[dab_id].astype(np.float32), (0, 0), 1.2)[..., None]
    pic = b.pic * s3 + (plB * sx + (plA * sr + a.pic * (1 - sr)) * (1 - sx)) * (1 - s3)
    # 날아가는 조각(떠나는 색: 결을 따라 흘러가며 옅어짐)과 내려앉는 조각(오는 색: 반대쪽에서 흘러와 자리 잡음)
    # 투명도를 4단계로 묶어 단계마다 한 번에 그린다
    buckets = [[] for _ in range(4)]
    for T, src, leaving in ((TR, a.pic, True), (TX, plA, True), (T3, b.pic, False)):
        u = (t - T) / DUR
        act = np.where((u > 0) & (u < 1))[0] if leaving else np.where((u > -1) & (u < 0))[0]
        for i in act:
            if leaving: e = u[i] ** 1.6; d = 70 * e; al = 1 - u[i]; w = DW * 0.62 * (1 - 0.35 * e)
            else:       e = 1 + u[i]; d = -60 * (1 - e); al = e; w = DW * 0.62 * (0.65 + 0.35 * e)
            if al < 0.05: continue
            col = src[min(int(cyd[i] * S), S - 1), min(int(cxd[i] * S), S - 1)]
            buckets[min(int(al * 4), 3)].append((cxd[i] * S + FLOW[i, 0] * d, cyd[i] * S + FLOW[i, 1] * d, w, i, col))
    layer = pic
    for k, items in enumerate(buckets):
        if not items: continue
        ov = layer.copy(); m = np.zeros((S, S), np.float32)
        for cx, cy, w, i, col in items:
            ang = float(np.degrees(np.arctan2(FLOW[i, 1], FLOW[i, 0])))
            ax = (max(int(w), 2), max(int(DH * 0.62), 1))
            cv2.ellipse(ov, (int(cx), int(cy)), ax, ang, 0, 360, [float(c) for c in col], -1, cv2.LINE_AA)
            cv2.ellipse(m, (int(cx), int(cy)), ax, ang, 0, 360, 1.0, -1, cv2.LINE_AA)
        a_ = (k + 0.5) / 4 * m
        layer = layer * (1 - a_[..., None]) + ov * a_[..., None]
    mix = smooth((t - 0.95) / 1.4)
    return compose(layer, *bg_mix(a, b, mix))

SEQ = [('stroke', stroke_frame, STROKE_LEN, A, B), ('zoom', zoom_frame, ZOOM_LEN, B, Cc)]
# ---------- 릴스 전체(15초): 1번 머묾 → 붓터치 전환 → 2번 머묾 → 흩어졌다 모이는 확대 전환 → 3번 머묾 ----------
# 머무는 동안에도 그림이 살아 있도록 1px 남짓 흔들리는 왜곡 3장을 3프레임마다 바꿔 끼운다(손그림처럼 끓는 느낌).
_br = np.random.RandomState(31)
BOIL = []
for _ in range(3):
    bx = cv2.GaussianBlur(_br.randn(S, S).astype(np.float32), (0, 0), 6); bx = bx / bx.std() * 0.9
    by = cv2.GaussianBlur(_br.randn(S, S).astype(np.float32), (0, 0), 6); by = by / by.std() * 0.9
    BOIL.append(((xx + bx).astype(np.float32), (yy + by).astype(np.float32)))
H1, H2, H3 = 3.0, 3.0, 3.05
REEL = [(H1, lambda t: compose(A.pic, A.top, A.bot, A.split)),
        (STROKE_LEN, lambda t: stroke_frame(t, A, B)),
        (H2, lambda t: compose(B.pic, B.top, B.bot, B.split)),
        (ZOOM_LEN, lambda t: zoom_frame(t, B, Cc)),
        (H3, lambda t: compose(Cc.pic, Cc.top, Cc.bot, Cc.split))]
def reel_frame(i):
    t = i / FPS
    for L, fn in REEL:
        if t < L or (L, fn) == REEL[-1]:
            fr = fn(min(t, L - 1e-3)); break
        t -= L
    mx, my = BOIL[(i // 3) % 3]
    fr = fr.copy(); fr[FY:FY + S, FX:FX + S] = cv2.remap(fr[FY:FY + S, FX:FX + S], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return fr
REEL_LEN = sum(L for L, _ in REEL)
if MODE == 'frames':
    fd = os.path.join(OUT, 'reel_frames'); os.makedirs(fd, exist_ok=True)
    n = int(round(REEL_LEN * FPS)); start = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    for i in range(start, n):
        Image.fromarray(reel_frame(i)).save(os.path.join(fd, f'f_{i:04d}.jpg'), quality=92)
        if i % 30 == 0: print(f'{i}/{n}', flush=True)
    print('done', n, 'frames', f'{REEL_LEN:.2f}s')
if MODE == 'sheet':
    for name, fn, L, a, b in SEQ:
        ts = [L * i / 9 for i in range(10)]
        tiles = [cv2.resize(fn(min(t, L - 1e-3), a, b), (216, 384), interpolation=cv2.INTER_AREA) for t in ts]
        sheet = np.hstack([np.pad(x, ((0, 22), (2, 2), (0, 0)), constant_values=20) for x in tiles])
        im = Image.fromarray(sheet); dd = ImageDraw.Draw(im)
        for i, t in enumerate(ts): dd.text((i * 220 + 8, 388), f'{t:.2f}s', fill=(230, 230, 230))
        im.save(os.path.join(OUT, f'sheet_{name}.png')); print(name, f'{L:.2f}s')
