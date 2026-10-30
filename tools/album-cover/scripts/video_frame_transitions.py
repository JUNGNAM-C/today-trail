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
spaced('RICH ROLL PODCAST', 1690, 26, (36, 52, 78, 255), os.path.join(FONTS, 'WantedSans-Medium.ttf'), gap=12)
txt = np.array(txt).astype(np.float32); TA = txt[..., 3:] / 255.0
# 인물 이름은 액자 아래로(위쪽은 자막 자리). 아래 띠 색이 밝으면 남색, 어두우면 크림색 글자
_nm = Image.new('L', (W, H), 0); d = ImageDraw.Draw(_nm)
spaced('KILIAN JORNET', FY + S + 44, 44, 255, os.path.join(FONTS, 'WantedSans-Bold.ttf'), fit=840)
NAME_A = np.array(_nm).astype(np.float32)[..., None] / 255.0
NAVY_C, CREAM_C = np.array([36, 52, 78], np.float32), np.array([245, 238, 222], np.float32)

def compose(pic, top, bot, split):
    cv = np.empty((H, W, 3), np.float32)
    sy = int(round(FY + split)); cv[:sy] = top; cv[sy:] = bot
    cv[FY:FY + S, FX:FX + S] = pic
    cv = cv * (1 - TA) + txt[..., :3] * TA
    lum = float(np.dot(np.asarray(bot, np.float32), [0.299, 0.587, 0.114]))
    s_ = min(1.0, max(0.0, (lum - 135) / 25)); nc = CREAM_C * (1 - s_) + NAVY_C * s_
    cv = cv * (1 - NAME_A) + nc * NAME_A
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
        # 능선 위로 솟은 수풀 언덕처럼, 땅에 붙어 있는 덩어리도 땅에 넣는다
        n_, lab_ = cv2.connectedComponents(G.astype(np.uint8))
        touch = np.unique(lab_[grd & G]); touch = touch[touch > 0]
        grd |= np.isin(lab_, touch) & ~run
        grd = cv2.morphologyEx(grd.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 0
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

# ---------- 머무는 동안의 움직임 ----------
# 1번: 하늘 층이 태양에서 바깥으로 퍼지는 물결(paint_cloudsea.py SKY_LOOP로 만든 3초 고리 영상) +
#      구름바다 덩어리가 가까운 것일수록 빠르게 왼쪽으로 흘러감(원근)
# 2·3번: 하늘이 물결처럼 출렁이고, 땅(수풀·바위) 덩어리가 저마다 다른 박자로 바람에 흔들림
LOOP_DIR = os.path.join(os.path.dirname(C1), 'sky_loop')
SKY1 = [np.array(Image.open(os.path.join(LOOP_DIR, f's_{k:02d}.jpg'))).astype(np.float32) for k in range(90)] if os.path.isdir(LOOP_DIR) else None
_L1 = np.load(os.path.join(os.path.dirname(C1), 'layers.npz'))
CLOUD1 = cv2.resize(_L1['cloud'].astype(np.uint8), (S, S), interpolation=cv2.INTER_NEAREST) > 0
CW1 = cv2.GaussianBlur(CLOUD1.astype(np.float32), (0, 0), 1.0)[..., None]
_dep = np.clip((yy - yy[CLOUD1].min()) / (S - yy[CLOUD1].min()), 0, 1)
_bob = cv2.GaussianBlur(np.random.RandomState(41).randn(S, S).astype(np.float32), (0, 0), 45); _bob /= _bob.std()
def anim_A(i):
    t = i / FPS
    pic = SKY1[i % 90].copy() if SKY1 is not None else A.pic.copy()
    dx = -(2.5 + 15 * _dep ** 1.4) * t                                # 원근: 수평선 쪽은 느리게, 가까울수록 빠르게(길어진 영상에서도 가장자리가 비치지 않게 천천히)
    dy = 1.6 * np.sin(2 * np.pi * (t / 3.2) + _bob * 2.5)              # 덩어리마다 다른 박자로 살짝 오르내림
    cl = cv2.remap(pic, (xx - dx).astype(np.float32), (yy + dy).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return pic * (1 - CW1) + cl * CW1
# ---------- 1번 그림이 쌓이며 만들어지는 시작 장면: 땅 → 구름 → 해 → 하늘 → 사람 ----------
# 빈 종이 위에 층마다 움직이며 나타나 제자리에 내려앉는다.
def _rs(m, interp=cv2.INTER_NEAREST): return cv2.resize(m.astype(np.float32), (S, S), interpolation=interp)
ROCK1 = cv2.GaussianBlur(_rs(_L1['rock']), (0, 0), 0.8)
CLD1 = cv2.GaussianBlur(_rs(_L1['cloud']), (0, 0), 0.8)
RUN1 = cv2.GaussianBlur(_rs(_L1['runner']), (0, 0), 0.6)
RUNW = cv2.GaussianBlur(cv2.dilate((RUN1 > 0.05).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(np.float32), (0, 0), 1.5)   # 윤곽 번짐까지 덮는 넓은 사람 자리
FM1 = _rs(_L1['fm'], cv2.INTER_LINEAR)
SX1, SY1, SR1 = [float(v) * S / 1000 for v in _L1['sun_xy']]
SUNDISC = np.clip((SR1 - np.hypot(xx - SX1, yy - SY1)) / 1.5, 0, 1)
PAPER = hx('#EFE4D3')
_pg = cv2.GaussianBlur(np.random.RandomState(61).randn(S, S).astype(np.float32), (0, 0), 0.8)
PAPER_IMG = np.tile(PAPER, (S, S, 1)) + (_pg * 5)[..., None]
def back_out(x, s=None):                                  # 지나치거나 튕기지 않고, 멈출수록 느려지며 제자리에 섬
    return 1 - (1 - clamp01(x)) ** 4
def ease_out(x, p=3): return 1 - (1 - clamp01(x)) ** p
def warp(img, dx, dy):
    return cv2.remap(img, (xx - dx).astype(np.float32), (yy - dy).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
def over(dst, src, a): return dst * (1 - a[..., None]) + src * a[..., None]
T_ROCK, T_CLOUD, T_SUN, T_SKY, T_RUN = (0.0, 0.9), (0.6, 1.7), (1.4, 2.3), (1.9, 3.4), (3.1, 4.4)
BUILD_LEN = 4.6
# 사람 붓터치 조각: 실루엣 안을 작은 벽돌 칸(6x3px)으로 나눠 조각마다 날아올 방향·거리·도착 순서를 정한다
def make_gather(runm, seed, cw=6, ch=3, angs=(150, 260), dist=(60, 220)):
    ry, rx = np.where(runm > 0.5); gr = np.random.RandomState(seed); cells = {}
    for y_, x_ in zip(ry, rx):
        cells.setdefault((y_ // ch, (x_ + cw // 2 * ((y_ // ch) % 2)) // cw), []).append((x_, y_))
    pts = np.array([np.mean(v, 0) for v in cells.values()], np.float32)            # 조각 자리(목표)
    ed = cv2.distanceTransform((runm > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    dep = np.clip(ed[pts[:, 1].astype(int), pts[:, 0].astype(int)] / max(ed.max(), 1), 0, 1)
    th = np.clip((1 - dep) * 0.55 + gr.rand(len(pts)) * 0.45, 0, 1)               # 몸통(깊은 곳)이 먼저 도착
    ang = np.radians(gr.uniform(angs[0], angs[1], len(pts)))                         # 하늘 쪽에서 날아옴
    frm = pts + np.stack([np.cos(ang), np.sin(ang)], 1) * gr.uniform(dist[0], dist[1], (len(pts), 1))
    return dict(pts=pts, th=th, frm=frm, spin=gr.uniform(-90, 90, len(pts)), var=gr.rand(len(pts)) - 0.5, runm=runm, cw=cw, ch=ch)
GA = make_gather(RUN1, 71)
def runner_gather(pic, u, g=None, rcol=None):
    g = g or GA; rcol = A.rcol if rcol is None else rcol
    present = np.zeros((S, S), np.float32)
    fly = []
    hw, hh = int(g['cw'] * 0.85), int(g['ch'] * 1.0)
    for k in range(len(g['pts'])):
        p = clamp01((u - g['th'][k] * 0.62) / 0.38)
        if p <= 0: continue
        # 실루엣 채움은 마지막 35% 동안 서서히 스며들어, 조각이 닿는 순간 툭 바뀌지 않게
        land = smooth((p - 0.65) / 0.35)
        if land > 0:
            cv2.ellipse(present, (int(g['pts'][k, 0]), int(g['pts'][k, 1])), (hw, hh), 0, 0, 360, float(land), -1, cv2.LINE_AA)
        if p >= 1: continue
        e = 1 - (1 - p) ** 4                                # 멈출수록 느려짐(튕김 없음)
        pos = g['frm'][k] + (g['pts'][k] - g['frm'][k]) * e
        fly.append((pos, (g['spin'][k] - 15) * (1 - e), e, g['var'][k] * (1 - e), 1 - land))
    rm = g['runm'] * np.clip(present, 0, 1)                 # 도착한 조각만큼만 실루엣이 채워짐(윤곽은 또렷)
    pic = over(pic, np.tile(rcol, (S, S, 1)), rm)
    ov = pic.copy(); mm = np.zeros((S, S), np.float32)
    sw, sh = g['cw'] / 6.0, g['ch'] / 3.0
    for pos, ang, e, vr, keep in fly:                       # 날아오는 조각: 기울기·크기·색이 자리에 가까울수록 실루엣 쪽으로 서서히 맞춰짐
        col = rcol + vr * np.array([30, 26, 40], np.float32) + 8 * (1 - e)
        ax = (max(int(round((5 + 3 * (1 - e)) * sw)), 2), max(int(round((3 + 1.0 * (1 - e)) * sh)), 1))
        cv2.ellipse(ov, (int(pos[0]), int(pos[1])), ax, float(ang), 0, 360, [float(c) for c in col], -1, cv2.LINE_AA)
        cv2.ellipse(mm, (int(pos[0]), int(pos[1])), ax, float(ang), 0, 360, float(min(1.0, 0.4 + e) * keep), -1, cv2.LINE_AA)
    return over(pic, ov, np.clip(mm, 0, 1))
def build_frame(i, t):
    full = anim_A(i)                                       # 머무는 장면과 같은 움직이는 하늘·구름에서 층을 떼어 온다
    sky_src = SKY1[i % 90] if SKY1 is not None else A.pic
    sky_src = cv2.inpaint(np.clip(sky_src, 0, 255).astype(np.uint8), (RUNW > 0.05).astype(np.uint8), 5, cv2.INPAINT_TELEA).astype(np.float32)   # 사람 자리는 주변 하늘로 메움
    pic = PAPER_IMG.copy()
    # 하늘: 해 곁에서부터 고리가 하나씩 바깥으로 번져 나가며 쌓임(경계는 붓결처럼 들쭉날쭉)
    u = (t - T_SKY[0]) / (T_SKY[1] - T_SKY[0])
    if u > 0:
        R = ease_out(u, 2.2) * 1.05
        rag = cv2.GaussianBlur(np.random.RandomState(62).randn(S, S).astype(np.float32), (0, 0), 3) * 0.012
        a = np.clip((R - FM1 + rag) / 0.02, 0, 1) * (yy < A.split + 40)
        pic = over(pic, sky_src, a)
    # 해: 구름바다 뒤에서 떠오름
    u = (t - T_SUN[0]) / (T_SUN[1] - T_SUN[0])
    if u > 0:
        dy = (1 - back_out(u, 1.2)) * 70
        sun = warp(full * SUNDISC[..., None], 0, dy); sa = warp(SUNDISC, 0, dy) * clamp01(u * 3)
        glow = np.clip(1 - np.hypot(xx - SX1, yy - SY1 - dy) / (SR1 * 2.2), 0, 1) ** 2 * 0.5 * clamp01(u * 2) * (1 - clamp01((t - T_SKY[0]) / 0.6))
        pic = over(pic, np.tile(hx('#F9D2A6'), (S, S, 1)), glow)
        pic = over(pic, sun / np.maximum(sa, 1e-3)[..., None] * (sa > 0)[..., None], sa)
    # 구름바다: 오른쪽에서 밀려 들어와 제자리에(가까운 줄일수록 더 멀리서, 더 빨리)
    u = (t - T_CLOUD[0]) / (T_CLOUD[1] - T_CLOUD[0])
    if u > 0:
        e = back_out(u, 0.9)
        dxr = (1 - e) * (120 + 380 * _dep ** 1.2)
        cl = cv2.remap(full, (xx - dxr).astype(np.float32), yy.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        ca = cv2.remap(CLD1, (xx - dxr).astype(np.float32), yy.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        ca = ca * np.clip((u * 1.6 - (1 - _dep) * 0.5) / 0.4, 0, 1)    # 먼 줄은 조금 늦게 또렷해짐
        pic = over(pic, cl, ca)
    # 땅(바위): 아래에서 솟아올라 내려앉음
    u = (t - T_ROCK[0]) / (T_ROCK[1] - T_ROCK[0])
    if u > 0:
        dy = (1 - back_out(u, 1.3)) * 240
        wr = lambda m: cv2.remap(m, xx.astype(np.float32), (yy - dy).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        rm = wr(ROCK1)                                     # 지나쳐 솟을 때 아래가 비지 않게 맨 아래 줄을 늘려 씀
        pic = over(pic, wr(A.pic * ROCK1[..., None]) / np.maximum(rm, 1e-3)[..., None], rm * clamp01(u * 2.5))
    # 사람: 붓터치 조각들이 하늘 쪽에서 날아와 몸통부터 쌓이며 실루엣이 됨
    u = (t - T_RUN[0]) / (T_RUN[1] - T_RUN[0])
    if u > 0:
        pic = runner_gather(pic, u)
    # 바깥 띠: 위는 땅이 오를 때, 아래는 하늘이 번질 때 종이색에서 제 색으로
    top = lerp(PAPER * 0.92, A.top, smooth((t - T_ROCK[0]) / 0.9))
    bot = lerp(PAPER * 0.92, A.bot, smooth((t - T_SKY[0]) / 1.2))
    return compose(pic, top, bot, A.split)
if MODE == 'build':
    ts = [BUILD_LEN * k / 11 for k in range(12)]
    tiles = [cv2.resize(build_frame(int(t * FPS), t), (180, 320), interpolation=cv2.INTER_AREA) for t in ts]
    sheet = np.hstack([np.pad(x, ((0, 22), (2, 2), (0, 0)), constant_values=20) for x in tiles])
    im = Image.fromarray(sheet); dd = ImageDraw.Draw(im)
    for k, t in enumerate(ts): dd.text((k * 184 + 8, 324), f'{t:.2f}s', fill=(230, 230, 230))
    im.save(os.path.join(OUT, 'sheet_build.png'))
    fd = os.path.join(OUT, 'build_frames'); os.makedirs(fd, exist_ok=True)
    n = int(round((BUILD_LEN + 1.5) * FPS))                 # 쌓인 뒤 1.5초 머묾까지
    for i in range(n):
        t = i / FPS
        fr = build_frame(i, min(t, BUILD_LEN)) if t < BUILD_LEN else compose(anim_A(i), A.top, A.bot, A.split)
        Image.fromarray(fr).save(os.path.join(fd, f'f_{i:04d}.jpg'), quality=92)
    print('build', n, 'frames')
ORC = objects_of(Cc)

# ---------- 2·3번 하늘: 층 경계는 그대로, 빛 줄기와 붓결만 한 방향으로 흐름 ----------
# 2번은 왼쪽→오른쪽, 3번은 지평선→하늘(위). 땅 덩어리는 저마다 다른 박자로 흔들림.
xs_ = np.arange(S)
class SkyParts:
    def __init__(s, cv, ob, mode):
        s.cv, s.mode = cv, mode
        s.grd = ob[1].astype(np.float32)
        s.gw = cv2.GaussianBlur(s.grd, (0, 0), 1.5)               # 땅 윤곽은 또렷하게
        ph = cv2.GaussianBlur(np.random.RandomState(53).randn(S, S).astype(np.float32), (0, 0), 38); s.ph = ph / ph.std()
        # 하늘 결은 땅을 걷어 내고 하늘을 아래로 이어 붙인 판에서 딴다(지평선 윤곽이 결에 섞여 끌려가지 않게)
        plate = ob[2]
        s.low = cv2.GaussianBlur(plate, (0, 0), 4); s.hp = plate - s.low
        # 잔 붓결만 남기고 뚜렷한 윤곽(산등성이·사람 자리 자국)은 걸러 낸다: 큰 값은 0으로
        mag = np.abs(s.hp).max(2, keepdims=True)
        s.hp = s.hp * np.clip((9 - mag) / 4, 0, 1)
        # 하늘 바탕도 땅·사람 자리 근처는 가로로 이어진 같은 줄 색으로(가린 자리 자국이 남지 않게)
        hole = cv2.dilate((ob[0] | ob[1]).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        low = s.low.copy()
        for y in range(S):
            hr = hole[y]
            if hr.any() and (~hr).any():
                for ch in range(3): low[y, hr, ch] = np.interp(xs_[hr], xs_[~hr], s.low[y, ~hr, ch])
        s.low = np.where(hole[..., None], low, s.low)
        s.hor = float(cv.split)
SPB, SPC = SkyParts(B, ORB, 'lr'), SkyParts(Cc, ORC, 'up')
def sky_flow(P, t):
    """사람을 뺀 움직이는 그림(하늘 흐름 + 땅 흔들림)"""
    v = 20.0                                                   # 붓결이 흐르는 속도(px/s)
    if P.mode == 'lr':
        det = cv2.remap(P.hp, (xx - v * t).astype(np.float32), yy.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        phase = xx / 420 - yy / 900 - t / 3.0 + 0.1 * np.sin(yy / 45)      # 오른쪽으로 나아가는 은은한 빛 줄기(살짝 기울어짐)
        fade = 0.7
    else:
        det = cv2.remap(P.hp, xx.astype(np.float32), (yy + v * t).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        phase = (P.hor - yy) / 150 - t / 2.4 + 0.1 * np.sin(xx / 60)       # 지평선에서 위로 올라가는 빛 줄기
        fade = np.clip(np.exp(-(P.hor - yy) / 420), 0, 1)
    ring = (0.5 + 0.5 * np.sin(2 * np.pi * phase)) ** (3 if P.mode == 'lr' else 4) * fade
    sky = (P.low + det) * (1 + 0.06 * ring)[..., None] + (ring * 5)[..., None]
    gdx = 3.0 * np.sin(2 * np.pi * t / 2.6 + P.ph * 2.8); gdy = 1.3 * np.sin(2 * np.pi * t / 3.4 + P.ph * 2.1 + 1.0)
    grd = cv2.remap(P.cv.norun, (xx + gdx).astype(np.float32), (yy + gdy).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return sky * (1 - P.gw[..., None]) + grd * P.gw[..., None]
def with_runner(cv, norun_pic): return over(norun_pic, cv.pic, cv.msoft)

class _Swap:                                               # 함수가 쓰는 그림을 잠시 움직이는 그림으로 바꿔 끼움
    def __init__(s, cv, attr, val): s.cv, s.attr, s.val = cv, attr, val
    def __enter__(s): s.old = getattr(s.cv, s.attr); setattr(s.cv, s.attr, s.val)
    def __exit__(s, *a): setattr(s.cv, s.attr, s.old)

# ---------- 1번이 붓터치로 흩어져 빈 종이가 됨: 사람 → 땅·구름 → 하늘(바깥에서 해 쪽으로), 해는 마지막 ----------
OUT1 = 1.45
_fm_d = dab_mean(FM1); _fm_d = (_fm_d - _fm_d.min()) / (np.ptp(_fm_d) + 1e-6)
_sun_d = dab_mean(SUNDISC) > 0.3
TO = np.where(fA_run, 0.02 + 0.3 * jit, np.where(fA_grd, 0.2 + 0.4 * cyd + 0.2 * jit, 0.45 + 0.55 * (1 - _fm_d) + 0.12 * jit))
TO = np.where(_sun_d, 1.05 + 0.1 * jit, TO)
def paper_out(t, apic):
    t = math.floor(t * FPS / 2) * 2 / FPS
    sw = cv2.GaussianBlur((t >= TO)[dab_id].astype(np.float32), (0, 0), 1.2)[..., None]
    pic = PAPER_IMG * sw + apic * (1 - sw)
    u = (t - TO) / DUR
    act = np.where((u > 0) & (u < 1))[0]
    buckets = [[] for _ in range(4)]
    for k in act:
        e = u[k] ** 1.6; d = 80 * e; al = 1 - u[k]
        if al < 0.05: continue
        col = apic[min(int(cyd[k] * S), S - 1), min(int(cxd[k] * S), S - 1)]
        buckets[min(int(al * 4), 3)].append((cxd[k] * S + FLOW[k, 0] * d, cyd[k] * S + FLOW[k, 1] * d, DW * 0.62 * (1 - 0.35 * e), k, col))
    for lv, items in enumerate(buckets):
        if not items: continue
        ov = pic.copy(); m = np.zeros((S, S), np.float32)
        for cx, cy, w, k, col in items:
            ang = float(np.degrees(np.arctan2(FLOW[k, 1], FLOW[k, 0]))); ax = (max(int(w), 2), max(int(DH * 0.62), 1))
            cv2.ellipse(ov, (int(cx), int(cy)), ax, ang, 0, 360, [float(c) for c in col], -1, cv2.LINE_AA)
            cv2.ellipse(m, (int(cx), int(cy)), ax, ang, 0, 360, 1.0, -1, cv2.LINE_AA)
        pic = over(pic, ov, (lv + 0.5) / 4 * m)
    return pic

# ---------- 2번이 쌓임: 땅이 솟아오름 → 하늘이 왼쪽에서 오른쪽으로 칠해짐 → 사람 붓터치가 모임 ----------
B2 = 3.5
TB_G, TB_S, TB_R = (0.0, 0.9), (0.6, 2.2), (2.1, 3.4)
GB = make_gather(B.msoft, 72)
_rag_lr = cv2.GaussianBlur(np.random.RandomState(73).rand(S // 6 + 1).astype(np.float32)[:, None], (0, 0), 0.8)[:, 0]
RAG_LR = np.repeat(_rag_lr, 6)[:S][:, None] * 70                       # 붓 앞머리가 줄마다 들쭉날쭉
_rag_up = cv2.GaussianBlur(np.random.RandomState(74).rand(S // 8 + 1).astype(np.float32)[None], (0, 0), 0.8)[0]
RAG_UP = np.repeat(_rag_up, 8)[:S][None, :] * 50
def build_B(tb):
    full = sky_flow(SPB, tb); pic = PAPER_IMG.copy()
    u = (tb - TB_S[0]) / (TB_S[1] - TB_S[0])
    if u > 0:                                              # 하늘: 왼쪽에서 오른쪽으로 붓질하듯
        fx = ease_out(u, 2.4) * (S + 120) - 70
        pic = over(pic, full, np.clip((fx + RAG_LR - xx) / 14, 0, 1))
    u = (tb - TB_G[0]) / (TB_G[1] - TB_G[0])
    if u > 0:                                              # 땅: 아래에서 솟아올라 느려지며 제자리
        dy = (1 - back_out(u)) * 220
        wr = lambda m: cv2.remap(m, xx.astype(np.float32), (yy - dy).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        gm = wr(SPB.gw)
        pic = over(pic, wr(full * SPB.gw[..., None]) / np.maximum(gm, 1e-3)[..., None], gm * clamp01(u * 2.5))
    u = (tb - TB_R[0]) / (TB_R[1] - TB_R[0])
    if u > 0: pic = runner_gather(pic, u, GB, B.rcol)
    top = lerp(PAPER * 0.92, B.top, smooth((tb - TB_G[0]) / 0.9))
    bot = lerp(PAPER * 0.92, B.bot, smooth((tb - TB_S[0]) / 1.3))
    return compose(pic, top, bot, B.split)

# ---------- 2→3: 확대하며 2번 사람이 흩어짐 → 붓터치 화면이 빈 종이로 → 물러나며 3번이 쌓임 ----------
Z_IN, Z_MID, Z_OUT = 1.3, 0.45, 2.7
TC_G, TC_S, TC_R = (0.0, 0.4), (0.2, 0.75), (0.45, 1.0)      # 물러나는 동안(비율): 땅 → 하늘(지평선에서 위로) → 사람
def built_C(tc, uo):
    full = sky_flow(SPC, tc); pic = PAPER_IMG.copy()
    u = (uo - TC_S[0]) / (TC_S[1] - TC_S[0])
    if u > 0:
        fy = SPC.hor + 40 - ease_out(u, 2.2) * (SPC.hor + 120)
        pic = over(pic, full, np.clip((yy - fy + RAG_UP) / 14, 0, 1))
    u = (uo - TC_G[0]) / (TC_G[1] - TC_G[0])
    if u > 0:
        dy = (1 - back_out(u)) * 200
        wr = lambda m: cv2.remap(m, xx.astype(np.float32), (yy - dy).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        gm = wr(SPC.gw)
        pic = over(pic, wr(full * SPC.gw[..., None]) / np.maximum(gm, 1e-3)[..., None], gm * clamp01(u * 2.5))
    return pic
def zoom_build(t, tb):
    t = math.floor(t * FPS / 2) * 2 / FPS
    if t < Z_IN:
        u = t / Z_IN; e = u ** 1.8
        z = math.exp(math.log(ZM) * e); k = smooth(min(1, u * 1.3))
        c = (lerp(S / 2, B.core[0], k), lerp(S / 2, B.core[1], k))
        with _Swap(B, 'norun', sky_flow(SPB, tb)):
            return compose(zoom_scatter(B, z, c, smooth((u - 0.3) / 0.7)), B.top, B.bot, B.split)
    if t < Z_IN + Z_MID:                                    # 붓터치 화면이 칸마다 빈 종이 붓터치로
        u = (t - Z_IN) / Z_MID
        with _Swap(B, 'norun', sky_flow(SPB, tb)): fa = zoom_scatter(B, ZM, B.core, 1.0)
        with _Swap(Cc, 'norun', PAPER_IMG): fb = zoom_scatter(Cc, ZM, Cc.core, 1.0, gather=True)
        sw = cv2.GaussianBlur(np.clip((u * 1.45 - SCELL * 0.45 - 0.1) / 0.12, 0, 1), (0, 0), 1.5)[..., None]
        m = smooth(u)
        return compose(fa * (1 - sw) + fb * sw, lerp(B.top, PAPER * 0.92, m), lerp(B.bot, PAPER * 0.92, m), B.split)
    uo = (t - Z_IN - Z_MID) / Z_OUT; e = 1 - (1 - uo) ** 3      # 물러나며 3번이 쌓임, 사람은 붓터치가 모여 마지막에
    z = math.exp(math.log(ZM) * (1 - e)); k = smooth(min(1, uo * 1.3))
    c = (lerp(Cc.core[0], S / 2, k), lerp(Cc.core[1], S / 2, k))
    tc = t - Z_IN - Z_MID
    sc = 1 - smooth((uo - TC_R[0]) / (TC_R[1] - TC_R[0]))
    with _Swap(Cc, 'norun', built_C(tc, uo)):
        img = zoom_scatter(Cc, z, c, sc, gather=True)
    top = lerp(PAPER * 0.92, Cc.top, smooth((uo - TC_G[0]) / 0.35))
    bot = lerp(PAPER * 0.92, Cc.bot, smooth((uo - TC_S[0]) / 0.5))
    return compose(img, top, bot, Cc.split)

# ---------- 릴스 순서 ----------
H1, H2, H3 = 6.0, 6.6, 8.4                                # 자막 시간(5·5·5·5·3·3·3·3·3초 = 35초)에 맞춘 머묾
def seg_build1(i, t): return build_frame(i, t)
def seg_A(i, t): return compose(anim_A(i), A.top, A.bot, A.split)
def seg_out1(i, t):
    m = smooth(t / OUT1)
    return compose(paper_out(t, anim_A(i)), lerp(A.top, PAPER * 0.92, m), lerp(A.bot, PAPER * 0.92, m), A.split)
def seg_build2(i, t): return build_B(t)
def seg_B(i, t): return compose(with_runner(B, sky_flow(SPB, B2 + t)), B.top, B.bot, B.split)
def seg_zoom(i, t): return zoom_build(t, B2 + H2 + t)
def seg_C(i, t): return compose(with_runner(Cc, sky_flow(SPC, Z_OUT + t)), Cc.top, Cc.bot, Cc.split)
REEL = [(BUILD_LEN, seg_build1), (H1, seg_A), (OUT1, seg_out1), (B2, seg_build2), (H2, seg_B),
        (Z_IN + Z_MID + Z_OUT, seg_zoom), (H3, seg_C)]
def reel_frame(i):
    t = i / FPS
    for L, fn in REEL:
        if t < L or (L, fn) == REEL[-1]:
            fr = fn(i, min(t, L - 1e-3)); break
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
if MODE == 'reelsheet':                                    # 릴스 전체를 일정 간격으로 뽑은 확인용 시트
    n = int(round(REEL_LEN * FPS)); idx = np.linspace(0, n - 1, 24).astype(int)
    tiles = [cv2.resize(reel_frame(i)[FY - 140:FY + S + 40, :], (162, 307), interpolation=cv2.INTER_AREA) for i in idx]
    rows = [np.hstack([np.pad(x, ((0, 18), (2, 2), (0, 0)), constant_values=20) for x in tiles[r * 12:(r + 1) * 12]]) for r in range(2)]
    im = Image.fromarray(np.vstack(rows)); dd = ImageDraw.Draw(im)
    for k, i in enumerate(idx): dd.text(((k % 12) * 166 + 6, (k // 12) * 325 + 309), f'{i / FPS:.1f}s', fill=(230, 230, 230))
    im.save(os.path.join(OUT, 'sheet_reel.png')); print('reel', f'{REEL_LEN:.2f}s')
if MODE == 'sheet':
    for name, fn, L, a, b in SEQ:
        ts = [L * i / 9 for i in range(10)]
        tiles = [cv2.resize(fn(min(t, L - 1e-3), a, b), (216, 384), interpolation=cv2.INTER_AREA) for t in ts]
        sheet = np.hstack([np.pad(x, ((0, 22), (2, 2), (0, 0)), constant_values=20) for x in tiles])
        im = Image.fromarray(sheet); dd = ImageDraw.Draw(im)
        for i, t in enumerate(ts): dd.text((i * 220 + 8, 388), f'{t:.2f}s', fill=(230, 230, 230))
        im.save(os.path.join(OUT, f'sheet_{name}.png')); print(name, f'{L:.2f}s')
