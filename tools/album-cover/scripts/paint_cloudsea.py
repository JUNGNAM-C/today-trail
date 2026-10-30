# 구름바다 위 바위 사이를 뛰는 사진 → 앨범 커버형 그림 + 1080x1350 이미지
# 사용: python paint_cloudsea.py <사진> <출력폴더>
import sys, os, cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import median_filter
SRC, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
FONTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fonts')
src = np.array(Image.open(SRC).convert('RGB'))
H, W = src.shape[:2]
sq = src[:, 0:H]                       # 왼쪽 정방형: 바위와 사람이 들어가는 쪽
N = 1000
img = cv2.resize(sq, (N, N), interpolation=cv2.INTER_CUBIC)
gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 2)
yy, xx = np.mgrid[0:N, 0:N]
def hx(h): return np.array([int(h[i:i+2], 16) for i in (1, 3, 5)], np.float32)
rng = np.random.RandomState(7)
def nz(sig, sx=None, sy=None):
    r = cv2.GaussianBlur(rng.randn(N, N).astype(np.float32), (0, 0), sigmaX=sx or sig, sigmaY=sy or sig)
    return r / (r.std() + 1e-6)

# ---------- 영역 나누기 ----------
HOR = int(N * 143 / 266)               # 구름바다 수평선
# 바위: 아래에서 이어지는 어두운 덩어리(위쪽 비네팅 제외)
dark = ((gray < 75) & (yy > N * 0.38)).astype(np.uint8)
n, lab, st, _ = cv2.connectedComponentsWithStats(dark)
ids = [i for i in range(1, n) if st[i, 4] > 3000 and st[i, 1] + st[i, 3] >= N - 5]
rock = np.isin(lab, ids)
rock = cv2.morphologyEx(rock.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
rock = cv2.GaussianBlur(rock.astype(np.float32), (0, 0), 5) > 0.5
# 바위 윗선을 열 단위로 정리(구멍 메우기)
top = np.full(N, N, np.float32)
for x in range(N):
    ys = np.where(rock[:, x])[0]
    if len(ys): top[x] = ys.min()
rock = yy >= top[None, :]
# 사람: 바위 위 하늘 속 어두운 덩어리
box = (xx > N * 38 / 266) & (xx < N * 95 / 266) & (yy > N * 92 / 266) & (yy < N * 140 / 266)
# 사진이 작아 원본 실루엣을 또렷하게: 주변 하늘 밝기와의 차이로 따고 살짝만 다듬는다
g0 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
bgl = cv2.GaussianBlur(g0, (0, 0), 30)
cand = (((bgl - g0) > 45) & box & ~cv2.dilate(rock.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)).astype(np.uint8)
n2, l2, st2, _ = cv2.connectedComponentsWithStats(cand)
big = 1 + np.argmax(st2[1:, 4])
runner = cv2.GaussianBlur((l2 == big).astype(np.float32) * 255, (0, 0), 1.6) > 115
from scipy.ndimage import binary_fill_holes
runner = binary_fill_holes(runner)
# 들린 팔 아래 머리 오른쪽에 붙은 혹 같은 조각만 지운다
runner &= ~((xx >= 244) & (yy >= 418) & (yy <= 427))
# 팔 밑 머리 오른쪽 덩어리: 팔 굵기(어깨→손 선분 둘레)는 남긴다
ax0, ay0, ax1, ay1 = 236.0, 413.0, 262.0, 399.0
tt = np.clip(((xx - ax0) * (ax1 - ax0) + (yy - ay0) * (ay1 - ay0)) / ((ax1 - ax0) ** 2 + (ay1 - ay0) ** 2), 0, 1)
armd = np.hypot(xx - (ax0 + tt * (ax1 - ax0)), yy - (ay0 + tt * (ay1 - ay0)))
runner &= ~((xx >= 243) & (yy >= 407) & (yy <= 418) & (armd > 4.5))
runner = cv2.GaussianBlur(runner.astype(np.float32), (0, 0), 1.5) > 0.5     # 잘린 자리 매끈하게

# 상체(머리·몸통·팔)는 다른 사진(역광 러너)의 실루엣으로 바꾸고, 하체는 기존 실루엣을 쓴다
if len(sys.argv) > 3:
    up = np.array(Image.open(sys.argv[3]).convert('RGB')).astype(np.float32)
    UH, UW = up.shape[:2]; uy, ux = np.mgrid[0:UH, 0:UW]
    # 하늘을 매끈한 3차 곡면으로 맞춘 뒤 그 차이로 사람을 딴다(흰 셔츠도 하늘과 색이 달라 잡힌다)
    sel = (uy < 440) & (ux > 380) & ~((ux > 530) & (ux < 810) & (uy > 140) & (uy < 500)) & (up.mean(2) > 150)
    def basis(x, y):
        x = x / UW; y = y / UH
        return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y, x ** 3, y ** 3, x * x * y, x * y * y], -1)
    Ball = basis(ux.astype(np.float32), uy.astype(np.float32)); model = np.zeros_like(up)
    for c in range(3):
        co, *_ = np.linalg.lstsq(Ball[sel], up[..., c][sel], rcond=None); model[..., c] = Ball @ co
    pm = (cv2.GaussianBlur(np.abs(up - model).sum(2), (0, 0), 1.2) > 60).astype(np.uint8)
    pm[:140] = 0; pm[:, :530] = 0; pm[:, 810:] = 0
    n3, l3, s3, _ = cv2.connectedComponentsWithStats(pm)
    pm = l3 == 1 + np.argmax(s3[1:, 4])
    WAIST_U, WAIST_X_U = 292, 672                  # 셔츠 끝(허리) 높이와 가운데
    HIP_O, HIP_X_O = 467, 236                      # 기존 실루엣에서 다리가 갈라지는 허리
    ys_ = np.where(pm.any(1))[0]; topu = ys_.min()
    sc = (HIP_O - 397) / float(WAIST_U - topu)     # 머리끝~허리 길이를 기존과 맞춘다
    # 가는 다리에 맞춰 줄인다: 가슴·머리는 72%, 두 팔은 66%
    ST, SA = sc * 0.72, sc * 0.66
    pm[WAIST_U + 8:] = False
    LA = pm & (ux <= 652)                                         # 왼팔(뒤로 뻗은 팔), 어깨에서 조금 겹치게
    RA = pm & (ux >= 720) & (uy >= 194) & (uy <= 228)             # 오른팔(앞으로 굽힌 팔)
    TOR = pm & ~((ux <= 646) | ((ux >= 726) & (uy >= 194) & (uy <= 228)))
    def place(part, s, src, dst):
        Mx = np.float32([[s, 0, dst[0] - s * src[0]], [0, s, dst[1] - s * src[1]]])
        return cv2.warpAffine(part.astype(np.float32), Mx, (N, N), flags=cv2.INTER_AREA)
    def to_dst(p): return (HIP_X_O + (p[0] - WAIST_X_U) * ST, HIP_O + (p[1] - WAIST_U) * ST)
    LSH, RSH = (648, 208), (724, 212)                             # 어깨: 팔을 줄이는 기준점
    newu = np.maximum.reduce([place(TOR, ST, (WAIST_X_U, WAIST_U), (HIP_X_O, HIP_O)),
                              place(LA, SA, LSH, to_dst(LSH)), place(RA, SA, RSH, to_dst(RSH))])
    ox = oy = 0; sc = ST
    lower = runner & (yy >= HIP_O - 4)
    runner = (newu > 0.5) | lower
    runner = cv2.GaussianBlur(runner.astype(np.float32), (0, 0), 1.0) > 0.5
    print('upper scale', round(sc, 3), 'offset', ox, oy)
SKY = (yy < HOR) & ~rock
CLOUD = (yy >= HOR) & ~rock
G = ~SKY                                  # 땅 쪽(바위+구름바다)
print('runner px', runner.sum(), 'rock px', rock.sum())

# ---------- 하늘: 6개 층이 수평선 위 태양을 중심으로 둥글게 휜다 ----------
SUNX, SUNY, SUNR = int(N * 0.78), HOR + 6, 46     # 태양: 구름바다 경계에 반쯤 걸침
AX, AY = N * 2.1, HOR * 1.0
u = (xx - SUNX) / AX; v = (yy - SUNY) / AY
F = np.sqrt(u ** 2 + v ** 2)                       # 태양에서 멀어지는 정도(0=태양, 1=하늘 꼭대기)
TH = np.arctan2(-v, u)                             # 태양을 도는 각도
disp = cv2.GaussianBlur(rng.randn(N, N).astype(np.float32), (0, 0), 40) * 0.7
# 후광: 태양 가까이는 둥근 고리, 멀어질수록 하늘 층의 넓은 물결 모양으로 서서히 바뀐다
RPX = np.hypot(xx - SUNX, (yy - SUNY) * 1.1)
WC = np.exp(-(RPX / 330.0) ** 2)                   # 둥근 정도(태양에서 1 → 멀리 0)
FM = WC * RPX / 3000.0 + (1 - WC) * F
DSP = disp * (1 - 0.85 * WC)                       # 태양 곁 고리는 매끈, 멀수록 다른 층처럼 너울
LV = 1 - FM + DSP                                  # 층 높이: 태양 쪽일수록 아래 층(밝은 색)
bands = [(0.00, '#77768A'), (0.20, '#9A8FA0'), (0.38, '#C4AEB0'), (0.56, '#E6CDBF'), (0.73, '#F4C9A2'), (0.87, '#F1A877'),
         (0.952, '#F6B98A'), (0.974, '#F9D2A6')]
cols = [hx(h) for _, h in bands]
kmap = np.zeros((N, N), np.int32)
for i, (b, _) in enumerate(bands): kmap[LV >= b] = i
sky = np.array(cols)[kmap]
SUN = (np.hypot(xx - SUNX, yy - SUNY) < SUNR) & (yy < HOR + 2)
# 붓결 좌표: 태양을 도는 호를 따라 흐르도록 (각도, 거리) 공간의 결을 가져온다
mapx = (TH / np.pi * 2000).astype(np.float32); mapy = ((FM - DSP * 0.25) * 1000).astype(np.float32)

# ---------- 구름바다: 뭉게 덩어리(그림자 몸통 + 밝은 윗머리) ----------
CL = [hx('#8E8193'), hx('#A897A6'), hx('#C6AFB3'), hx('#E2C6BC'), hx('#F3D8C6'), hx('#FCECDD')]
cloud = np.tile(CL[1], (N, N, 1))
lum = cv2.GaussianBlur(img.astype(np.float32).mean(2), (0, 0), 14)
lv = lum[CLOUD]; lo, hi = np.percentile(lv, 8), np.percentile(lv, 96)
Ln = np.clip((lum - lo) / (hi - lo + 1e-6), 0, 1)
Ln = np.clip(Ln + 0.45 * np.exp(-((xx - SUNX) / 260.0) ** 2 - ((yy - HOR) / 160.0) ** 2), 0, 1)   # 태양 쪽 구름 윗머리가 더 밝게
mass = nz(60)
crng = np.random.RandomState(31)
placed = []
pts = [(crng.randint(HOR - 6, N + 60), crng.randint(-80, N + 80)) for _ in range(20000)]
pts.sort()                               # 먼 것(위)부터 칠해 가까운 구름이 덮도록
for y, x in pts:
    dep = np.clip((y - HOR) / (N - HOR), 0, 1)
    w = int((26 + 190 * dep ** 1.3) * crng.uniform(0.8, 1.2))      # 원근: 수평선 쪽은 작고 납작
    h = int(w * crng.uniform(0.30, 0.40))
    if any(abs(y - py) < 0.5 * max(h, ph) and abs(x - px) < 0.75 * max(w, pw) for px, py, pw, ph in placed[-300:]): continue
    placed.append((x, y, w, h))
    cx, cy = min(max(x, 0), N - 1), min(max(y, 0), N - 1)
    t = np.clip(0.35 + Ln[cy, cx] * 0.6 + mass[cy, cx] * 0.12 + (1 - dep) * 0.15 + (crng.rand() - 0.5) * 0.2, 0, 1)
    k = int(np.clip(t * 4.2, 0, 4))
    body, cap = [float(c) for c in CL[max(k - 1, 0)]], [float(c) for c in CL[min(k + 1, 5)]]
    # 뭉게구름 한 덩이: 납작한 몸통 + 위로 솟은 둥근 봉우리 2~4개, 봉우리 윗머리만 밝게
    cv2.ellipse(cloud, (x, y), (w, h), 0, 0, 360, body, -1, cv2.LINE_AA)
    nb = crng.randint(2, 5)
    for j in range(nb):
        bx = int(x + (j - (nb - 1) / 2) * w * 0.55 + crng.uniform(-0.1, 0.1) * w)
        r = int(w * crng.uniform(0.28, 0.42)); by = int(y - h * 0.35)
        cv2.ellipse(cloud, (bx, by), (r, int(r * 0.72)), 0, 0, 360, body, -1, cv2.LINE_AA)
        cv2.ellipse(cloud, (bx - r // 6, by - r // 5), (int(r * 0.82), int(r * 0.5)), 0, 180, 360, cap, -1, cv2.LINE_AA)
cloud = cv2.GaussianBlur(cloud, (0, 0), 1.2)

# ---------- 바위: 보로노이 면 + 노을빛 윗가장자리 ----------
RT = [hx('#4A3231'), hx('#5D3E37'), hx('#724C41'), hx('#8A5D4B')]
rockimg = np.zeros((N, N, 3), np.float32)
step = 62
seeds = np.array([(gx + rng.randint(-24, 24), gy + rng.randint(-24, 24)) for gy in range(0, N + step, step) for gx in range(0, N + step, step)], np.float32)
rys, rxs = np.where(rock)
wx = nz(24) * 14; wy = nz(24) * 14
P = np.stack([rxs + wx[rys, rxs], rys + wy[rys, rxs]], 1).astype(np.float32)
cid = np.empty(len(P), np.int32)
for s in range(0, len(P), 200000):
    cid[s:s + 200000] = ((P[s:s + 200000, None, :] - seeds[None]) ** 2).sum(2).argmin(1)
ftone = rng.rand(len(seeds))
# 빛은 오른쪽 위(해 쪽)에서: 오른쪽·위쪽 면이 밝다
tone = np.clip((ftone * 0.6 + (seeds[:, 0] / N) * 0.25 + (1 - seeds[:, 1] / N) * 0.15) * 4, 0, 3).astype(int)
lab_r = np.full((N, N), -1, np.int32); lab_r[rys, rxs] = cid
tmap = np.full((N, N), -1, np.int32); tmap[rys, rxs] = tone[cid]
rockimg[rys, rxs] = np.array(RT)[tone[cid]]
up = np.zeros((N, N), bool); up[1:] = (lab_r[1:] != lab_r[:-1]) & (lab_r[1:] >= 0)
above = np.full((N, N), -1); above[1:] = tmap[:-1]
lit = cv2.dilate(up.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & rock & (tmap > above) & (above >= 0)
rockimg[lit] = hx('#C98A68')
rockimg = cv2.GaussianBlur(rockimg, (0, 0), 0.9)

out = sky.copy()
out[SUN] = hx('#FFF3DC')
out[CLOUD] = cloud[CLOUD]
out[rock] = rockimg[rock]
out[runner] = np.array([42, 26, 48], np.float32)
f = out.copy()

# ---------- 수채 질감 ----------
def pnz(sx, sy):                                  # (각도 2000 x 거리 1000) 공간의 결
    r = cv2.GaussianBlur(rng.randn(1000, 2000).astype(np.float32), (0, 0), sigmaX=sx, sigmaY=sy)
    return r / (r.std() + 1e-6)
def follow(t): return cv2.remap(t, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
PZ = [pnz(140, 1.2), pnz(70, 2.4), pnz(220, 3.5)]
st1, st2_, st3 = [follow(p) for p in PZ]
amp = 0.45 + 0.9 * np.clip(nz(80) * 0.5 + 0.5, 0, 1)
bandamp = np.ones(N, np.float32)
for y0 in range(0, N, 70): bandamp[y0:y0 + 70] = rng.uniform(0.6, 1.4)
amp = amp * bandamp[:, None]
f *= (1 + (0.016 * st1 + 0.010 * st2_ + 0.012 * st3) * amp * SKY)[..., None]
f *= (1 + 0.012 * nz(0, sx=60, sy=1.6) * CLOUD)[..., None]            # 구름바다: 가로 붓결
M = cv2.getRotationMatrix2D((N / 2, N / 2), 24, 1.0)
diag = cv2.warpAffine(nz(0, sx=34, sy=1.1), M, (N, N), borderMode=cv2.BORDER_REFLECT)
f += diag[..., None] * (rock * 2.6)[..., None]
wv = np.where(G, 1.0, 0.25)
W26, W7 = nz(26), nz(7)
f *= (1 + 0.045 * W26 * wv)[..., None]
f *= (1 + 0.03 * W7 * wv)[..., None]
edge = cv2.Canny(np.clip(f, 0, 255).astype(np.uint8), 30, 80)
edge = cv2.GaussianBlur(edge.astype(np.float32), (0, 0), 2.6) / 255.0
# 산 윤곽(하늘·구름과 만나는 곳)에는 테두리처럼 보이는 어두운 고임을 두지 않는다
rockline = cv2.dilate(rock.astype(np.uint8), np.ones((15, 15), np.uint8)) & (1 - cv2.erode(rock.astype(np.uint8), np.ones((15, 15), np.uint8)))
edge *= 1 - cv2.GaussianBlur(rockline.astype(np.float32), (0, 0), 3)
f *= (1 - 0.18 * np.clip(edge * 2.0, 0, 1) * np.where(G, 1.0, 0.5))[..., None]
grain = cv2.GaussianBlur(rng.randn(N, N).astype(np.float32), (0, 0), 0.7)
dk = 1 - np.clip(f.mean(2) / 255, 0, 1)
f += (grain * (6 + 14 * dk) * np.where(G, 1.0, 0.35))[..., None]
f = np.clip(f, 0, 255)
fb = cv2.addWeighted(f, 0.72, cv2.GaussianBlur(f, (0, 0), 1.6), 0.28, 0)
f = np.where(G[..., None], fb, f)
paint = np.clip(f, 0, 255).astype(np.uint8)
Image.fromarray(paint).save(os.path.join(OUT, 'paint.png'))
# 전환 영상용 개체 마스크(사람·바위·구름바다·해)
np.savez_compressed(os.path.join(OUT, 'layers.npz'), runner=runner, rock=rock, cloud=CLOUD, sun=SUN, hor=HOR,
                    fm=FM.astype(np.float16), sun_xy=np.array([SUNX, SUNY, SUNR]))   # fm: 태양에서 멀어지는 정도(하늘이 쌓이는 순서)

# ---------- 릴스용 하늘 고리 영상(3초, 30fps, 끝과 처음이 이어짐) ----------
# 하늘 층 경계는 제자리에 두고(오므라드는 느낌이 없도록), 태양에서 밝은 고리가 바깥으로만 퍼져 나가며
# 층 안의 붓결도 바깥으로 흘러간다.
if os.environ.get('SKY_LOOP'):
    LD = os.path.join(OUT, 'sky_loop'); os.makedirs(LD, exist_ok=True)
    NF, LAM = 90, 0.22                                   # 고리 간격(태양에서 멀어지는 정도 단위)
    SK = SKY & ~runner & ~SUN
    base = paint.astype(np.float32)
    for fi in range(NF):
        ph = fi / NF
        s_ = sky.copy()
        # 바깥으로만 나아가는 빛의 고리: 날카로운 밝은 띠가 태양에서 멀어질수록 넓고 옅어짐
        ring = (0.5 + 0.5 * np.sin(2 * np.pi * (FM / LAM - ph))) ** 4
        fade = np.exp(-FM / 0.55)
        s_ = s_ * (1 + 0.075 * ring * fade)[..., None] + (ring * fade * 6)[..., None]
        # 붓결이 바깥으로 흘러가도록 결 좌표를 밀고, 앞뒤 두 장을 섞어 끝과 처음을 잇는다
        def flow(p, off):
            return cv2.remap(p, mapx, (mapy - off).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        stt = []
        for p in PZ:
            a_ = flow(p, LAM * 1000 * ph); b_ = flow(p, LAM * 1000 * (ph - 1))
            stt.append((a_ * (1 - ph) + b_ * ph) / np.sqrt((1 - ph) ** 2 + ph ** 2))
        fs = s_ * (1 + (0.016 * stt[0] + 0.010 * stt[1] + 0.012 * stt[2]) * amp)[..., None]
        fs *= (1 + 0.045 * W26 * 0.25)[..., None]; fs *= (1 + 0.03 * W7 * 0.25)[..., None]
        e_ = cv2.GaussianBlur(cv2.Canny(np.clip(fs, 0, 255).astype(np.uint8), 30, 80).astype(np.float32), (0, 0), 2.6) / 255.0
        fs *= (1 - 0.09 * np.clip(e_ * 2.0, 0, 1))[..., None]
        fs += (grain * (6 + 14 * (1 - np.clip(fs.mean(2) / 255, 0, 1))) * 0.35)[..., None]
        fr = base.copy(); fr[SK] = fs[SK]
        Image.fromarray(np.clip(fr, 0, 255).astype(np.uint8)).resize((840, 840), Image.LANCZOS).save(os.path.join(LD, f's_{fi:02d}.jpg'), quality=94)
    print('sky loop', NF, 'frames')

# ---------- 1080x1350 배치 ----------
topc = tuple(int(v) for v in RT[2])        # 위 띠 = 땅(바위) 색
botc = tuple(int(v) for v in cols[0])      # 아래 띠 = 맨 위 하늘 색
CW, CH = 1080, 1350; S = 840; sx = (CW - S) // 2; sy = 190
cv = Image.new('RGB', (CW, CH), topc); d = ImageDraw.Draw(cv)
split = int(round(sy + HOR * S / N))
d.rectangle([0, split, CW, CH], fill=botc)
cv.paste(Image.fromarray(paint).resize((S, S), Image.LANCZOS), (sx, sy))
def spaced(text, y, size, fill, gap, fontpath, fit=None):
    fnt = ImageFont.truetype(fontpath, size)
    ws = [d.textlength(c, font=fnt) for c in text]
    if fit: gap = (fit - sum(ws)) / (len(text) - 1)
    x = (CW - (sum(ws) + gap * (len(text) - 1))) / 2
    for c, w in zip(text, ws):
        d.text((x, y), c, font=fnt, fill=fill); x += w + gap
spaced('KILIAN JORNET', 100, 44, (245, 238, 222), 0, os.path.join(FONTS, 'WantedSans-Bold.ttf'), fit=840)
spaced('RICH ROLL PODCAST', 1232, 26, (36, 52, 78), 12, os.path.join(FONTS, 'WantedSans-Medium.ttf'))
mask = Image.new('L', (CW, CH), 0); ImageDraw.Draw(mask).rounded_rectangle([0, 0, CW - 1, CH - 1], 40, fill=255)
bg = Image.new('RGB', (CW, CH), (12, 22, 48)); bg.paste(cv, (0, 0), mask)
bg.save(os.path.join(OUT, 'album_cover.png'))
print('split', split, 'top', topc, 'bottom', botc)
