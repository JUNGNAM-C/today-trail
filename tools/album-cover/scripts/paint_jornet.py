import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
src=np.array(Image.open('/mnt/user-data/attachments/5a7d1354-b855-5f9c-b6ac-3454f8809492/51ed2dd9-image.png').convert('RGB'))
H,W=src.shape[:2]; print(W,H)
x0=70; sq=src[:, x0:x0+H]
N=1000
img=cv2.resize(sq,(N,N),interpolation=cv2.INTER_AREA)
gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY).astype(np.float32)
sm=cv2.GaussianBlur(gray,(0,0),4)
dark=(sm<95).astype(np.uint8)
# ground = dark connected to bottom; runner = dark component above ground
n,lab,st,_=cv2.connectedComponentsWithStats(dark)
bot=lab[N-1,:]; gid=np.bincount(bot[bot>0]).argmax() if (bot>0).any() else 0
ground=(lab==gid)
# fill ground area below ridge line: for each column, below first ground pixel is ground
ridge=np.full(N,N)
for x in range(N):
    ys=np.where(ground[:,x])[0]
    if len(ys): ridge[x]=ys.min()
ridge=cv2.GaussianBlur(ridge.astype(np.float32).reshape(1,-1),(0,0),6).reshape(-1)
yy=np.arange(N)[:,None]
G=(yy>=ridge[None,:]-2)
hi=cv2.resize(sq,(N,N),interpolation=cv2.INTER_CUBIC)
hg=cv2.cvtColor(hi,cv2.COLOR_RGB2GRAY).astype(np.float32)
hg=cv2.GaussianBlur(hg,(0,0),1.0)
cand=((hg<135)&(~G)).astype(np.uint8)
cand=cv2.morphologyEx(cand,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
n2,l2,st2,_=cv2.connectedComponentsWithStats(cand)
keep=np.zeros_like(cand)
for i in range(1,n2):
    if st2[i,4]>150: keep[l2==i]=1
runner=cv2.GaussianBlur(keep.astype(np.float32)*255,(0,0),1.4)>120
runner=cv2.dilate(runner.astype(np.uint8),np.ones((3,3),np.uint8))>0
rs=np.random.RandomState(5)
# sky: banded gouache gradient sampled from photo's column-wise vertical profile
skyprof=cv2.GaussianBlur(img.astype(np.float32),(0,0),25)
prof=np.median(skyprof[:, :], axis=1)  # N x3
# lift & quantise into bands
def lift(c): 
    c=np.array(c,np.float32); return np.clip(c*0.9+28,0,255)
bands=[(0,'#82A9D8'),(200,'#A8C9EA'),(330,'#E3EBEF'),(420,'#F7E3BC'),(480,'#F7CB8A'),(540,'#F3A758')]
sky=np.zeros((N,N,3),np.float32)
cols=[np.array([int(h[i:i+2],16) for i in (1,3,5)],np.float32) for _,h in bands]
ys=[b for b,_ in bands]
for y in range(N):
    # soft stepped bands with wobble
    k=0
    for i,b in enumerate(ys):
        if y>=b: k=i
    sky[y]=cols[k]
# wobbly band edges: displace rows by smooth noise
disp=cv2.GaussianBlur(rs.randn(N,N).astype(np.float32),(0,0),40)*900
mapy=(np.arange(N)[:,None]+disp).astype(np.float32); mapx=np.tile(np.arange(N,dtype=np.float32),(N,1))
sky=cv2.remap(sky,mapx,mapy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
sky=cv2.GaussianBlur(sky,(0,0),1.5)
# ---------- ground: shrubs + rock outcrops ----------
def hx(h): return np.array([int(h[i:i+2],16) for i in (1,3,5)],np.float32)
hi_f=hi.astype(np.float32)
bright=cv2.GaussianBlur(hi_f.mean(2),(0,0),6)
Lg=bright
gr=hi_f[...,1]-np.maximum(hi_f[...,0],hi_f[...,2])          # green dominance
grs=cv2.GaussianBlur(gr,(0,0),10)
rockness=(grs<np.percentile(grs[G],38))&(Lg>np.percentile(Lg[G],30))
rock=cv2.morphologyEx(rockness.astype(np.uint8)*255,cv2.MORPH_CLOSE,np.ones((25,25),np.uint8))
rock=cv2.GaussianBlur(rock,(0,0),9)>120
rock=cv2.medianBlur((rock*255).astype(np.uint8),31)>0
rock&=G
# extra rock caps along the ridge on the right and near the left top
yy2,xx2=np.mgrid[0:N,0:N]
ridgeband=(yy2<ridge[None,:]+70)&G
rock|=(ridgeband&(xx2>700)&(grs<np.percentile(grs[G],55)))
rock=cv2.medianBlur((rock*255).astype(np.uint8),25)>0
rock&=G
rng=np.random.RandomState(23)
rock_old=rock.copy()
def nz(sig):
    r=cv2.GaussianBlur(rng.randn(N,N).astype(np.float32),(0,0),sig); return r/(r.std()+1e-6)
depth=np.clip((yy2-ridge[None,:])/(N-ridge.min()),0,1)
fld=nz(46)*0.8+nz(20)*0.35+(0.55-depth)*0.9
rock=rock_old.copy()
rock&=G
shr=np.tile(hx('#2C5A47'),(N,N,1))
GREENS=[hx('#1F4A3B'),hx('#2C5A47'),hx('#3E7557'),hx('#5C9468'),hx('#86B67A')]
rng=np.random.RandomState(11)
Lnorm=(Lg-np.percentile(Lg[G],5))/(np.percentile(Lg[G],95)-np.percentile(Lg[G],5)+1e-6)
Lnorm=np.clip(Lnorm,0,1)
shrub=shr.copy()
mass=rnd0=cv2.GaussianBlur(rng.randn(N,N).astype(np.float32),(0,0),55); mass=mass/mass.std()
for _ in range(900):
    x=rng.randint(0,N); y=rng.randint(int(ridge.min()),N)
    if not G[min(y,N-1),x] or rock[y,x]: continue
    t=np.clip(0.45+mass[y,x]*0.26+(Lnorm[y,x]-0.4)*0.5-(y-ridge[x])/900*0.25,0,1)
    k=int(np.clip(t*4.4,0,4)); w=rng.randint(50,95); h=int(w*rng.uniform(0.5,0.75))
    cv2.ellipse(shrub,(x,y),(w,h),rng.uniform(-20,20),0,360,[float(c) for c in GREENS[k]],-1,cv2.LINE_AA)
pts=[]
for _ in range(15000):
    x=rng.randint(0,N); y=rng.randint(int(ridge.min())-10,N)
    if not G[min(y,N-1),x] or rock[y,x]: continue
    pts.append((y,x))
pts.sort()
for y,x in pts:
    t=0.45+mass[y,x]*0.24+(Lnorm[y,x]-0.4)*0.45+(rng.rand()-0.5)*0.35-(y-ridge[x])/900*0.2
    k=int(np.clip(t*4.6,0,4))
    w=rng.randint(18,40); h=int(w*rng.uniform(0.55,0.85)); ang=rng.uniform(-25,25)
    col=GREENS[max(k-1,0)]; hl=GREENS[min(k+1,4)]
    cv2.ellipse(shrub,(x,y),(w,h),ang,0,360,[float(c) for c in col],-1,cv2.LINE_AA)
    cv2.ellipse(shrub,(x-w//5,y-h//3),(int(w*0.72),int(h*0.5)),ang,180,360,[float(c) for c in hl],-1,cv2.LINE_AA)
shrub=cv2.GaussianBlur(shrub,(0,0),1.1)
# rocks: voronoi facets
rockimg=shrub.copy()
RT=[hx('#4D6073'),hx('#6C8196'),hx('#8FA3B4'),hx('#B8C6CF')]
step=84
seeds=[]
for gy in range(0,N+step,step):
    for gx in range(0,N+step,step):
        seeds.append((gx+rng.randint(-34,34),gy+rng.randint(-34,34)))
seeds=np.array(seeds,np.float32)
rys,rxs=np.where(rock)
if len(rys):
    wx=cv2.GaussianBlur(rng.randn(N,N).astype(np.float32),(0,0),28); wx=wx/wx.std()*26
    wy=cv2.GaussianBlur(rng.randn(N,N).astype(np.float32),(0,0),28); wy=wy/wy.std()*26
    P=np.stack([rxs+wx[rys,rxs],rys+wy[rys,rxs]],1).astype(np.float32)
    d=((P[:,None,:]-seeds[None,:,:])**2).sum(2)
    cid=d.argmin(1)
    # light comes from upper-left: facet tone from random normal + vertical position
    ftone=rng.rand(len(seeds))
    lab_r=np.zeros((N,N),np.int32)-1
    lab_r[rys,rxs]=cid
    for i in range(len(seeds)):
        m_=lab_r==i
        if not m_.any(): continue
        sy_=seeds[i,1]
        base=ftone[i]*0.7+ (1-sy_/N)*0.3
        t=int(np.clip(base*4,0,3))
        rockimg[m_]=RT[t]
    # cream light edge on upper boundary of each facet + dark underside
    lab_f=lab_r.astype(np.float32)
    up=np.zeros((N,N),bool); dn=np.zeros((N,N),bool)
    up[1:]=(lab_r[1:]!=lab_r[:-1])&(lab_r[1:]>=0)
    dn[:-1]=(lab_r[:-1]!=lab_r[1:])&(lab_r[:-1]>=0)
    up=cv2.dilate(up.astype(np.uint8),np.ones((3,3),np.uint8))>0
    dn=cv2.dilate(dn.astype(np.uint8),np.ones((4,4),np.uint8))>0
    tone_map=np.zeros((N,N),np.float32)-1
    tone_map[rys,rxs]=np.array([min(int(np.clip((ftone[c]*0.7+(1-seeds[c,1]/N)*0.3)*4,0,3)),3) for c in cid],np.float32)
    above=np.zeros_like(tone_map)-1; above[1:]=tone_map[:-1]
    lit=up&rock&(tone_map>above)&(above>=0)
    rockimg[lit]=hx('#E9E2CF')
    # slight opening between the big rock facets, lower part only
    bnd=np.zeros((N,N),bool)
    d1=(lab_r[:,1:]!=lab_r[:,:-1])&(lab_r[:,1:]>=0)&(lab_r[:,:-1]>=0)&(np.abs(tone_map[:,1:]-tone_map[:,:-1])>=1)
    d2=(lab_r[1:,:]!=lab_r[:-1,:])&(lab_r[1:,:]>=0)&(lab_r[:-1,:]>=0)&(np.abs(tone_map[1:,:]-tone_map[:-1,:])>=1)
    bnd[:,1:]|=d1; bnd[1:,:]|=d2
    lowzone=(yy2>int(N*0.80))
    bnd&=lowzone
    gapm=cv2.dilate(bnd.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))>0
    gapm=cv2.GaussianBlur(gapm.astype(np.float32),(0,0),1.3)>0.5
    # fade in with depth so the opening only grows toward the bottom
    gapm&=(nz(10)>-0.6)
    rockimg[gapm&rock]=shrub[gapm&rock]*0.75
    # moss tufts on rock
    for _ in range(40):
        i=rng.randint(len(rys)); x=rxs[i]; y=rys[i]
        cv2.ellipse(rockimg,(int(x),int(y)),(rng.randint(8,18),rng.randint(5,9)),rng.uniform(-20,20),0,360,[float(c) for c in GREENS[rng.randint(2,4)]],-1,cv2.LINE_AA)
rockimg=cv2.GaussianBlur(rockimg,(0,0),0.9)
ground_img=shrub.copy()
ground_img[rock]=rockimg[rock]
# cover the opened crevices with shrub so they read as vegetation growing between the rocks
gz=(gapm&rock)
gz2=cv2.dilate(gz.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))>0
ground_img[gz]=shrub[gz]
gy_,gx_=np.where(gz2&rock|gz)
rng2=np.random.RandomState(77)
placed=[]
for _ in range(min(2500,len(gy_))):
    j=rng2.randint(len(gy_)); y=int(gy_[j]); x=int(gx_[j])
    dep=float(np.clip((y-ridge[x])/(N-ridge[x]+1e-6),0,1))
    w=max(5,int(rng2.uniform(0.75,1.2)*(5+11*dep)))
    h=int(w*rng2.uniform(0.45,0.65)); kk=rng2.randint(1,4)
    if any((x-px)**2+(y-py)**2<(1.05*max(w,pw))**2 for px,py,pw in placed): continue
    placed.append((x,y,w))
    cv2.ellipse(ground_img,(x,y),(w,h),rng2.uniform(-25,25),0,360,[float(c) for c in GREENS[kk]],-1,cv2.LINE_AA)
    cv2.ellipse(ground_img,(x-w//5,y-h//3),(int(w*0.7),int(h*0.5)),0,180,360,[float(c) for c in GREENS[min(kk+1,4)]],-1,cv2.LINE_AA)
edgeband=(cv2.dilate(rock.astype(np.uint8),np.ones((19,19),np.uint8))>0)&(~rock)&G
eys,exs=np.where(edgeband)
for _ in range(0):
    i=rng.randint(len(eys)); y=eys[i]; x=exs[i]
    w=rng.randint(8,17); h=int(w*rng.uniform(0.55,0.8)); k=rng.randint(1,4)
    cv2.ellipse(ground_img,(int(x),int(y)),(w,h),rng.uniform(-25,25),0,360,[float(c) for c in GREENS[k]],-1,cv2.LINE_AA)
    cv2.ellipse(ground_img,(int(x-w//5),int(y-h//3)),(int(w*0.7),int(h*0.5)),0,180,360,[float(c) for c in GREENS[min(k+1,4)]],-1,cv2.LINE_AA)
out=sky.copy()
out[G]=ground_img[G]
out[runner]=np.array([26,30,56],np.float32)
out=out.astype(np.uint8)
f=out.astype(np.float32)

# ---------- watercolor treatment ----------
def rnd(sig,sx=None,sy=None):
    r=rs.randn(N,N).astype(np.float32)
    r=cv2.GaussianBlur(r,(0,0),sigmaX=sx or sig,sigmaY=sy or sig)
    return r/(r.std()+1e-6)
# 1 layered washes in the sky: translucent overlapping strokes
for _ in range(0):
    cy=rs.randint(0,int(np.median(ridge))); hgt=rs.randint(24,70); x0=rs.randint(-100,600); wid=rs.randint(380,1000)
    col=f[min(cy,N-1),rs.randint(0,N)] if False else None
    layer=np.zeros((N,N),np.float32)
    cv2.rectangle(layer,(x0,cy),(x0+wid,cy+hgt),1.0,-1)
    wob=rnd(0,sx=30,sy=3)*7
    mapy=(np.arange(N)[:,None]+wob).astype(np.float32); mapx=np.tile(np.arange(N,dtype=np.float32),(N,1))
    layer=cv2.GaussianBlur(cv2.remap(layer,mapx,mapy,cv2.INTER_LINEAR),(0,0),sigmaX=10,sigmaY=3)
    sign=rs.choice([-1,1])
    f[:,:,:]+= (layer*(sign*rs.uniform(5,12)))[...,None]*(~G)[...,None]
# 2 brush bristle streaks (horizontal in sky, diagonal in ground)
bristle=rnd(0,sx=70,sy=2.2)
def follow(t): return cv2.remap(t,mapx,mapy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
st1=follow(rnd(0,sx=140,sy=1.2)); st2=follow(rnd(0,sx=70,sy=2.4)); st3=follow(rnd(0,sx=220,sy=3.5))
amp=0.45+0.9*np.clip(nz(80)*0.5+0.5,0,1)   # intensity varies across the sky
bandamp=np.ones(N,np.float32)
for y0 in range(0,N,70): bandamp[y0:y0+70]=rs.uniform(0.6,1.4)
amp=amp*bandamp[:,None]
f*=(1+(0.016*st1+0.010*st2+0.012*st3)*amp*np.where(G,0,1))[...,None]
M=cv2.getRotationMatrix2D((N/2,N/2),-24,1.0)
diag=cv2.warpAffine(rnd(0,sx=34,sy=1.1),M,(N,N),borderMode=cv2.BORDER_REFLECT)
f+=diag[...,None]*np.where(G,2.6,0)[...,None]
# 3 pigment wash variation
wv=np.where(G,1.0,0.25)
f*= (1+0.045*rnd(26)*wv)[...,None]
f*= (1+0.03*rnd(7)*wv)[...,None]
# 4 pooled edges: darker rims where colour regions meet
edge=cv2.Canny(np.clip(f,0,255).astype(np.uint8),30,80)
edge=cv2.GaussianBlur(edge.astype(np.float32),(0,0),2.6)/255.0
f*= (1-0.22*np.clip(edge*2.0,0,1)*np.where(G,1.0,0.5))[...,None]
# 5 paper grain & granulation (more in dark pigment)
grain=cv2.GaussianBlur(rs.randn(N,N).astype(np.float32),(0,0),0.7)
dk=(1-np.clip(f.mean(2)/255,0,1))
f+= (grain*(6+14*dk)*np.where(G,1.0,0.35))[...,None]
f=np.clip(f,0,255)
# soft bleed
fb=cv2.addWeighted(f,0.72,cv2.GaussianBlur(f,(0,0),1.6),0.28,0)
f=np.where(G[...,None],fb,f)
paint=np.clip(f,0,255).astype(np.uint8)
Image.fromarray(paint).save('jn/paint.png')
# colors for background

topc=tuple(int(v) for v in GREENS[2])
botc=tuple(int(v) for v in cols[0])                        # upper sky tone
print(topc,botc)
# layout 1080x1350
CW,CH=1080,1350
cv=Image.new('RGB',(CW,CH),topc)
d=ImageDraw.Draw(cv)
S=840; sy=190
split=int(round(sy+np.median(ridge)*S/N)); print('split',split)
d.rectangle([0,split,CW,CH],fill=botc)
S=840; sx=(CW-S)//2; sy=190
pi=Image.fromarray(paint).resize((S,S),Image.LANCZOS)
cv.paste(pi,(sx,sy))
def spaced(text,y,size,fill,gap,fontpath,center=True,fit=None):
    f=ImageFont.truetype(fontpath,size)
    ws=[d.textlength(c,font=f) for c in text]
    if fit: gap=(fit-sum(ws))/(len(text)-1)
    tot=sum(ws)+gap*(len(text)-1); x=(CW-tot)/2
    for c,w in zip(text,ws):
        d.text((x,y),c,font=f,fill=fill); x+=w+gap
fp='/mnt/user-data/attachments/5a7d1354-b855-5f9c-b6ac-3454f8809492/3ab4a850-WantedSans-Bold.ttf'
cream=(245,238,222)
spaced('KILIAN JORNET',100,44,cream,0,fp,fit=840)
spaced('RICH ROLL PODCAST',1232,26,(36,52,78),12,'/mnt/user-data/attachments/5a7d1354-b855-5f9c-b6ac-3454f8809492/5a27bef6-WantedSans-Medium.ttf')
# rounded corners like reference
mask=Image.new('L',(CW,CH),0); ImageDraw.Draw(mask).rounded_rectangle([0,0,CW-1,CH-1],40,fill=255)
bg=Image.new('RGB',(CW,CH),(12,22,48)); bg.paste(cv,(0,0),mask)
bg.save('jn/sample23.png')
