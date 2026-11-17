"""RESource logo generator. Author: EL.
Writes the animated logo, full, minimal, mark and dark-background SVGs into ../ (src/assets).
Run: python gen_logo.py   (needs fonttools)
"""
import math, random, os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),".."))
random.seed(7)
S3=math.sqrt(3)
CX,CY=260,250
r=22                      # grid cell radius (local units)
SQ=.58; TH=math.radians(30)
def proj(x,y):            # local -> screen under the tilt
    xr=x*math.cos(TH)-y*math.sin(TH); yr=x*math.sin(TH)+y*math.cos(TH)
    return CX+xr, CY+yr*SQ
def ax2px(q,s): return r*S3*(q+s/2), r*1.5*s
HEX=" ".join(f"{math.cos(math.radians(-90+60*k)):.4f},{math.sin(math.radians(-90+60*k)):.4f}" for k in range(6))
def hexpts(k=1.0): return " ".join(f"{k*math.cos(math.radians(-90+60*i)):.4f},{k*math.sin(math.radians(-90+60*i)):.4f}" for i in range(6))

# ---------- palette ----------
LU={"forest":"#4f8a5b","grass":"#b8d26d","crop":"#e3cd78","shrub":"#93b86a","water":"#6fa8d6","urban":"#a39a96","wet":"#7fb7a8"}
EXCL={"water","urban","wet"}
LOGO=[  # (axial q,s), final dir deg, gradient top,bottom, art, landuse
 ((1,-2),-90,"#d3e8a6","#a8d07c","contours","grass"),
 ((2,-1),-30,"#fbea8e","#f1d24f","sun","crop"),
 ((1,1),  30,"#bfe09b","#8cc46c","turbine1","shrub"),
 ((-1,2), 90,"#bccbdb","#93abc6","gusts","grass"),
 ((-2,1),150,"#aec1d9","#87a3c5","streaks2","forest"),
 ((-1,-1),210,"#93c2e3","#6aa1cf","streaks","crop"),
]
RING,RO,RC=165,88,120

# ---------- grid ----------
cells=[]
N=5
for q in range(-N,N+1):
    for s in range(max(-N,-q-N),min(N,-q+N)+1):
        cells.append((q,s))
logo_keys={c[0] for c in LOGO}|{(0,0)}
T=7.2
def pct(t): return f"{100*t/T:.2f}%"

css=[]
css.append(f"""
.tilt{{transform-origin:0 0;animation:tilt {T}s both}}
@keyframes tilt{{0%,{pct(1.0)}{{transform:scaleY(1) rotate(0deg)}}{pct(2.0)},{pct(5.0)}{{transform:scaleY({SQ}) rotate(30deg)}}{pct(6.0)},100%{{transform:scaleY(1) rotate(0deg)}}}}
@keyframes appear{{from{{opacity:0}}to{{opacity:1}}}}
@keyframes drop{{0%{{transform:translate(0,0);opacity:1}}100%{{transform:translate({0.862*140:.1f}px,{1.493*140:.1f}px);opacity:0}}}}
@keyframes fade{{to{{opacity:0;transform:scale(.6)}}}}
@keyframes pop{{0%{{transform:scale(0)}}70%{{transform:scale(1.15)}}100%{{transform:scale(1)}}}}
@keyframes out{{to{{opacity:0}}}}
@keyframes spin{{to{{transform:rotate(360deg)}}}}
.ico{{transform-box:fill-box;transform-origin:50% 100%}}
.rot{{transform-box:fill-box;transform-origin:50% 50%;animation:spin 2.4s linear infinite}}
.rot.slow{{animation-duration:7s}}
.nl{{transform-box:fill-box;transform-origin:50% 50%}}
.wm{{opacity:0;animation:appear .7s 6.3s both}}
@media (prefers-reduced-motion:reduce){{.tilt,.L,.nl,.d,.hx,.ov,.icons,.ico,.wm,.art,.rot{{animation:none!important}}.icons,.gridonly{{display:none}}}}
""")
for k,v in LU.items():
    css.append(f"@keyframes c_{k}{{from{{fill:#c4cbd3}}to{{fill:{v}}}}}")

body=[]
# draw: non-logo cells, then logo outer, then center
grid_svg=[]; icon_svg=[]
survivors=[]
def ring(q,s): return max(abs(q),abs(s),abs(-q-s))
for (q,s) in cells:
    if (q,s) in logo_keys: continue
    x,y=ax2px(q,s)
    lu=random.choices(list(LU),weights=[4,4,3,3,2,1.5,1])[0]
    a_in=0.1+ring(q,s)*0.09
    a_col=2.05+random.random()*0.7
    hx=f'<polygon class="hx" points="{hexpts(.92)}" transform="scale({r})" fill="#c4cbd3" style="animation:appear .45s {a_in:.2f}s both,c_{lu} .45s {a_col:.2f}s both"/>'
    if lu in EXCL:
        d=f"animation:drop .7s {3.1+random.random()*0.6:.2f}s cubic-bezier(.5,0,.9,.6) both"
    else:
        d=f"animation:fade .5s {4.95+random.random()*0.3:.2f}s ease-in both"
        survivors.append((q,s,lu,x,y))
    grid_svg.append(f'<g transform="translate({x:.2f} {y:.2f})"><g class="d{" nl" if lu not in EXCL else ""}" style="{d}">{hx}</g></g>')

def turbine(sc,sw,col="#fff",cls="rot"):
    L=1.0
    bl="".join(f'<path d="M0 0 L{0.09*math.cos(math.radians(a+90)):.3f} {0.09*math.sin(math.radians(a+90)):.3f} L{L*math.cos(math.radians(a)):.3f} {L*math.sin(math.radians(a)):.3f} Z" />' for a in (-90,30,150))
    return (f'<g transform="scale({sc})" stroke="{col}" stroke-width="{sw/sc:.3f}" stroke-linejoin="round" fill="none">'
            f'<path d="M-0.06 2.4 L0 0 L0.06 2.4" fill="{col}" stroke="none"/>'
            f'<g class="{cls}"><circle r="{L}" fill="none" stroke="none"/>{bl}<circle r=".12" fill="{col}"/></g></g>')
def sun(rr,sw,col="#fff",n=8,ray=(1.35,1.8)):
    rays="".join(f'<line x1="{rr*ray[0]*math.cos(a):.3f}" y1="{rr*ray[0]*math.sin(a):.3f}" x2="{rr*ray[1]*math.cos(a):.3f}" y2="{rr*ray[1]*math.sin(a):.3f}"/>' for a in [2*math.pi*i/n for i in range(n)])
    return f'<g stroke="{col}" stroke-width="{sw}" stroke-linecap="round" fill="none"><circle r="{rr}"/>{rays}</g>'


def outer_art(art):
    if art=="turbine1":
        return f'<g opacity=".75"><g transform="translate(.05 -.22)">{turbine(.36,.022,cls="")}</g></g>'
    if art=="sun":
        return f'<circle cx=".12" cy="-.05" r=".42" fill="#fff6b8" opacity=".75"/><g transform="translate(.12 -.05)" opacity=".85">{sun(.42,.025,"#fff",n=16,ray=(1.12,1.5))}</g>'
    if art=="streaks":
        return '<g stroke="#fff" stroke-width=".018" opacity=".55">'+"".join(f'<line x1="-.9" y1="{-.6+i*.17:.2f}" x2=".9" y2="{-.85+i*.17:.2f}"/>' for i in range(9))+'</g>'
    if art=="streaks2":
        return '<g stroke="#fff" stroke-width=".018" opacity=".5">'+"".join(f'<line x1="-.9" y1="{-.85+i*.17:.2f}" x2=".9" y2="{-.55+i*.17:.2f}"/>' for i in range(10))+'</g>'
    if art=="contours":
        return '<g stroke="#fff" stroke-width=".02" fill="none" opacity=".6">'+"".join(f'<path d="M-1 {y:.2f} C-.5 {y-.16:.2f} -.1 {y+.14:.2f} .35 {y-.04:.2f} S .8 {y-.12:.2f} 1 {y-.02:.2f}"/>' for y in [-.45,-.25,-.05,.15,.35,.55])+'</g>'
    if art=="gusts":
        return ('<g stroke="#fff" stroke-width=".024" fill="none" stroke-linecap="round" opacity=".65">'
                '<path d="M-.7 -.25 H.25 a.16 .16 0 1 0 -.16 -.16"/>'
                '<path d="M-.8 .02 H.5 a.2 .2 0 1 1 -.2 .2"/>'
                '<path d="M-.55 .28 H.05"/></g>')
    return ""

# icons on some survivors (screen space, upright)
random.shuffle(survivors)
for i,(q,s,lu,x,y) in enumerate(survivors[:11]):
    sx,sy=proj(x,y)
    t=3.95+i*0.05
    if i not in (2,7):
        ic=(f'<g transform="translate({sx:.1f} {sy+4:.1f})"><g class="ico" style="animation:pop .4s {t:.2f}s both">'
            f'<path d="M-9 -2 L5 -2 L10 -11 L-4 -11 Z" fill="#0d4c81" stroke="#fff" stroke-width="1.2" stroke-linejoin="round"/>'
            f'<path d="M-6.5 -2 L-1.5 -11 M-1.5 -2 L3 -11 M-6.5 -6.5 L7.5 -6.5" stroke="#9cc8f0" stroke-width=".8"/>'
            f'<line x1="1" y1="-2" x2="1" y2="3" stroke="#fff" stroke-width="1.6"/></g></g>')
    else:
        ic=(f'<g transform="translate({sx:.1f} {sy+4:.1f})"><g class="ico" style="animation:pop .4s {t:.2f}s both">'
            f'<g transform="translate(0 -22)">{turbine(9,1.6)}</g></g></g>')
    icon_svg.append(ic)

# logo cells
defs=[]
logo_svg=[]
def logocell(idx,q,s,fx,fy,R,g1,g2,art,lu,center=False):
    x,y=ax2px(q,s)
    name=f"L{idx}"
    css.append(f".{name}{{animation:{name} {T}s both}}@keyframes {name}{{0%,{pct(5.0)}{{transform:translate({x:.2f}px,{y:.2f}px) scale({r})}}{pct(6.05)},100%{{transform:translate({fx:.2f}px,{fy:.2f}px) scale({R})}}}}")
    gid=f"g{idx}"
    if center:
        defs.append(f'<linearGradient id="{gid}" x1=".85" y1="0" x2=".2" y2="1"><stop offset="0" stop-color="#3bb54a"/><stop offset=".45" stop-color="#1c8a78"/><stop offset="1" stop-color="#0d4c81"/></linearGradient>')
    else:
        defs.append(f'<linearGradient id="{gid}" x1="0" y1="0" x2=".3" y2="1"><stop offset="0" stop-color="{g1}"/><stop offset="1" stop-color="{g2}"/></linearGradient>')
    a_in=0.1+ring(q,s)*0.09; a_col=2.05+random.random()*0.7
    hx=f'<polygon class="hx" points="{hexpts(.92)}" fill="#c4cbd3" style="animation:appear .45s {a_in:.2f}s both,c_{lu} .45s {a_col:.2f}s both"/>'
    if center:
        ov=(f'<polygon points="{hexpts(1)}" fill="url(#{gid})" stroke="#fff" stroke-width=".07" stroke-linejoin="round"/>'
            f'<g class="art" style="opacity:0;animation:appear .6s 6.1s both">'
            f'<g transform="translate(-.33 -.27)">{sun(.15,.035)}</g>'
            f'<g transform="translate(.2 -.12)">{turbine(.36,.032,cls="rot slow")}</g></g>')
    else:
        a=outer_art(art)
        ov=(f'<polygon points="{hexpts(.97)}" fill="url(#{gid})"/>'
            f'<clipPath id="c{idx}"><polygon points="{hexpts(.97)}"/></clipPath><g clip-path="url(#c{idx})">{a}</g>')
    logo_svg.append(f'<g class="L {name}">{hx}<g class="ov" style="opacity:0;animation:appear .6s 5.6s both">{ov}</g></g>')

for i,((q,s),ang,g1,g2,art,lu) in enumerate(LOGO):
    fx=RING*math.cos(math.radians(ang)); fy=RING*math.sin(math.radians(ang))
    logocell(i,q,s,fx,fy,RO,g1,g2,art,lu)
logocell(6,0,0,0,0,RC,None,None,None,"forest",center=True)

css.append(".icons{animation:out .35s 4.9s both}")
import sys; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from wm import wordmark
WM='<g aria-hidden="true">'+wordmark("Outfit",[("RES",500,"#1b4f8f"),("ource",400,"#3b4048")],width=380,track=10)+'</g>'

anim=f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 520 520" role="img" aria-label="RESource">
<!-- RESource animated logo. Author: EL. Grid -> perspective -> land use -> exclusions -> solar/wind -> logo. -->
<style>{"".join(css)}</style>
<defs>{"".join(defs)}</defs>
<g transform="translate({CX} {CY})"><g class="tilt">
<g class="gridonly">{"".join(grid_svg)}</g>
{"".join(logo_svg)}
</g></g>
<g class="icons gridonly">{"".join(icon_svg)}</g>
<g class="wm">{WM}</g>
</svg>'''
open("logo_dash_animated.svg","w").write(anim)

# ---------- static, faithful (final frame) ----------
def static_logo(minimal=False, wordmark=True, wm_light=False):
    parts=[]; d=[]
    for i,((q,s),ang,g1,g2,art,lu) in enumerate(LOGO):
        fx=RING*math.cos(math.radians(ang)); fy=RING*math.sin(math.radians(ang))
        if minimal:
            parts.append(f'<polygon points="{hexpts(.95)}" transform="translate({fx:.2f} {fy:.2f}) scale({RO})" fill="{g2}"/>')
        else:
            d.append(f'<linearGradient id="s{i}" x1="0" y1="0" x2=".3" y2="1"><stop offset="0" stop-color="{g1}"/><stop offset="1" stop-color="{g2}"/></linearGradient>')
            a=outer_art(art)
            parts.append(f'<g transform="translate({fx:.2f} {fy:.2f}) scale({RO})"><clipPath id="sc{i}"><polygon points="{hexpts(.97)}"/></clipPath><polygon points="{hexpts(.97)}" fill="url(#s{i})"/><g clip-path="url(#sc{i})">{a}</g></g>')
    d.append('<linearGradient id="sc" x1=".85" y1="0" x2=".2" y2="1"><stop offset="0" stop-color="#3bb54a"/><stop offset=".45" stop-color="#1c8a78"/><stop offset="1" stop-color="#0d4c81"/></linearGradient>')
    if minimal:
        art=(f'<g transform="translate(-.3 -.25)">{sun(.15,.06,n=8,ray=(1.45,1.95))}</g>'
             f'<g transform="translate(.2 -.1)">{turbine(.38,.055,cls="")}</g>')
    else:
        art=(f'<g transform="translate(-.33 -.27)">{sun(.15,.035)}</g><g transform="translate(.2 -.12)">{turbine(.36,.032,cls="")}</g>')
    parts.append(f'<g transform="scale({RC})"><polygon points="{hexpts(1)}" fill="url(#sc)" stroke="#fff" stroke-width=".07" stroke-linejoin="round"/>{art}</g>')
    if wordmark:
        vb="0 0 520 520"; wm=WM if not wm_light else WM.replace("#1b4f8f","#ffffff").replace("#3b4048","#cfd8e3")
    else:
        vb="2 -8 516 516"; wm=""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vb}" role="img" aria-label="RESource">\n'
            f'<!-- RESource logo{" (minimal)" if minimal else ""}. Author: EL. -->\n<defs>{"".join(d)}</defs>'
            f'<g transform="translate({CX} {CY})">{"".join(parts)}</g>{wm}</svg>')
open("logo_dash.svg","w").write(static_logo())
open("logo_dash_mark.svg","w").write(static_logo(minimal=True,wordmark=False))
open("logo_dash_mark_light.svg","w").write(static_logo(minimal=True,wordmark=True,wm_light=True))
open("logo_dash_min.svg","w").write(static_logo(minimal=True,wordmark=True))
print("ok")
