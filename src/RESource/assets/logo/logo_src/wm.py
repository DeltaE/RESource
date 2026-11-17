"""Wordmark to SVG outlines (Outfit, OFL). Author: EL."""
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
import functools, os
@functools.lru_cache(None)
def inst(name,w):
    f=TTFont(os.path.join(os.path.dirname(os.path.abspath(__file__)),f"{name}.ttf"))
    axes={a.axisTag for a in f["fvar"].axes}
    loc={"wght":w}
    if "opsz" in axes: loc["opsz"]=32
    return instantiateVariableFont(f,loc)
def runs(name,parts,track=0):
    # parts: [(text,weight,color)] -> list of (pathd,color), total width (font units), upm
    out=[];x=0
    for text,w,col in parts:
        f=inst(name,w); gs=f.getGlyphSet(); cmap=f.getBestCmap(); hm=f["hmtx"]
        for ch in text:
            g=cmap[ord(ch)]; out.append((g,x,col,f,gs)); x+=hm[g][0]+track
    return out,x-track,inst(name,parts[0][1])["head"].unitsPerEm
def wordmark(name,parts,cx=260,base=500,width=400,track=0):
    glyphs,tw,upm=runs(name,parts,track)
    s=width/tw; x0=cx-width/2
    by={}
    for g,x,col,f,gs in glyphs:
        pen=SVGPathPen(gs); tp=TransformPen(pen,(s,0,0,-s,x0+x*s,base))
        gs[g].draw(tp); by.setdefault(col,[]).append(pen.getCommands())
    return "".join(f'<path fill="{c}" d="{" ".join(d)}"/>' for c,d in by.items())
