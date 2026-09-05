"""Gera uma apresentacao de slides autocontida (HTML) em pt-BR.

Narrativa: como ler o PR-AUC (infografico) -> Sessao 1 sem grid search (Pass 1) ->
Sessao 1 com grid search (Pass 2, a regressao) -> o gancho da validacao cruzada ->
Sessao 2 (o que foi feito + resultados) -> proximos passos (mudar o pipeline).

Reutiliza as figuras/paleta de `report.py` e os SVGs de `docs/`. Navegacao por
setas do teclado, clique nos botoes, ou hash na URL.

    python -m cv_strategies.slides   ->  results/slides.html
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import report as R


# --------------------------------------------------------------------------- #
# figuras (titulos em pt-BR; reusam _mpl/_b64/paleta de report.py)
# --------------------------------------------------------------------------- #
def fig_lift(imb, base):
    plt = R._mpl()
    d = pd.DataFrame(imb["session1_lift"])[["model", "pr_auc"]].sort_values("pr_auc").tail(8)
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(y, d["pr_auc"], color=R.BLUE, height=.62)
    for yi, v in zip(y, d["pr_auc"]):
        ax.text(v + .0012, yi, f"{v:.3f}  ({v/base:.1f}x)", va="center", fontsize=9, color=R.INK)
    ax.axvline(base, color=R.RED, ls="--", lw=1.2)
    ax.text(base, -0.7, f"baseline {base:.3f}", color=R.RED, fontsize=9, ha="left")
    ax.set_yticks(y); ax.set_yticklabels([m.replace("Classifier", "") for m in d["model"]], fontsize=9)
    ax.set_xlabel("PR-AUC (teste)"); ax.set_xlim(0, max(d["pr_auc"]) * 1.28)
    ax.set_title("Pass 1: PR-AUC vs. baseline correto", color=R.INK, fontsize=13, loc="left")
    return R._b64(fig)


def fig_regression(base):
    plt = R._mpl()
    df = pd.read_csv(C.BENCH_RESULTS_CSV)
    models = ["LightGBM", "CatBoost"]
    b = [float(df[(df.model == m) & (df["pass"] == 1)].pr_auc.iloc[0]) for m in models]
    p = [float(df[(df.model == m) & (df["pass"] == 2)].pr_auc.iloc[0]) for m in models]
    x = np.arange(len(models)); w = .36
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w/2, b, w, color=R.MUTED, label="Pass 1 (base)")
    ax.bar(x + w/2, p, w, color=R.RED, label="Pass 2 (grid search)")
    for xi, v in zip(x - w/2, b): ax.text(xi, v + .001, f"{v:.3f}", ha="center", fontsize=9, color=R.INK)
    for xi, v in zip(x + w/2, p): ax.text(xi, v + .001, f"{v:.3f}", ha="center", fontsize=9, color=R.INK)
    ax.axhline(base, color=R.MUTED, ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(models); ax.set_ylabel("PR-AUC (teste)")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Pass 2: o ajuste PIOROU os modelos", color=R.INK, fontsize=13, loc="left")
    return R._b64(fig)


def fig_cv(base):
    plt = R._mpl()
    cv = pd.read_csv(C.CV_COMPARISON_CSV).sort_values("cv_ap_mean")
    y = np.arange(len(cv)); h = .38
    fig, ax = plt.subplots(figsize=(8, 3.9))
    ax.barh(y + h/2, cv["cv_ap_mean"], h, color=R.BLUE, label="CV-AP (o que acredita)")
    ax.barh(y - h/2, cv["test_pr_auc"], h, color=R.ORANGE, label="teste (verdade)")
    for yi, v in zip(y + h/2, cv["cv_ap_mean"]): ax.text(v + .006, yi, f"{v:.3f}", va="center", fontsize=8.5, color=R.INK)
    for yi, v in zip(y - h/2, cv["test_pr_auc"]): ax.text(v + .006, yi, f"{v:.3f}", va="center", fontsize=8.5, color=R.INK)
    ax.axvline(base, color=R.MUTED, ls="--", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(cv["strategy"]); ax.set_xlabel("average precision")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.set_title('Quanto cada CV "trapaceia" (CV-AP vs. teste)', color=R.INK, fontsize=13, loc="left")
    return R._b64(fig)


def fig_temporal():
    plt = R._mpl(); d = R.TEMPORAL_SELECT
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(d["rounds"], d["test_ap"], color=R.BLUE, lw=2, marker="o", ms=5, label="PR-AUC (teste)")
    ax.axvline(d["temporal_argmax"], color=R.GOOD, lw=1.8, label="monitor temporal -> 120")
    ax.axvline(d["random_argmax"], color=R.RED, ls="--", lw=1.8, label="monitor mesmo-periodo -> 260")
    ax.set_xlabel("numero de arvores"); ax.set_ylabel("PR-AUC (teste)")
    ax.legend(frameon=False, fontsize=9, loc="lower center")
    ax.set_title("So o monitor temporal acerta a complexidade", color=R.INK, fontsize=13, loc="left")
    return R._b64(fig)


def fig_tuned(base):
    plt = R._mpl()
    t = pd.read_csv(C.TUNING_RESULTS_CSV); t = t[t.status == "ok"]
    x = np.arange(len(t)); w = .38
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.bar(x - w/2, t["base_pr_auc"], w, color=R.MUTED, label="Sessao 1 (base)")
    ax.bar(x + w/2, t["test_pr_auc"], w, color=R.BLUE, label="Sessao 2 (tunado)")
    for xi, v in zip(x - w/2, t["base_pr_auc"]): ax.text(xi, v + .001, f"{v:.3f}", ha="center", fontsize=8.5, color=R.INK)
    for xi, v in zip(x + w/2, t["test_pr_auc"]): ax.text(xi, v + .001, f"{v:.3f}", ha="center", fontsize=8.5, color=R.INK)
    ax.axhline(base, color=R.RED, ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([m.replace("Classifier", "") for m in t["model"]], fontsize=9)
    ax.set_ylabel("PR-AUC (teste)"); ax.legend(frameon=False, fontsize=9)
    ax.set_title("Sessao 2: ajuste honesto reverte a regressao", color=R.INK, fontsize=13, loc="left")
    return R._b64(fig)


# --------------------------------------------------------------------------- #
# montagem
# --------------------------------------------------------------------------- #
def _img(src, cap=""):
    c = f"<figcaption>{cap}</figcaption>" if cap else ""
    return f'<figure class="fig"><img src="{src}">{c}</figure>'


def _svg(svg, cap=""):
    c = f"<figcaption>{cap}</figcaption>" if cap else ""
    return f'<figure class="fig">{svg}{c}</figure>'


def _slide(inner):
    return f'<section class="slide"><div class="wrap">{inner}</div></section>'


CSS = """
*{box-sizing:border-box}
:root{--bg:#f4f5f3;--fg:#0b0b0b;--mut:#52514e;--line:#d9d8d2;--card:#fcfcfb;--accent:#2a78d6;--bad:#d03b3b;--good:#0a7a0a}
@media(prefers-color-scheme:dark){:root{--bg:#0e0e0d;--fg:#f4f4f2;--mut:#c3c2b7;--line:#2c2c2a;--card:#161615;--accent:#3987e5;--bad:#e66767;--good:#0ca30c}}
html,body{margin:0;height:100%}
body{background:var(--bg);color:var(--fg);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
.deck{position:relative;height:100vh;width:100vw;overflow:hidden}
.slide{position:absolute;inset:0;display:none;padding:4vh 6vw 11vh;overflow:auto}
.slide.active{display:flex;align-items:center;justify-content:center;animation:f .25s ease}
@keyframes f{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.wrap{width:100%;max-width:1040px;margin:auto}
.tag{color:var(--accent);font-weight:650;font-size:clamp(11px,1.3vw,14px);letter-spacing:.04em;text-transform:uppercase;margin-bottom:6px}
h1{font-size:clamp(26px,4.6vw,48px);line-height:1.08;margin:.08em 0}
h2{font-size:clamp(20px,3vw,32px);line-height:1.15;margin:.05em 0 .3em}
.lead{font-size:clamp(14px,1.7vw,20px);max-width:980px;line-height:1.5}
.mut{color:var(--mut)}
.fig{background:#fcfcfb;border:1px solid var(--line);border-radius:14px;padding:14px;max-width:min(960px,90vw);margin:12px auto 6px}
.fig img,.fig svg{width:100%;height:auto;display:block;border-radius:6px}
.fig figcaption{color:#52514e;font-size:clamp(11px,1.2vw,13px);margin-top:8px;text-align:center}
ul{font-size:clamp(14px,1.55vw,19px);line-height:1.5;margin:.4em 0;padding-left:1.15em}
li{margin:.26em 0}
.big{font-size:clamp(34px,7vw,72px);font-weight:750;color:var(--accent);line-height:1}
.row{display:flex;gap:22px;flex-wrap:wrap;align-items:center;margin-top:6px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 18px;line-height:1.5}
.hl{color:var(--bad);font-weight:650}.good{color:var(--good);font-weight:650}
.prog{position:fixed;top:0;left:0;height:3px;background:var(--accent);width:0;transition:width .2s;z-index:10}
.foot{position:fixed;left:0;right:0;bottom:0;height:7vh;min-height:46px;display:flex;align-items:center;justify-content:space-between;padding:0 18px;background:var(--card);border-top:1px solid var(--line);font-size:13px;color:var(--mut);z-index:10}
.nav{display:flex;gap:8px;align-items:center}
.btn{cursor:pointer;border:1px solid var(--line);background:transparent;color:var(--fg);border-radius:8px;padding:6px 13px;font-size:14px}
.btn:hover{border-color:var(--accent);color:var(--accent)}
.counter{min-width:56px;text-align:center;font-variant-numeric:tabular-nums}
"""

JS = """
const S=[...document.querySelectorAll('.slide')];let i=0;
const go=n=>{i=Math.max(0,Math.min(S.length-1,n));
S.forEach((s,k)=>s.classList.toggle('active',k===i));
document.querySelector('.counter').textContent=(i+1)+' / '+S.length;
document.querySelector('.prog').style.width=((i+1)/S.length*100)+'%';
history.replaceState(null,'',' #'+(i+1));};
addEventListener('keydown',e=>{const k=e.key;
if(k==='ArrowRight'||k==='ArrowDown'||k===' '||k==='PageDown'){e.preventDefault();go(i+1);}
else if(k==='ArrowLeft'||k==='ArrowUp'||k==='PageUp'){e.preventDefault();go(i-1);}
else if(k==='Home'){go(0);}else if(k==='End'){go(S.length-1);}});
document.querySelector('.next').onclick=()=>go(i+1);
document.querySelector('.prev').onclick=()=>go(i-1);
go((parseInt(location.hash.slice(1))||1)-1);
"""


def build() -> None:
    C.ensure_dirs()
    import json
    base = R.E.baseline()
    imb = json.loads(C.IMBALANCE_JSON.read_text())
    infographic = (C.DOCS_DIR / "pr_auc_baseline_infographic.svg").read_text(encoding="utf-8")
    overview = R._overview_svg()

    slides = []

    slides.append(_slide(
        '<div class="tag">TCC &middot; UNIFEI &middot; Eng. Eletronica</div>'
        '<h1>Previsao proativa de falhas em nos HPC</h1>'
        '<p class="lead">Validacao cruzada honesta, baseline do PR-AUC e proximos passos.<br>'
        '<span class="mut">De uma janela de ~10 min de metricas de um no GPU, prever se o job vai '
        'falhar &mdash; cedo o suficiente para drenar o no.</span></p>'))

    slides.append(_slide(
        '<div class="tag">Como ler o resultado</div>'
        '<h2>O que significa um PR-AUC &ldquo;bom&rdquo;?</h2>'
        + _svg(infographic, "Baseline = 0,0403 = taxa de falha no teste. Tudo se mede em relacao a isso.")
        + '<p class="lead">Falhas sao raras (~4%). Um chute aleatorio ja &ldquo;acerta&rdquo; 0,0403 '
          '&mdash; todo modelo precisa superar isso.</p>'))

    slides.append(_slide(
        '<div class="tag">Sessao 1 &middot; sem ajuste</div>'
        '<h2>Pass 1 &mdash; 15 modelos, configuracao padrao</h2>'
        + _img(fig_lift(imb, base), "PR-AUC de teste vs. o baseline correto, com o multiplo (x).")
        + '<ul><li>Boosting por histograma vence: <b>CatBoost / LightGBM &asymp; 0,079 &asymp; 2x</b> o baseline.</li>'
          '<li>Sem validacao cruzada e sem tuning: treina uma vez, testa uma vez.</li></ul>'))

    slides.append(_slide(
        '<div class="tag">Sessao 1 &middot; com ajuste</div>'
        '<h2>Pass 2 &mdash; busca de hiperparametros (grid search)</h2>'
        + _img(fig_regression(base), "Base (Pass 1) vs. tunado (Pass 2), com a linha do baseline.")
        + '<p class="lead">Esperavamos melhora. <span class="hl">Piorou.</span> '
          'LightGBM 0,079&rarr;0,061; CatBoost 0,080&rarr;0,070. <b>Por que?</b></p>'))

    slides.append(_slide(
        '<div class="tag">O gancho</div>'
        '<h2>Por que o ajuste piorou? A validacao cruzada vazou</h2>'
        + _svg(R._fold_svg(), "Validacao cruzada: gira qual bloco fica de fora para avaliar.")
        + '<p class="lead">O grid search usa CV para comparar configuracoes. Com divisao '
          '<b>aleatoria</b> e nossas <b>janelas sobrepostas</b> (39/40), janelas quase identicas do '
          'mesmo job caem nos <b>dois lados</b> &rarr; escore inflado (<span class="hl">0,46</span> '
          'vs 0,08 real) &rarr; escolheu modelos complexos demais. <b>&rarr; Sessao 2: consertar isso.</b></p>'))

    slides.append(_slide(
        '<div class="tag">Sessao 2</div>'
        '<h2>O que fizemos</h2>'
        + _svg(overview, "Da regressao ao modelo recomendado, em quatro passos.")))

    slides.append(_slide(
        '<div class="tag">Sessao 2 &middot; evidencia</div>'
        '<h2>Quanto cada estrategia &ldquo;trapaceia&rdquo;</h2>'
        + _img(fig_cv(base), "CV-AP (o que a CV acredita) vs. o PR-AUC de teste (a verdade).")
        + '<ul><li>Aleatoria (padrao): <span class="hl">5,7x</span> inflado &mdash; a causa da regressao.</li>'
          '<li>Agrupar por job ajuda (2,7x) &mdash; mas nao basta.</li>'
          '<li>So respeitar o <b>TEMPO</b> e honesto: <b>tscv 1,1x</b>, purged 1,5x.</li></ul>'))

    slides.append(_slide(
        '<div class="tag">Sessao 2 &middot; descoberta</div>'
        '<h2>Validar no tempo &mdash; nao so agrupar</h2>'
        + _img(fig_temporal(), "PR-AUC de teste vs. numero de arvores; onde cada monitor aponta.")
        + '<p class="lead">Para escolher a complexidade (n&ordm; de arvores), um monitor do '
          '<b>mesmo periodo</b> pede arvores demais (overfit no futuro). So um monitor '
          '<b>temporal</b> acerta.</p>'))

    slides.append(_slide(
        '<div class="tag">Sessao 2 &middot; resultado</div>'
        '<h2>O ajuste honesto reverte a regressao</h2>'
        + _img(fig_tuned(base), "Sessao 1 (base) vs. Sessao 2 (tunado com CV honesta).")
        + '<div class="row"><div class="big">0,0860</div>'
          '<div class="card lead"><b>CatBoost</b> &middot; <b>2,13x o baseline</b><br>'
          '+0,0066 vs. o melhor da Sessao 1<br>'
          'a regressao do Pass 2 virou <span class="good">+0,0066</span></div></div>'))

    slides.append(_slide(
        '<div class="tag">Recomendacao</div>'
        '<h2>Modelo para a tese</h2>'
        '<ul><li><b>CatBoost</b> (depth 8, lr 0,1, ~18 arvores).</li>'
        '<li>Ponto de operacao <b>F2</b>: recall <b>0,56</b> (favorece capturar falhas).</li>'
        '<li>Probabilidades <b>calibradas (sigmoid)</b> &mdash; Brier 0,28 &rarr; 0,04, sem mudar o PR-AUC.</li>'
        '<li>Ensemble <b>nao</b> ajudou &rarr; usar o melhor modelo unico.</li></ul>'))

    slides.append(_slide(
        '<div class="tag">Proximos passos</div>'
        '<h2>Para subir de patamar: mudar o pipeline de dados</h2>'
        '<p class="lead">Tuning esgotado (~2x). O proximo salto esta nas <b>features</b> &mdash; '
        'agora medivel com a CV honesta + baseline 0,0403.</p>'
        '<ul>'
        '<li><b>1. Dinamica temporal</b> na janela (inclinacao, delta, EWMA, percentis) &mdash; '
        'captura a trajetoria que min/max/mean/std descartam. <i>Maior impacto.</i></li>'
        '<li><b>2. Mais/novas metricas</b>: erros de hardware da GPU (ECC/Xid, throttling, PCIe) '
        '&rarr; atacar o NODE_FAIL (hoje quase cego).</li>'
        '<li><b>3. Normalizacao por no</b> (desvio do &ldquo;normal&rdquo; do proprio no).</li>'
        '<li><b>4. Repensar a classe positiva</b>: separar falhas de hardware do TIMEOUT '
        '(70% dos positivos, muitas vezes o usuario, nao o no).</li>'
        '<li><b>5. Janela/horizonte</b>: multi-escala, horizonte mais curto, stride maior '
        '(reduz redundancia e o vazamento na raiz).</li></ul>'))

    slides.append(_slide(
        '<div class="tag">Resumo</div>'
        '<h2>Mensagem final</h2>'
        '<ul><li>Os maiores ganhos foram <b>metodologicos</b>: baseline correto + CV honesta (temporal).</li>'
        '<li>Sinal <b>real, mas modesto</b> (~2x o baseline).</li>'
        '<li>O proximo salto esta nas <b>features</b> &mdash; com a base para medir sem nos enganarmos.</li></ul>'
        '<p class="lead" style="margin-top:18px"><b>Obrigado.</b></p>'))

    body = f'<div class="prog"></div><div class="deck">{"".join(slides)}</div>' + \
        ('<div class="foot"><div>TCC &middot; Previsao proativa de falhas em nos HPC</div>'
         '<div class="nav"><button class="btn prev">&lsaquo; Anterior</button>'
         '<span class="counter"></span><button class="btn next">Proximo &rsaquo;</button></div></div>')

    html = ("<!doctype html><html lang=\"pt-BR\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            "<title>TCC - Slides - Previsao de falhas em nos HPC</title>"
            f"<style>{CSS}</style></head><body>{body}<script>{JS}</script></body></html>")

    out = C.RESULTS_DIR / "slides.html"
    out.write_text(html, encoding="utf-8")
    print("wrote", out, f"({len(slides)} slides)")


if __name__ == "__main__":
    build()
