# Resumo para o grupo — Validação cruzada honesta, baseline do PR-AUC e próximos passos

*Documento-síntese da Sessão 2. Complementos visuais: `session_2_overview.svg` (o que fizemos,
em um diagrama) e `pr_auc_baseline_infographic.svg` (como ler o PR-AUC). Detalhes técnicos em
`session_2_methodology.md` (o porquê) e `session_2_findings.md` (os números).*

**A tarefa, em uma frase:** a partir de uma janela de 40 snapshots (~10 min) das métricas de um
nó GPU, prever se o job está prestes a **falhar**, cedo o suficiente para drenar o nó. Falhas
são **raras** (~4% das janelas), o que torna os números pequenos — daí a importância do baseline.

**Vocabulário:** *Sessão 1* = o benchmark original = **Pass 1** (treinar 15 modelos uma vez para
ranquear) + **Pass 2** (a **busca em grade / grid search** dos melhores hiperparâmetros).
*Sessão 2* (este trabalho) não mudou modelos nem dados — corrigiu **como julgamos** os resultados
e **refez o Pass 2** com uma validação cruzada honesta.

---

## 1. PR-AUC: qual o baseline e o que diferentes valores representam

- **Baseline = 0,0403** = a **prevalência de positivos no conjunto de TESTE** (a taxa de falha
  entre as janelas, ~1 em 25).
- **Por quê:** um classificador aleatório/constante tem, em qualquer recall, precisão igual à
  fração de positivos → sua curva precisão-recall é uma **linha horizontal** em `y = prevalência`
  → a área sob ela (que é o PR-AUC) **é a prevalência**. Logo, todo modelo precisa superar 0,0403.
- **Cuidado (a confusão que corrigimos):** o `mean_proba` = **0,0320** do Dummy é o *prior de
  treino* (o que ele prevê); o `pr_auc` = **0,0403** é o que ele *atinge no teste* = o baseline.
  Usar 0,0320 superestimaria os resultados. **Baseline do ROC-AUC = 0,5.**
- **O que é um "positivo":** é o **rótulo de JANELA** (voto majoritário: ≥20 dos 40 snapshots),
  ou seja, uma janela nas **2 h finais de um job que falhou**. **Não** é o estado do job e **não**
  é o rótulo de um snapshot isolado. (Por isso um TIMEOUT longo tem só ~10% de janelas positivas —
  apenas a cauda — enquanto jobs OOM curtos têm ~99%.)
- **O que diferentes valores representam** (leia como múltiplo, "×-lift", de 0,0403 — cada passo
  dobra):

  | ×-lift | PR-AUC | faixa | onde estamos |
  |---:|---:|---|---|
  | 1× | 0,040 | ~aleatório | Dummy |
  | 1,2–2× | 0,048–0,081 | fraco | base da Sessão 1 (~1,97×) |
  | 2–4× | 0,081–0,161 | moderado | **melhor modelo tunado (2,13×)** |
  | 4–8× | 0,161–0,322 | forte | — |
  | >8× | >0,322 | excelente | — |
  | 24,8× | 1,000 | perfeito | — |

  **Resumo:** nossos modelos estão em **~2× o baseline** — um sinal **real, porém modesto**
  (coerente com ROC-AUC ~0,62).

---

## 2. Qual era o problema de cross-validation no grid search

- **Onde a CV entra:** *apenas no Pass 2* (o grid search). O Pass 1 treina uma vez e testa uma
  vez — sem CV. No Pass 2, o grid search divide o **conjunto de treino** em partes (folds) para
  decidir quais hiperparâmetros são melhores. O **conjunto de teste nunca entra na CV**, então
  ele permanece honesto.
- **A CV usada:** `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)` — a escolha
  **padrão** para classificação. Ninguém escolheu algo "vazado" de propósito; a padrão é que é
  **errada para o nosso tipo de dado**.
- **Por que vazou:** nossas linhas são **janelas sobrepostas** (cada janela compartilha 39 de
  seus 40 snapshots com a seguinte). Com divisão **aleatória**, janelas quase idênticas do
  **mesmo job** caem nos **dois lados** da divisão → a validação "vê a resposta" → o escore de CV
  fica **falsamente alto** (**0,46** vs. **0,08** real).
- **A consequência:** confiando nesse 0,46, o grid search escolheu os modelos **mais complexos**,
  que fazem *overfit* e pioram no teste → a **regressão** do Pass 2 (LightGBM 0,079 → 0,061;
  CatBoost 0,080 → 0,070).

---

## 3. A nova estratégia de cross-validation proposta

**Princípio:** a divisão precisa respeitar duas coisas que o embaralhamento ignora — **a qual
job** a linha pertence e **quando** ela ocorreu. Importante: estas são **regras de divisão, não
modelos** (os modelos continuam LightGBM/CatBoost/HGB).

| Estratégia | O que garante |
|---|---|
| **GroupKFold** (grupo = `slurm_id`) | todas as janelas de um job ficam de um lado só (mata a sobreposição) |
| **TimeSeriesSplit** (gap=39) | treina no **passado**, valida no **futuro**; o gap purga a emenda |
| **PurgedGroupKFold** | blocos temporais + jobs inteiros + **purga** jobs que se sobrepõem no tempo à validação (+ embargo de 2 h) |
| **LeaveOneGroupOut semanal** | deixa uma semana inteira de fora por vez |

**Descoberta central (o ponto forte para a banca):**
1. **Agrupar não basta.** O `GroupKFold` ainda superestima **2,7×** — ele remove a sobreposição
   do mesmo job, mas não o vazamento **temporal**/de jobs concorrentes. Só respeitando o **TEMPO**
   a CV fica honesta (TimeSeriesSplit ~**1,1×**, PurgedKFold ~**1,5×**).
2. **Para escolher a complexidade, é preciso validar no TEMPO.** Um monitor do *mesmo período*
   (mesmo sem vazamento de janelas) pede **árvores demais** (overfit no futuro); só um monitor
   **temporal** escolhe o número de árvores que generaliza para frente.

**Escolha:** `PurgedGroupKFold` como padrão (a mais rigorosa — única que também purga
concorrência) e `TimeSeriesSplit` como a mais fiel ao cenário real de uso (passado → futuro). A
`StratifiedKFold(shuffle)` original é a única **inadequada**.

---

## 4. Resumo da nova implementação e resultados

**Entradas e saídas (visão de alto nível).** O sistema **reaproveita o conjunto já janelado da
Sessão 1** — as **janelas de 32 features**, o **rótulo de cada janela** (falha / não-falha), o
**id do job** e o **estado** de cada janela — e acrescenta **uma única informação nova: o horário
(fim) de cada janela**, necessária para as validações que respeitam o tempo. A partir disso, produz
como **saídas**: (1) o **baseline exato** e a distribuição das classes; (2) o **diagnóstico** de
quanto cada estratégia de validação distorce o escore; (3) os **modelos re-ajustados**, com seus
hiperparâmetros e desempenho no teste; e (4) o **ponto de operação** e a **calibração** recomendados.
Nada do pipeline de dados original é alterado.

**Como funciona, em três etapas:**

- **Verificações anti-vazamento.** Cada estratégia de validação passa por checagens automáticas que
  *provam* que não há vazamento: nenhum job aparece dos dois lados de uma divisão; no caso temporal,
  o treino é sempre anterior à validação; e nada que se sobreponha no tempo ao trecho de validação
  entra no treino.

- **Diagnóstico.** Rodamos a mesma busca sob cada estratégia e comparamos o que a validação
  *acredita* (CV-AP) com a *verdade* (PR-AUC no teste):

  | estratégia de validação | CV-AP (o que acredita) | teste (verdade) | inflação |
  |---|---:|---:|---:|
  | aleatória (a padrão antiga) | 0,463 | 0,081 | **5,7×** |
  | por job (GroupKFold) | 0,215 | 0,081 | 2,7× |
  | temporal purgada (PurgedKFold) | 0,120 | 0,081 | 1,5× |
  | **temporal (tscv)** | 0,088 | 0,081 | **1,1×** |

  > **O que é *tscv*?** É a abreviação de **TimeSeriesSplit** ("divisão para séries temporais"):
  > treina sempre no **passado** e valida no **futuro**, com uma pequena folga na emenda para não
  > misturar janelas vizinhas. É a que mais se aproxima do uso real do modelo (prever o futuro a
  > partir do passado) e, por isso, a que menos distorce o escore.

- **Re-ajuste honesto.** Escolhemos os hiperparâmetros e o número de árvores por **validação
  temporal**: separamos os *últimos* jobs do treino como conjunto de conferência (nunca o teste) e
  paramos de adicionar árvores quando ele deixa de melhorar. Confirmamos a estabilidade com a
  validação temporal purgada, treinamos no conjunto de treino **completo** e avaliamos no **teste
  temporal** (jobs posteriores, intocados).

**Resultados:**

| Modelo | PR-AUC tunado | base (Sessão 1) | Δ | ×-lift |
|---|---:|---:|---:|---:|
| **CatBoost** | **0,0860** | 0,0795 | **+0,0066** | **2,13×** |
| LightGBM | 0,0782 | 0,0793 | −0,0011 | 1,94× |
| HGB | 0,0699 | 0,0752 | −0,0053 | 1,73× |

- **A regressão do Pass 2 foi revertida:** de **−0,018** (piora) para **+0,0066** (melhora) — os
  mesmos modelos, tunados sob uma CV válida, agora **melhoram** em vez de piorar.
- **Ensemble não ajudou** (média 0,0840 < CatBoost sozinho 0,0860) → usar o **melhor modelo
  único**.
- **Calibração:** os escores brutos são mal calibrados (Brier 0,276); a calibração **sigmoid**
  reduz para 0,038 **preservando o PR-AUC** (a isotônica muda levemente por empates) → usar
  **sigmoid**.
- **Ponto de operação (CatBoost, favorecendo recall, F2):** limiar 0,54 → recall **0,56**,
  precisão 0,062; ou recall **0,80** com limiar 0,38. Precisão baixa é esperada no desbalanceamento
  extremo — o valor está em **capturar falhas cedo**, não em precisão perfeita.
- **Por estado (physics):** OOM ~0,66 e TIMEOUT ~0,52 são os mais previsíveis (falhas graduais);
  **NODE_FAIL ~0,40** é o mais difícil (falha instantânea, sem rastro nas métricas).
- **Comparação com Skrzeczek (0,98 ROC-AUC):** **não é comparável** — ele classifica **jobs
  inteiros** (agregados), ~920 jobs, desbalanceamento leve (~2,6:1), com o job **bem-sucedido**
  como classe positiva. Nossa tarefa é por **janela**, **extremamente desbalanceada** e de
  **previsão antecipada** da falha rara — mais difícil por construção.

**Modelo recomendado para a tese:** **CatBoost** (depth 8, lr 0,1, ~18 árvores,
`auto_class_weights='Balanced'`), reportado no **ponto de operação F2** com probabilidades
**calibradas por sigmoid**. **PR-AUC 0,0860 = 2,13× o baseline.**

---

## 5. Conclusão e próximos passos: mudança de features / pipeline

**Conclusão.** Os maiores ganhos desta sessão foram **metodológicos**: o baseline correto
(0,0403), o diagnóstico do vazamento de CV e a descoberta de que a validação precisa ser
**temporal** (não só agrupada). O sinal é **real, mas modesto** (~2×). Para ultrapassar isso de
forma significativa, o próximo passo **não é mais tuning** — é mexer no **pipeline de dados**.
A boa notícia: agora temos a **infraestrutura honesta** (CV temporal/purged + baseline) para
medir cada mudança **sem nos enganarmos de novo**.

**Propostas, em ordem de custo-benefício:**

1. **[Maior impacto] Features de dinâmica temporal dentro da janela.** Hoje usamos apenas
   min/max/mean/std — que **descartam a trajetória** dentro da janela. Falhas graduais aparecem
   como **tendências** (memória subindo, temperatura subindo). Adicionar por métrica:
   - **inclinação (slope)** de uma reta ajustada na janela (sobe/desce?);
   - **delta** (último − primeiro) e **último valor** (estado atual, não só o resumo);
   - **EWMA** (média com peso maior nos snapshots recentes) e **percentis** (mediana, p90/p95).
   É a mudança que mais deve mexer no número, e ataca justamente as falhas previsíveis.

2. **Rever e expandir o conjunto de métricas.** As 8 atuais vieram do Skrzeczek, escolhidas para
   uma tarefa **diferente** (job inteiro). O dataset tem ~90 métricas. Incluir sinais de
   **hardware/erro da GPU** — **ECC/Xid errors, throttling, clocks, utilização, erros de PCIe** —
   que são precursores clássicos de falha de hardware e podem dar sinal ao **NODE_FAIL** (hoje
   quase cego). Fazer uma **seleção de features por importância** em vez de herdar a lista antiga.

3. **Normalização por nó (features relativas ao "normal" do próprio nó).** 80 °C significam
   coisas diferentes em GPUs diferentes; o sinal de saúde é o **desvio** em relação à linha de base
   histórica **daquele nó** (z-score por nó, ou diferença para a mediana do nó). Torna o sinal
   comparável entre hardwares heterogêneos.

4. **Repensar a definição da classe positiva.** Hoje o **TIMEOUT domina 70%** dos positivos — mas
   timeout frequentemente é o **usuário subestimando o walltime**, não um problema de saúde do nó.
   Considerar **separar falhas de hardware** (FAILED/OOM/NODE_FAIL) do TIMEOUT, e tratar
   **NODE_FAIL honestamente** (talvez reportá-lo como "não previsível a partir das métricas"). Isso
   reduz **ruído de rótulo** e alinha a tarefa a "saúde do nó".

5. **Experimentos de janela e horizonte.** Testar **tamanhos de janela** múltiplos (multi-escala:
   10 min + 1 h), **horizonte de previsão** mais curto (o sinal é mais forte perto da falha) e
   **stride maior** (janelas menos sobrepostas → menos redundância e ataca a **própria fonte** do
   vazamento na raiz).

**Como vamos medir.** Cada mudança será avaliada com a **CV temporal/purged** e comparada ao
**baseline 0,0403** (em ×-lift), no mesmo conjunto de teste temporal — para que qualquer ganho
seja **honesto e defensável**.
