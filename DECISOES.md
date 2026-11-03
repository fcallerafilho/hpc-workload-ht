# Decisões, interpretações e desvios do plano v1.2

Registro das decisões tomadas na implementação do plano "Passo-a-Passo do TCC: ranqueamento de
métricas de telemetria (M100 ExaData) v1.2". O código lê o dado bruto: job table do PM100 e
parquet do M100 ExaData.

Cada entrada tem data e diz **quando** foi fixada em relação aos resultados. As entradas P1 a P10
foram escritas em 16/10/2026, antes de qualquer contagem de completude, divisão ou modelo. P11 e
P12 foram escritas antes de qualquer modelo.

---

## P1 — Coorte de referência · 16/10/2026 · antes da etapa 0

Janela da coorte: `start_time ≥ 2020-05-06` e `end_time ≤ 2020-10-12 23:00` (UTC), tempo de parede
≥ 20 min, sem `CANCELLED`. São os limites de `10_exadata/data/build_summary.json`, que produzem os
69.096 jobs do README citado no plano. Com os limites literais "maio a outubro" (01/05 a 31/10) a
coorte teria 69.207 jobs (111 a mais, todos nas bordas). Usamos os limites do README para que as
contagens de referência do plano sejam comparáveis.

## P2 — Pisos de positivos · 16/10/2026 · antes da etapa 0

Piso = **100 positivos**, por alvo, em cada conjunto de teste (teste principal e teste interno de
cada bloco) e, para as filas, em cada fila (qos 1, qos 11) no teste principal. É a sugestão do
plano. Não será alterado depois.

## P3 — Regra de "dado completo" (decisão 12) · 16/10/2026 · antes da etapa 0

**Janela:** `[início, início + 20 min)`, semiaberta, com início = `start_time`.

**Fração por canal.** Para cada nó do job: `min(n_leituras, esperado) / esperado`, com
`esperado = 1200 s / cadência nominal`. A cadência nominal de cada canal é o intervalo modal entre
leituras consecutivas do mesmo nó, medido no inventário da etapa 0 sobre o dado bruto. A fração do
job é a **média sobre os nós do job** (total de leituras presentes / leituras esperadas de todos os
nós). É coerente com "telemetria = média sobre os nós": um nó ausente num job de 16 nós custa 1/16.

**Canais exigidos:** todos os canais de ganglia e IPMI usados nas features, exceto os constantes
(ver etapa 1).

**Procedimento de escolha (fixado antes de ver qualquer contagem):**

1. **R80** (plano): fração ≥ 0,80 em todos os canais. É a regra principal **se** passar nos dois
   testes abaixo.
   - **T1, dado normal:** para todo canal, a entrega típica de um nó que está reportando (mediana da
     fração entre as janelas nó×job com ≥ 1 leitura) é ≥ 0,80. Se não for, o limiar mede a taxa de
     batidas perdidas do coletor, e não a presença de dado.
   - **T2, viabilidade:** sob a regra, o teste interno de **cada** bloco tem ≥ 100 positivos do alvo
     (a), que é o alvo principal.
2. Se R80 falhar em T1 ou T2: **Rrel**, com limiar por canal = 0,80 × entrega típica do canal ("80 %
   do que um nó reportando normalmente entrega"). É a regra principal se passar em T2.
3. Se Rrel falhar em T2: **R1**, ≥ 1 leitura em todos os canais (fração > 0).

A regra do plano (R80), quando não for a principal, roda como **análise de sensibilidade**
(população `estrito`) nas etapas 2 a 6. A etapa 0 reporta as contagens com 0,60, 0,80 e 1,00, como
o plano pede, além de Rrel e R1.

*Por que T1 e Rrel:* no exame inicial dos arquivos, um canal de 40 s do ganglia (`mem_free`) mostrava
intervalos frequentes de 80 s e 120 s (batidas perdidas). A cadência nominal pode, então,
superestimar o que o coletor entrega num nó normal. T1 verifica isso, e Rrel é um degrau
intermediário antes de R1.

## P4 — Composição do B0 (interpretação) · 16/10/2026 · antes da etapa 3

O plano diz "B0 = qos, metadados de submissão (mem_req, número de nós, time_limit)" e, no critério de
aceite, "só qos e metadados de submissão, incluindo time_limit". Lemos o parêntese como exemplos da
categoria "o que o usuário pede na submissão". B0 = `qos` (one-hot), `partition` (one-hot),
`time_limit`, `mem_req`, `num_nodes_req`, `num_cores_req`, `num_gpus_req`, `num_tasks`,
`cores_per_task`.

Ficam fora: duração real, `start/end_time`, `priority` (calculada pelo escalonador, depende do
histórico do usuário), `user_id`, horário de submissão (funcionaria como relógio), nó, telemetria e
`has_ganglia`. Um B0 mais completo é conservador: dificulta que a telemetria "ganhe" só por
substituir recurso pedido. B0 secundário = B0 + `has_ganglia`.

## P5 — Modelos e hiperparâmetros · 16/10/2026 · antes da etapa 5

Fixados a priori, sem busca, iguais em todas as divisões e populações:

- **HistGradientBoostingClassifier** (principal): `learning_rate=0.05`, `max_iter=300`,
  `max_leaf_nodes=15`, `min_samples_leaf=50`, `l2_regularization=1.0`, `early_stopping=False`,
  `random_state=42`.
- **RandomForestClassifier** (verificação): `n_estimators=500`, `min_samples_leaf=20`,
  `max_features="sqrt"`, `n_jobs=-1`, `random_state=42`. O NaN é tratado nativamente (sklearn ≥ 1.4).

Escore = `predict_proba[:, 1]`. A regularização um pouco maior que o padrão do HGB (folhas menores,
≥ 50 amostras por folha, L2) é uma escolha comum para ~2–3 mil positivos: evita que um grupo de 4
colunas sem sinal seja sobreajustado. Foi fixada antes de qualquer ajuste.

## P6 — Placebo · 16/10/2026 · antes da etapa 3

O placebo *k* (k = 0..19) é uma permutação das linhas (semente 42 + k) das 4 colunas de um grupo
real, feita dentro da população analisada. Para k, o grupo é `GRUPOS[k mod 10]` na análise principal
e `GRUPOS_IPMI[k mod 5]` na secundária. Mantém a distribuição marginal e a estrutura entre as 4
estatísticas, e quebra a associação com o rótulo. O "grupo placebo" da tabela final é o placebo 0.
Os 20 formam o piso de ruído.

## P7 — Importância usada na condição D · 16/10/2026 · antes da etapa 5

**Importância de um grupo g** = queda de PR-AUC no teste principal quando as 4 colunas de g são
permutadas juntas (10 repetições, semente fixa) no modelo B0 + g (HGB). A mesma medida, no modelo
B0 + placebo_k, dá a distribuição dos 20 placebos. D = importância(g) > p95 dessa distribuição. A
permutation importance no modelo com **todos** os grupos (item 6 da etapa 5, permutando juntas as 4
estatísticas de cada métrica e, separadamente, os blocos de métricas correlacionadas com |ρ| ≥ 0,7
de Spearman) é só descritiva.

## P8 — Condições B, C e E (interpretações) · 16/10/2026 · antes da etapa 5

- **B:** os postos usam ΔPR-AUC (HGB) no teste interno de cada bloco, só entre os 10 grupos. W de
  Kendall com correção de empates. "Bloco abaixo do piso" = teste interno do bloco com < 100
  positivos **daquele alvo**.
- **C:** ΔPR-AUC (HGB, modelo treinado em todas as filas) avaliado no teste principal restrito a
  qos 1 e a qos 11. Passa se os dois ganhos têm o mesmo sinal do ganho no teste principal. Se alguma
  das duas filas tiver < 100 positivos do alvo no teste principal, C = "dispensada".
- **E:** ΔPR-AUC do Random Forest no teste principal com o mesmo sinal do ganho do HGB (> 0, já que A
  exige ganho positivo).
- **Relevante:** "sim" se A, B, D e E passam e C passa ou está dispensada; "não conclusiva" se B é
  "não conclusiva" e as demais passam; "não" nos outros casos.

## P9 — Divisões temporais · 16/10/2026 · antes da etapa 4

- **Ordem e cortes:** os jobs da população, com os 5 estados, são ordenados por `end_time` e,
  depois, `job_id`. Os cortes são definidos sobre essa lista. Cada alvo usa o subconjunto dos seus
  estados (alvo a: COMPLETED + FAILED + OUT_OF_MEMORY + NODE_FAIL; alvo b: COMPLETED + TIMEOUT). Os
  dois alvos ficam assim nos mesmos períodos.
- **Principal:** os primeiros 80 % dos jobs são treino, os últimos 20 % teste.
- **Blocos:** 3 blocos consecutivos, cortados nos jobs em que o acumulado de positivos do alvo (a)
  atinge 1/3 e 2/3 do total. Cada bloco tem treino/teste interno 80/20 por número de jobs.
- **Empates:** todo corte avança até não haver `end_time` empatado entre os dois lados.

## P10 — Bootstrap e múltiplas comparações · 16/10/2026 · antes da etapa 5

- **Reamostragem:** 1.000 reamostras de **usuários** com reposição (semente 42). Cada usuário
  sorteado entra com todos os seus jobs do conjunto, e um usuário sorteado k vezes pesa k. Em cada
  reamostra, B0 e B0 + g são avaliados com os **mesmos** pesos. As mesmas reamostras servem para
  todos os grupos.
- **IC:** percentil 2,5–97,5.
- **Conjunto agrupado:** concatenação das predições dos testes internos dos 3 blocos, cada uma vinda
  do modelo do seu bloco.
- **Holm (V7, descritivo):** p unilateral = fração das reamostras com Δ ≤ 0 no conjunto agrupado,
  com correção de Holm sobre os 10 grupos, por alvo e modelo.

## P11 — Como cada leitura da etapa 7 é ativada · 16/10/2026 · antes da etapa 5

O plano dá a tabela de leituras, mas não define "topo" nem quais métricas são "reguladas pelo
hardware". A etapa 7 aplica o que segue mecanicamente, por alvo, na análise principal.

**Classes de métrica.**

- **Reguladas pelo hardware** (seguem malhas de controle térmico): `gpu_temp_nucleo`,
  `gpu_temp_memoria`, `ventoinha`, `temp_ar_entrada`.
- **De carga** (seguem o que o job faz): `mem_ativa`, `swap_free`, `cpu_wio`, `proc_run`, `load_one`,
  `potencia_no`.

**Topo.** Os até 3 grupos de maior ΔPR-AUC (HGB, teste principal) **entre os que têm ΔPR-AUC > 0**.
"Dominado por uma classe" = ao menos 2 grupos do topo são dessa classe. Com menos de 2 grupos de
ganho positivo, não há topo dominado.

**Leituras.**

1. **SNR alto (Ma et al. §3.2.2):** topo dominado por métricas reguladas pelo hardware **e**
   W entre blocos ≥ 0,7.
2. **Emaranhamento com a carga / crítica a Skrzeczek:** topo dominado por métricas de carga, **ou**
   nenhum grupo passa a condição A (ganho nulo sobre B0).
3. **Incompatível com Ma et al. para falha de job:** ao menos um grupo com relevante = "sim".
   Usa a regra inteira, e não só A, B e C, para não chamar de ganho estável um ganho que não supera
   o placebo ou que depende do modelo.
4. **Fragilidade da importância / não conclusivo:** W entre blocos < 0,7 **ou** algum bloco abaixo
   do piso naquele alvo.

Mais de uma leitura pode valer ao mesmo tempo, como o plano prevê.

**Features de Skrzeczek.** Uma feature sobrevive num alvo se tem canal (direto ou proxy) e o grupo
correspondente sai com relevante = "sim". Sem canal: "não testável". Grupo com relevante = "não
conclusiva": "não conclusiva". Nos demais casos: "não".

## P12 — LightGBM como terceiro modelo · 16/10/2026 · antes de qualquer modelo

**Desvio do plano, a pedido do autor.** O plano fixa dois modelos ("três ou mais não se justificam
no escopo de 2 semanas"). O LightGBM foi acrescentado antes da etapa 0 terminar, sem nenhum modelo
ajustado neste repositório.

**Papel:** verificação adicional, **só descritiva**. A regra A–E não muda: o HGB decide e o RF
verifica (condição E). O LightGBM roda onde o RF roda (análise principal e sensibilidade), com as
mesmas divisões, conjuntos e placebos, e aparece em:

- `desempenho.csv` e `ganho_vs_b0.csv` (ganho com os dois ICs por usuário);
- `concordancia_modelos.csv` (V4 estendido: sinal e Spearman HGB × LGBM e RF × LGBM);
- colunas descritivas `delta_lgbm_principal` e `E_lgbm` (mesmo sinal do HGB) na tabela final, fora
  do critério de relevância.

**Por que descritivo.** Tornar o LightGBM uma condição a mais mudaria a regra fixada no plano. Além
disso, o LightGBM é da mesma família do HGB (boosting de árvores por histograma, crescimento por
folha). Como verificação ele é menos independente que o RF: mede sensibilidade à implementação e à
regularização, não à família de modelo.

**Hiperparâmetros (fixos, sem busca):** `n_estimators=300`, `learning_rate=0.05`, `num_leaves=15`,
`min_child_samples=50`, `reg_lambda=1.0`, `subsample=0.8`, `subsample_freq=1`,
`colsample_bytree=0.8`, `random_state=42`, `deterministic=True`, `force_row_wise=True`,
`verbose=-1`. A capacidade é a mesma do HGB (P5). A subamostragem de linhas e colunas é a diferença
prática mais comum no uso do LightGBM. NaN é tratado nativamente.

*O papel do LightGBM foi alterado por P13.*

## P13 — LightGBM passa a contar na condição E · 16/10/2026 · **depois** dos resultados da etapa 6

**Alteração da regra, a pedido do autor, feita depois de ver a tabela final.** Os hiperparâmetros e
os ajustes do LightGBM (P12) não mudam. Muda o papel dele na regra.

- **E (nova):** o ganho no teste principal tem o mesmo sinal do HGB no Random Forest **e** no
  LightGBM. Antes (plano e P8), E exigia só o RF.
- **A, B, C e D:** sem mudança. O HGB continua sendo o modelo que decide.

**Por que a mudança não pode fabricar um resultado.** A nova E é a antiga E mais uma exigência, então
a regra só fica mais restritiva: nenhum grupo pode passar a ser relevante por causa dela. Com os
resultados observados nenhum grupo passa a condição A, e o veredito (nenhum grupo relevante) é o
mesmo nas duas versões.

**Transparência:** a tabela final traz `E_rf`, `E_lgbm`, `E` (as duas juntas) e
`relevante_regra_plano` (a regra original, só com o RF). As duas versões ficam lado a lado. A
etapa 5 foi rodada de novo do zero, com reajuste de todos os modelos, como checagem de
determinismo.
