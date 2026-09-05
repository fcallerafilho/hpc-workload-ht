# Sessão 3 — resumo para o grupo

*Pipeline reconstruído do zero em `pipeline_v2/`. Nada das sessões 1–2 foi reaproveitado; as noções de
tempo que desenvolvemos, sim.*

## O número

| | PR-AUC | baseline aleatório | ganho |
|---|---|---|---|
| Sessão 1 | 0,0795 | 0,0403 | 1,97x |
| Sessão 2 | 0,0860 | 0,0403 | 2,13x |
| **Sessão 3** | **0.1253** | 0.0402 | **3.12x** |

Intervalo de 95 % por bootstrap reamostrando *jobs* (não janelas — janelas do mesmo job não são
independentes): [2.35x, 4.30x].

A população de teste é a mesma das sessões anteriores — mesma divisão 80/20 por data de término do job,
todas as janelas de teste mantidas —, por isso os baselines coincidem e a comparação é legítima.

**Falem "cerca de 3x melhor que o acaso", não um número exato.** A mesma configuração treinada em 85 % do
treino em vez de 100 % dá 0.1382 (3.44x), e o bootstrap
pareado não separa os dois (-0.0097, IC 95 % [-0.0258, +0.0114]).
O valor acima é o pré-registrado — treinado em todo o conjunto de treino. Citar o outro seria escolher o
mais sortudo entre dois ajustes que este experimento não consegue distinguir.

## De onde veio o ganho

1. **Medir mais coisas do nó.** O pipeline antigo usava 8 métricas; este usa 36. Na ablação do notebook 03
   esse passo sozinho vale **+0,059 de PR-AUC** (0,0788 -> 0,1382), e +0,066 somado à conversão para taxas
   — ou seja, praticamente todo o ganho.
2. **Taxas em vez de contadores acumulados.** Um contador acumulado carrega a *idade* do job, não a saúde
   do nó. Converter os três contadores para taxas por segundo valeu +0,006 sozinho.
3. **O resto quase não ajudou.** Derivada e inclinação da janela carregam menos de 1 % do ganho do modelo.
   Normalizar por nó deu 0,0837 e usar a identidade do nó deu 0,1172 — ambos bem abaixo dos 0,1382 do
   conjunto expandido simples. Ajustar hiperparâmetros **piorou** (abaixo).

## O resultado metodológico (o mais importante para o TCC)

Quatro *folds* temporais purgados conseguem **rejeitar**, mas não conseguem **escolher**. Dos vinte
candidatos, quatro saíram comprovadamente piores que o incumbente (os quatro folds concordando, |t| >= 2)
— o conjunto antigo de 8 métricas com folga —, então os folds não são cegos. Mas **nenhum** candidato saiu
comprovadamente melhor: nenhum conjunto top-K, nenhuma das oito configurações de hiperparâmetros, nem
CatBoost nem HistGradientBoosting. A variação de **um mesmo candidato** entre folds (0,082 a 0,191) é
maior que a variação entre **todos os vinte e um** candidatos dentro de um mesmo fold.

Obedecer ao melhor resultado da validação cruzada mesmo assim custou 30,7 % de PR-AUC (3,44x -> 2,38x;
os dois valores vêm do protocolo de 85 % de treino usado nos notebooks 04 e 05, por isso são comparados
com 3,44x e não com o número de manchete). O
notebook 04 mostra o mecanismo em duas partes: as 50 colunas escolhidas não eram o problema (-0,6 %); as
árvores mais profundas fizeram o *early stopping* parar em 26 rodadas em vez de ~150 (-27,4 %), e refazer
o ajuste com número de rodadas fixo mostra que só um quarto dessa perda vem da profundidade em si. A
validação temporal que o *early stopping* observa quase não acusou o estrago (AP 0,164 vs 0,170), porque
ela é vizinha no tempo enquanto o teste está mais distante.

**Regras que ficam:** não herdar o número de rodadas de um *early stopping* feito em validação
temporalmente vizinha; não adotar um candidato que os folds não separam do incumbente; e mostrar um
intervalo pareado antes de afirmar qualquer diferença.

## As decisões do pipeline e o que as alternativas custam

| decisão | mantido | alternativas | observação |
|---|---|---|---|
| janela | 40 amostras (20 min) | 20 -> 2.94x, 10 -> 3.14x | W=40 ganha em ganho e antecedência; W=10 em cobertura |
| horizonte | 2 h | 0,5 h -> 2.59x, 1 h -> 3.31x, 4 h -> 2.59x | PR-AUC cru sobe com o horizonte só porque o baseline sobe |
| regra do rótulo | maioria | qualquer -> 2.91x | |
| classe positiva | as quatro falhas | só hardware -> 1.94x | ver a ressalva abaixo |
| stride de treino | 5 | 10 -> 3.23x, 20 -> 2.89x | curva ainda subindo: stride menor não foi testado |
| divisão | por data de término, 80/20 | embargo já implícito no protocolo | |

## A ressalva mais importante

TIMEOUT é 70 % dos positivos, e um TIMEOUT normalmente é o usuário subestimando o walltime, não um nó
doente. Restringindo a classe positiva a falhas de hardware de verdade (FAILED / OUT_OF_MEMORY /
NODE_FAIL), o resultado cai de 3,44x para 1.94x (os dois no protocolo de
85 % de treino, então o que importa é a razão entre eles).

A tabela por estado, no nível do job, diz a mesma coisa de forma mais concreta:

| estado | jobs que falharam | tiveram janela | alertados | recall operacional |
|---|---|---|---|---|
| TIMEOUT | 298 | 224 (75%) | 177 | 59.4% |
| FAILED | 772 | 113 (15%) | 84 | 10.9% |
| OOM | 25 | 19 (76%) | 16 | 64.0% |
| NODE_FAIL | 63 | 22 (35%) | 16 | 25.4% |

Jobs FAILED são 67 % de todas as falhas do teste, e só 15 % deles duram o suficiente para gerar uma única
janela de 20 minutos. **Qualquer afirmação de que este modelo prevê saúde de nó tem que ser feita em
1.94x, não em 3.12x.**

## Ponto de operação

No limiar F2 escolhido na validação, o modelo marca 28.1% de todas as
janelas com precisão de 0.070 — muito voltado a recall e, sozinho, não é
um ajuste operável. A visão por orçamento de alertas é a útil:

| orçamento | janelas | precisão | recall | ganho de precisão |
|---|---|---|---|---|
| top 0.1% | 1,412 | 0.900 | 0.022 | 22.4x |
| top 0.5% | 7,162 | 0.343 | 0.043 | 8.5x |
| top 1.0% | 14,291 | 0.228 | 0.058 | 5.7x |
| top 2.0% | 28,144 | 0.230 | 0.114 | 5.7x |
| top 5.0% | 70,363 | 0.155 | 0.192 | 3.8x |
| top 10.0% | 140,719 | 0.113 | 0.282 | 2.8x |

As 0,1 % janelas de maior score são 90% verdadeiras. É esse o número
para levar a um operador: uma fila pequena e de alta pureza, não um limiar que marca um quarto do cluster.

Calibração: a saída bruta é um *ranking*, não uma probabilidade — a média é 0,42 contra uma taxa real de
positivos de 0.0402 (Brier 0.2278). Uma regressão
isotônica ajustada na validação leva o Brier a 0.0466, ao custo de
-12.7% de PR-AUC: a isotônica é
monótona *não-decrescente*, então ela colapsa faixas de score em empates, e o PR-AUC cobra por essa ordem
perdida. Ordenar e alertar com os scores brutos; calibrar só onde uma probabilidade for realmente lida.

## O número honesto, no nível do job

Dos **1158 jobs de teste que falharam**, apenas
378 (32.6%) duraram o suficiente para gerar ao menos
uma janela de 20 minutos. Desses, 293 foram alertados —
**25.3% de todas as falhas** — com
50.3% de alarme falso nos jobs que terminaram bem. Antecedência mediana:
152 minutos no geral, mas só 36 minutos nas
falhas de hardware; a cauda longa é de jobs TIMEOUT que o modelo reconheceu cedo como longos, o que não é
o mesmo que ver uma falha chegando.

O gargalo é a **cobertura**, não a qualidade do modelo. O notebook 05 mede quanto dá para recuperar com
janelas mais curtas: W=10 cobre 50% dos jobs que falham em vez de
33% e alerta 29.3% de todas as falhas em vez
de 24.4% — ao custo de ganho e de um aviso bem mais curto.

## Por que não se compara com o Skrzeczek

Aquele trabalho reporta ROC-AUC ~0,98 agregando a série temporal inteira de cada job em uma única linha e
depois classificando — ou seja, as features incluem o momento da falha. Aqui a previsão sai de uma janela
de 20 minutos que termina **antes** de o horizonte de 2 horas abrir. Pergunta diferente, unidade
diferente, métrica diferente. Comparar os dois números seria desonesto.
