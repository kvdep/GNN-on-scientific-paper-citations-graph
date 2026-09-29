# Link Prediction & Citation Recommendation in Scientific Networks under Cold-Start

Репозиторий содержит реализацию исследовательской системы рекомендаций цитирований для научных статей на базе графовых нейронных сетей (GNN) и градиентного бустинга (GBDT).

Я решил задачу предсказания цитирований (Link Prediction) в условиях индуктивного холодного старта: когда новая публикация поступает в систему, она не имеет исторических входящих и исходящих ребер в графе цитирования. Рекомендации формируются путем объединения текстовых признаков (SciBERT + Autoencoder), проекций графа соавторства (FastRP) и структурных эвристик.

---

## 1. Результаты экспериментов (Results First)

Я оценивал все модели в строгом протоколе полного ранжирования 1-vs-all (каждая тестовая статья ранжирует всех допустимых кандидатов графа). Кандидаты фильтруются по временному критерию: год публикации кандидата не может превышать год публикации целевой статьи ($y_v \le y_u$).

### Таблица 1. Сравнение лучших моделей на гетерогенном графе (Индуктивный холодный старт)

| Архитектура | Предиктор | Функция потерь | Признак Hub | MRR | Hits@10 | Rec@10 | NDCG@10 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **LightGCN** | **BUDDY (Late Fusion)** | **BCE** | **True** | **0.3516** | **0.5950** | **0.3054** | **0.2575** |
| LightGCN | Standard | BCE | True | 0.3355 | 0.5745 | 0.2846 | 0.2377 |
| MLP (No GNN) | BUDDY (Late Fusion) | BCE | True | 0.3286 | 0.5665 | 0.2857 | 0.2392 |
| GraphSAGE | BUDDY (Late Fusion) | BCE | True | 0.3268 | 0.5617 | 0.2836 | 0.2383 |
| GraphSAGE | Standard | BCE | True | 0.3210 | 0.5533 | 0.2645 | 0.2221 |
| MLP (No GNN) | Standard | BCE | True | 0.3076 | 0.5296 | 0.2388 | 0.2036 |
| LightGCN | BUDDY (Late Fusion) | BCE | False | 0.3190 | 0.5268 | 0.2583 | 0.2204 |
| CatBoost (GBDT) | — | QueryRMSE | False | 0.1188 | 0.2117 | 0.0680 | 0.0610 |
| CatBoost Inductive | — | YetiRank | False | 0.0774 | 0.1386 | 0.0479 | 0.0429 |
| CatBoost Cold Start | — | YetiRank | False | 0.0138 | 0.0217 | 0.0046 | 0.0052 |

### Таблица 2. Сравнение гомогенных GNN и функций потерь (Этап 1)

| Архитектура | Предиктор | Функция потерь | MRR | Hits@10 | Rec@10 | NDCG@10 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **NeoGNN** | **Standard** | **Margin Loss** | **0.2094** | **0.4799** | **0.0639** | **0.0889** |
| LightGCN | Standard | Margin Loss | 0.1900 | 0.4254 | 0.0528 | 0.0739 |
| SGC | Standard | Margin Loss | 0.1611 | 0.3786 | 0.0482 | 0.0648 |
| LightGCN | NCN | Margin Loss | 0.1634 | 0.3698 | 0.0522 | 0.0704 |
| NeoGNN | NCN | Margin Loss | 0.1575 | 0.3592 | 0.0472 | 0.0646 |
| SGC | NCN | Margin Loss | 0.1351 | 0.3078 | 0.0415 | 0.0567 |
| DirLightGCN | NCN | BPR Loss | 0.1001 | 0.2412 | 0.0260 | 0.0357 |
| **Baseline Topo** | **Heuristics** | **—** | **0.1019** | **0.1820** | **0.0165** | **0.0296** |
| GraphSAGE | Standard | InfoNCE | 0.0480 | 0.1081 | 0.0080 | 0.0135 |
| LightGCN | Standard | Focal Loss | 0.0026 | 0.0032 | 0.0002 | 0.0004 |
| LightGCN | Standard | ASL | 0.0035 | 0.0034 | 0.0002 | 0.0005 |

---

### Ключевые выводы по числам

1. **Эффективность связки LightGCN + BUDDY Predictor:**
   Конфигурация `LightGCN + BUDDY + Hub=True` показала наивысшее качество: **NDCG@10 = 0.2575**, **Hits@10 = 0.5950**, **MRR = 0.3516**. Архитектура позднего слияния (Late Fusion) в BUDDY обрабатывает семантику статьи и эвристики соавторства в раздельных полносвязных слоях, предотвращая доминирование одного типа признаков над другим.
2. **Влияние априорной центральности (Hub Feature):**
   Добавление логарифмированной входящей степени целевой статьи $\log(\mathrm{deg}_{in}(v) + 1)$ увеличило NDCG@10 модели `LightGCN + BUDDY` с **0.2204** до **0.2575** (+16.8% относительного прироста). Входящие степени в сетях цитирования подчиняются степенному закону; распределение цитирований сильно скошено в сторону цитируемых статей-хабов.
3. **Коллапс классификационных лоссов:**
   Классификационные функции потерь (Focal Loss, ASL) показали значение NDCG@10 < **0.001**. При ранжировании 1-vs-all количество отрицательных примеров превышает положительные в $10^4$ раз, из-за чего классификаторы зануляют вероятности для всех пар. Ранжирующие функции потерь (Margin Loss, BPR Loss) оптимизируют относительный зазор между парами, сохраняя точность порядка ранжирования.
4. **Ограничения табличного градиентного бустинга (GBDT):**
   Модель `CatBoostRanker` с 390 признаками на индуктивном холодном старте показала **NDCG@10 = 0.0429** (в 6 раз ниже LightGCN), а при чистом холодном старте — **NDCG@10 = 0.0052**. Причина заключается в отсутствии у изолированной статьи общих соседей (Common Neighbors = 0, Adamic-Adar = 0) и невозможности бустинга выучить нелинейные проекции графа на этапе инференса.

---

## 2. Постановка задачи (Problem Formulation)

Пусть задан ориентированный граф цитирования $G = (V_p, E_{\mathrm{cites}})$, где $V_p$ — множество научных статей, а $(u, v) \in E_{\mathrm{cites}}$ обозначает факт цитирования статьи $v$ статьей $u$. Одновременно задано множество авторов $V_a$ и двудольные ребра авторства $E_{\mathrm{writes}} \subseteq V_a \times V_p$.

Для каждой статьи $u \in V_p$ известны:
* Год публикации $y_u \in \mathbb{N}$;
* Текстовые поля: заголовок $T_u$, аннотация $A_u$, ключевые концепты $C_u$;
* Множество соавторов $\mathcal{A}_u \subseteq V_a$.

### Условие индуктивного холодного старта
Для тестовой статьи $u_{test} \in V_{test}$ на момент инференса в матрице цитирований:

$$
\mathrm{deg}_{in}(u_{test}) = 0, \quad \mathrm{deg}_{out}(u_{test}) = 0
$$

Задача: для статьи $u_{test}$ отранжировать все статьи $v \in V_p$, удовлетворяющие причинно-следственному временному ограничению:

$$
\mathcal{C}(u_{test}) = \{ v \in V_p \setminus \{u_{test}\} \mid y_v \le y_{u_{test}} \}
$$

так, чтобы истинные статьи из библиографического списка $u_{test}$ заняли наивысшие позиции в выдаче.

---

## 3. Набор данных и предварительная обработка

Я собрал выборку через REST API каталога **OpenAlex** по предметной области Machine Learning за 10 лет (хронологический диапазон: 2017–2026 гг.).

### Фильтрация плотного ядра (`k-core`)
Исходный граф разрежен и содержит статьи без связей. Для формирования связной структуры соавторства применен алгоритм `k-core` с порогом $k=3$ к двудольному графу «автор-статья»:

$$
G_{\mathrm{core}} = \mathrm{k-core}(G_{\mathrm{bipartite}}, k=3)
$$

### Количественные параметры выборки
* **Всего статей в ядре:** 54 262
* **Ребер цитирования в обучающей выборке:** 101 719
* **Обучающая выборка (Train, 85%):** 50 688 статей
* **Валидационная выборка (Val, 5%):** 2 981 статья
* **Тестовая выборка (Test, 10%):** 593 статьи

### Метод маскирования ребер (Masked Graph Splitting)
Строгий хронологический сплит приводил к потере профилей авторов последних лет. Поэтому я применил схему маскирования:
1. Авторы всех статей сохраняются в матрице смежности соавторства;
2. Ребра цитирования тестовых и валидационных статей удаляются из графа агрегации сообщений $G_m$;
3. Статьи Train имеют полные связи цитирования; статьи Test и Val выступают изолированными вершинами в $G_m$.

### Фильтрация временных аномалий
В открытых данных присутствуют ребра, где $y_u < y_v$ (вызванные обновлением версий препринтов на arXiv и постдатированием номеров журналов). Я ввел фильтр, принудительно исключающий любые ребра из будущего в прошлое ($y_v \le y_u$).

---

## 4. Конструирование признаков (Feature Engineering)

```
[Title + Abstract + Concepts]
             │
             ▼
   SciBERT (2304 dim)
             │
             ▼
   TextAE (256 dim)  ◄── MSE Reconstruction Loss (AdamW, 40 epochs)
             │
             ├──────────────────────────┐
             ▼                          ▼
    Joint Feature Vector        CatBoost Vector (390 dim)
    [Text (256) || FastRP (128)] ───►  4 Heuristics (dt, author_jaccard, concept_jaccard, hub)
             │                  ───► 129 FastRP Cosine + Hadamard
             ▼                  ───► 257 Text Cosine + Hadamard
       GNN Backbone
```

### 1. Текстовые эмбеддинги (SciBERT + Autoencoder)
* Модель: `pritamdeka/S-SciBERT-snli-multinli-stsb`.
* Векторизация полей: Заголовок (768), Аннотация (768), Концепты (768). Конкатенация формирует вектор размерности 2304.
* Сжатие: полносвязный автоэнкодер (`TextAE`):
  * **Encoder:** `Linear(2304 → 1024) → LayerNorm → GELU → Dropout(0.1) → Linear(1024 → 256)`
  * **Decoder:** `Linear(256 → 1024) → LayerNorm → GELU → Dropout(0.1) → Linear(1024 → 2304)`
  * **Обучение:** 40 эпох, AdamW, $\mathrm{lr} = 10^{-3}$, целевая функция — MSE. Выходной латентный вектор статьи имеет размерность **256**.

### 2. Топологические эмбеддинги (FastRP)
Алгоритм Fast Random Projection рассчитывает проекции без градиентного спуска. На матрице смежности цитирований и соавторства $A$ строится матрица переходов случайного блуждания:

$$
P = D^{-1} A
$$

Генерируется случайная матрица проекций $R \in \mathbb{R}^{|V| \times d}$ ($d=128$). Финальное представление рассчитывается по 3 шагам блуждания с весами $w = [0.1, 0.4, 0.5]$:

$$
Z_{\mathrm{topo}} = \sum_{l=1}^{3} w_l P^l R
$$

### 3. Агрегация авторской топологии
Для каждой статьи $u$ вычисляется средний топологический вектор ее соавторов:

$$
z_{\mathrm{author}}(u) = \frac{1}{|\mathcal{A}_u|} \sum_{a \in \mathcal{A}_u} z_a
$$

### 4. Парные структурные эвристики
Для пары статей $(u, v)$ рассчитывается вектор $h_{uv}$:

#### Разница лет

$$
\Delta t = y_u - y_v
$$

#### Перекрытие соавторов (Jaccard)

$$
J_{\mathrm{author}}(u, v) = \frac{|\mathcal{A}_u \cap \mathcal{A}_v|}{|\mathcal{A}_u \cup \mathcal{A}_v|}
$$

#### Перекрытие концептов (Jaccard)

$$
J_{\mathrm{concept}}(u, v) = \frac{|\mathcal{C}_u \cap \mathcal{C}_v|}{|\mathcal{C}_u \cup \mathcal{C}_v|}
$$

#### Априорная центральность (Hubness)

$$
\mathrm{Hub}(v) = \log(\mathrm{deg}_{in}(v) + 1)
$$

---

## 5. Архитектуры моделей

### Графовые энкодеры

#### LightGCN
Симметричное линейное сглаживание по ребрам:

$$
\tilde{A} = \tilde{D}^{-1/2} (A + I) \tilde{D}^{-1/2}, \quad Z = \tilde{A} (X W)
$$

#### DirLightGCN
Раздельная агрегация входящих и исходящих ребер:

$$
Z = X W + D_{\mathrm{out}}^{-1} A (X W) + D_{\mathrm{in}}^{-1} A^T (X W)
$$

#### NeoGNN
Параллельное кодирование признаков через LightGCN и топологии через обучаемую матрицу эмбеддингов узлов $E \in \mathbb{R}^{|V| \times (d/2)}$:

$$
Z = [Z_{\mathrm{feat}} \,\|\, Z_{\mathrm{struct}}], \quad Z_{\mathrm{struct}} = \tilde{A} E
$$

#### GraphSAGE
Конкатенация собственного вектора и среднего по соседям:

$$
Z = \mathrm{ReLU}\left( W_1 X + W_2 (D^{-1} A X) \right)
$$

#### SGC
$k$-шаговое предварительное сглаживание:

$$
Z = \tilde{A}^k X W
$$

### Предикторы связей

#### BUDDY Predictor (Late Fusion)

$$
\mathbf{h}_{\mathrm{sem}} = \mathrm{MLP}_{\mathrm{sem}}([z_u \,\|\, z_v]), \quad \mathbf{h}_{\mathrm{str}} = \mathrm{MLP}_{\mathrm{str}}(h_{uv})
$$

$$
\mathrm{Score}(u, v) = \sigma\left( \mathrm{MLP}_{\mathrm{fuse}}([\mathbf{h}_{\mathrm{sem}} \,\|\, \mathbf{h}_{\mathrm{str}}]) \right)
$$

#### Standard Predictor

$$
\mathrm{Score}(u, v) = \sigma\left( \mathrm{MLP}([z_u \,\|\, z_v \,\|\, z_u \odot z_v \,\|\, z_u - z_v \,\|\, h_{uv}]) \right)
$$

#### NCN Predictor

$$
\mathrm{Score}(u, v) = \sigma\left( \mathrm{MLP}([z_u \odot z_v \,\|\, h_{uv}]) \right)
$$

### Функции потерь

#### Margin Ranking Loss

$$
\mathcal{L}_{\mathrm{Margin}} = \frac{1}{|B|} \sum_{(u, v^+, v^-) \in B} \max(0, s(u, v^-) - s(u, v^+) + \gamma)
$$

#### Bayesian Personalized Ranking (BPR Loss)

$$
\mathcal{L}_{\mathrm{BPR}} = -\frac{1}{|B|} \sum_{(u, v^+, v^-) \in B} \log \sigma(s(u, v^+) - s(u, v^-))
$$

#### InfoNCE Loss

$$
\mathcal{L}_{\mathrm{InfoNCE}} = -\frac{1}{|B|} \sum_{i} \left( \frac{s(u_i, v_i^+)}{\tau} - \log \sum_{j} \exp\left( \frac{s(u_i, v_{i, j}^-)}{\tau} \right) \right)
$$

#### Asymmetric Loss (ASL)

$$
\mathcal{L}_{\mathrm{ASL}} = -y (1 - p)^{\gamma_+} \log(p) - (1 - y) p^{\gamma_-} \log(1 - p)
$$

---

## 6. Метрики оценки качества

Оценка выполняется по ранжированному списку кандидатов длины $N$:

### 1. Mean Reciprocal Rank (MRR)

$$
\mathrm{MRR} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \frac{1}{\mathrm{rank}_q^{(1)}}
$$

где $\mathrm{rank}_q^{(1)}$ — позиция первого релевантного документа в выдаче.

### 2. Hits@K

$$
\mathrm{Hits@K} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \mathbb{I}(\mathrm{rank}_q^{(1)} \le K)
$$

### 3. Recall@K

$$
\mathrm{Recall@K} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \frac{|\mathrm{Top}_K(q) \cap \mathrm{GT}(q)|}{|\mathrm{GT}(q)|}
$$

### 4. Normalized Discounted Cumulative Gain (NDCG@K)

$$
\mathrm{DCG@K} = \sum_{p \in \mathrm{GT}(q), p \le K} \frac{1}{\log_2(p + 1)}, \quad \mathrm{IDCG@K} = \sum_{p=1}^{\min(|\mathrm{GT}(q)|, K)} \frac{1}{\log_2(p + 1)}
$$

$$
\mathrm{NDCG@K} = \frac{\mathrm{DCG@K}}{\mathrm{IDCG@K}}
$$

---

## 7. Структура проекта

```
.
├── configs/
│   └── default_config.yaml          # Гиперпараметры и пути
├── docs/
│   ├── coursework_kydysiuk_2026.pdf # Полный текст курсовой работы
│   └── presentation_kydysiuk_2026.pdf # Слайды презентации
├── notebooks/
│   └── citation_network_gnn.ipynb   # Исследовательский Jupyter Notebook
├── scripts/
│   ├── fetch_data.py                # Сбор данных через OpenAlex API
│   ├── build_graph.py               # Прунинг k-core и генерация сплитов
│   ├── extract_features.py          # Расчет SciBERT, TextAE, FastRP
│   ├── train_gnn.py                 # Обучение и бенчмарк GNN моделей
│   └── train_gbdt.py                # Обучение CatBoostRanker
├── src/
│   ├── data/
│   │   ├── openalex_fetcher.py      # Клиент OpenAlex API с пагинацией
│   │   └── graph_builder.py         # NetworkX граф, k-core, маскирование
│   ├── features/
│   │   ├── text_encoder.py          # SciBERT + TextAE PyTorch
│   │   ├── fastrp.py                # Fast Random Projection
│   │   ├── heuristics.py            # Векторизованные структурные эвристики
│   │   └── negative_sampler.py      # Причинно-следственный сэмплинг
│   ├── models/
│   │   ├── gnn_encoders.py          # LightGCN, DirLightGCN, NeoGNN, SAGE, SGC
│   │   ├── predictors.py            # BUDDY Late Fusion, Standard, NCN, GM
│   │   ├── losses.py                # MarginLoss, BPRLoss, InfoNCE, ASL, FL
│   │   └── gbdt_ranker.py           # CatBoostRanker (390 признаков)
│   ├── evaluation/
│   │   ├── metrics.py               # Реализация MRR, Hits@10, Rec@10, NDCG@10
│   │   └── evaluator.py             # Движок 1-vs-all индуктивной оценки
│   └── training/
│       └── trainer.py               # Цикл обучения GNN с ранней остановкой
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 8. Воспроизводимость (Quickstart)

### Установка зависимостей
```bash
git clone https://github.com/kvdep/GNN-on-scientific-paper-citations-graph.git
cd GNN-on-scientific-paper-citations-graph
pip install -r requirements.txt
```

### 1. Сбор данных
```bash
python scripts/fetch_data.py --output data/raw_data.jsonl --start-year 2017 --end-year 2026 --target-per-year 200000
```

### 2. Построение графа и прунинг (`k-core`)
```bash
python scripts/build_graph.py --input data/raw_data.jsonl --output data/processed_graph.pkl --k-core 3
```

### 3. Извлечение признаков (SciBERT + TextAE + FastRP)
```bash
python scripts/extract_features.py --graph-pkl data/processed_graph.pkl --raw-jsonl data/raw_data.jsonl --output-pt data/features.pt
```

### 4. Обучение лучшей модели (LightGCN + BUDDY + BCE + Hub)
```bash
python scripts/train_gnn.py --encoder lightgcn --predictor buddy --loss bce --hub --epochs 15
```

### 5. Обучение бейзлайна CatBoost
```bash
python scripts/train_gbdt.py --loss-function YetiRank --iterations 1000
```
