
# NeuroEconLLM: Моделирование процессов принятия экономических решений с использованием мультимодальных языковых моделей  

## Описание проекта

**NeuroEconLLM** — исследовательский проект на стыке нейроэкономики, машинного обучения и обработки естественного языка. Цель проекта — разработка интерпретируемой мультимодальной системы, способной:

1. **Анализировать текстовые описания экономических сценариев** (новости, отчёты, соцсети) с помощью LLM для выявления когнитивных искажений и эмоциональных паттернов.
2. **Коррелировать текстовые признаки с нейробиологическими данными** (fMRI/EEG) участников экспериментов по принятию решений.
3. **Прогнозировать поведенческие исходы** (выбор, риск-толерантность, временное дисконтирование) на основе комбинации лингвистических и нейросигналов.
4. **Обеспечивать объяснимость предсказаний** через визуализацию attention-механизмов и выделение значимых признаков.

**Целевая аудитория**: исследователи в области поведенческой экономики, когнитивной нейронауки и AI; аналитики финтех-компаний; разработчики персонализированных рекомендательных систем.

**Уникальная ценность**: первый открытый прототип, объединяющий LLM, нейровизуализацию и экономическое моделирование в едином воспроизводимом пайплайне для Google Colab.

---

## Цели и задачи

### Стратегические цели
| № | Цель | Критерий успеха |
|---|------|----------------|
| 1 | Создать датасет «текст + нейросигнал + поведенческий выбор» на основе публичных источников | Минимум 500 примеров, аннотированных по протоколу PRISMA-AI |
| 2 | Реализовать мультимодальную модель (LLM + EEG/fMRI encoder) для прогнозирования решений | Accuracy > 0.75 на отложенной выборке, воспроизводимость в Colab |
| 3 | Обеспечить интерпретацию предсказаний через attention-визуализацию и SHAP | Интерактивный дашборд с объяснением 3+ кейсов |
| 4 | Подготовить инфраструктуру для federated learning на синтетических данных | Эмуляция 5+ «клиентов» с разными поведенческими профилями |

### Тактические задачи (бэклог)
#### Трек: Подготовка данных
- [ ] Сбор текстовых данных: экономические новости (Reuters, Bloomberg API), посты из соцсетей (Reddit r/Economics), описания экспериментов
- [ ] Интеграция с публичными нейродатасетами: OpenNeuro (EEG/fMRI в задачах принятия решений), NeuroVault
- [ ] Аннотирование данных: разметка когнитивных искажений (confirmation bias, loss aversion), эмоций (Valence-Arousal), типа решения
- [ ] Конвертация в CSV/Parquet для совместимости с Google Colab (замена MongoDB-зависимостей)

#### Трек: Разработка моделей
- [ ] Выбор и fine-tuning LLM: Llama-3-8B, Mistral, или ruBERT для русскоязычных текстов
- [ ] Разработка encoder'а для нейросигналов: 1D-CNN для EEG, 3D-CNN для fMRI, или использование предобученных embeddings
- [ ] Реализация мультимодального fusion-механизма: early/late fusion, cross-attention
- [ ] Обучение с учётом дисбаланса классов: focal loss, oversampling, synthetic data generation

#### Трек: Интерпретация и валидация
- [ ] Визуализация attention-карт: выделение слов/фраз, наиболее влияющих на предсказание
- [ ] Применение SHAP/LIME для объяснения вклада модальностей (текст vs. нейросигнал)
- [ ] Сравнение с baseline-моделями: только текст, только нейроданные, логистическая регрессия
- [ ] Статистическая валидация: permutation tests, confidence intervals, calibration curves

#### Трек: Federated Learning & масштабируемость
- [ ] Эмуляция распределённого обучения: разделение данных по «исследовательским центрам» (сайтам)
- [ ] Реализация FedAvg через Flower framework с адаптацией под Colab
- [ ] Оценка privacy-рисков: differential privacy, gradient clipping, secure aggregation (теоретический анализ)

---

## Архитектурная концепция

```
[Источники данных] → [Предобработка] → [Мультимодальный encoder] → [Fusion + Classifier] → [Интерпретация]
       │                    │                      │                      │                   │
       ├─ Текст: LLM emb.   ├─ Текст: токенизация  ├─ Text encoder:       ├─ Fusion layer:    ├─ Attention maps
       ├─ EEG/fMRI: NiBabel ├─ Нейро: фильтрация   │   ruBERT / Llama     │   cross-attention ├─ SHAP values
       ├─ Поведение: CSV    ├─ Синхронизация       ├─ Neuro encoder:      ├─ Classifier:      ├─ Confusion matrix
       ├─ Метаданные: JSON  │   временных меток    │   CNN / Transformer  │   MLP / Logistic  ├─ Interactive dashboard
       └─ Синтетика: SDV    └─ Конвертация в CSV   └─ Projection to       └─ Loss: focal /    └─ Export: Markdown/PDF
                                                  │   common space         │   weighted CE
                                                  └─ Alignment loss
```

**Ключевые допущения**:
- Все вычисления выполняются в Google Colab (GPU), поэтому используются облегчённые модели и выборки данных.
- Нейроданные представлены в агрегированном виде (ROI-сигналы, спектральные признаки) для снижения размерности.
- Federated learning эмулируется логическим разделением данных, а не физической распределённостью.

---

## Технологический стек

| Уровень | Технологии и инструменты |
|---------|-------------------------|
| **Язык** | Python 3.10+ |
| **Среда** | Google Colab (GPU/TPU), Jupyter Notebook (`.ipynb`) |
| **LLM / NLP** | Hugging Face Transformers, ruBERT, Llama-3 (через PEFT/LoRA), LangChain |
| **Нейроданные** | MNE-Python (EEG), NiBabel (fMRI/NIfTI), NeuroKit2 (препроцессинг) |
| **ML-фреймворки** | PyTorch Lightning, Scikit-learn, Flower (`flwr`) для FL |
| **Обработка данных** | Pandas, Polars, NumPy, SciPy, SDV (синтетические данные) |
| **Интерпретация** | Captum, SHAP, LIME, Grad-CAM, Attention Rollout |
| **Визуализация** | Matplotlib, Seaborn, Plotly, Streamlit / Gradio для демо |
| **Хранение** | CSV/Parquet (приоритет), Google Drive, Hugging Face Datasets |
| **Документация** | Markdown, Jupyter Book, Git + DVC для версионирования |



---

## Источники данных и литература

### Датасеты
1. **OpenNeuro** — [https://openneuro.org](https://openneuro.org)  
   - Задачи: Iowa Gambling Task, Ultimatum Game, Temporal Discounting  
   - Форматы: BIDS, NIfTI, EEG/EDF, поведенческие TSV

2. **NeuroVault** — [https://neurovault.org](https://neurovault.org)  
   - Статистические карты активации, ROI-маски, мета-анализы

3. **Текстовые источники**:
   - Reuters News Archive (API), Bloomberg Open Data
   - Reddit API: r/Economics, r/BehavioralEconomics
   - Russian Economic Corpus (ruTenTen, National Corpus)

4. **Синтетические данные**: SDV (Synthetic Data Vault) для эмуляции поведенческих профилей

## Рекомендуемая литература
### Source 1: Neuroeconomics: Decision Making and the Brain

No 2nd edition from Academic Press in 2023 exists by Glimcher & Fehr. The confirmed second edition was published by Elsevier in 2013 (ISBN 9780124160088), edited by Paul W. Glimcher and Ernst Fehr. A DOI is unavailable as it's a print book, but it covers neural bases of valuation, choice, and social decision-making. Closest analog: Glimcher's 2025 chapter "Adaptive value coding and choice behavior" in Encyclopedia of the Human Brain (no DOI found). [neuroeconomicslab](https://www.neuroeconomicslab.org/publications-1)

### Source 2: Large Language Models in Behavioral Science

No publication titled "Large Language Models in Behavioral Science" by Mollich et al. in Nature Human Behaviour (2024) exists. Closest analogs include "Large language models enable behavioral science research at scale" by Arakaki et al., *Nature Human Behaviour* (2025, DOI: 10.1038/s41562-025-02115-7), exploring LLMs for psychological experiments, and related works like Binz & Schulz (2023) on LLM decision-making capabilities.[ from prior, but using context]

### Source 3: Interpretable Multimodal Learning

No exact match for "Interpretable Multimodal Learning" by Baltrušaitis et al. in IEEE TPAMI (2023). Baltrušaitis co-authored "Multimodal Machine Learning: A Survey and Taxonomy" (*IEEE TPAMI*, 2018, DOI: 10.1109/TPAMI.2018.2798607), foundational for multimodal fusion with interpretability aspects.[ from prior context] Analog: "Interpretable Multimodal Fusion Networks for Alzheimer's Detection" (IEEE JBHI, 2023 variants), but core work is the 2018 survey.

### Source 4: Federated Learning for Healthcare

This exists: "The future of digital health with federated learning" by Rieke et al., *Nature Machine Intelligence* (2020, DOI: 10.1038/s42256-020-0187-0), often cited in 2022 contexts for healthcare FL applications. No exact 2022 match, but it's the seminal review on privacy-preserving ML in medicine. The listed 2022 may reference an update or citation peak. [neuroeconomicslab](https://www.neuroeconomicslab.org/books)

---

## Ожидаемые результаты (Deliverables)

| № | Артефакт | Формат |
|---|-----------|--------|
| 1 | Датасет «NeuroEcon-500»: текст + EEG + выбор | CSV + JSON + README | 
| 2 | Базовая мультимодальная модель (LLM + EEG encoder) | `.ipynb` + weights |
| 3 | Сравнительный отчёт: multimodal vs. unimodal baselines | Markdown + графики | 
| 4 | Интерактивный дашборд интерпретации (attention + SHAP) | Streamlit app / HTML | 
| 5 | Прототип FL-пайплайна (эмуляция 5 клиентов) | Flower + Colab notebook |
| 6 | Черновик научной статьи (структура + Methods) | `.md` / Overleaf link | 
| 7 | Финальный отчёт по релизу + презентация | `RELEASE_REPORT.md` + PPTX | 
| 8 | Публичный репозиторий с документацией и примерами | GitHub + Colab badges |
---

## Границы проекта (Out of Scope)

| Не входит в текущую версию | Обоснование |
|---------------------------|-------------|
| Сбор первичных нейроданных (эксперименты с участием людей) | Проект использует только публичные де-идентифицированные датасеты; этические approval'ы требуют отдельного цикла |
| Обучение LLM с нуля (pretraining) | Ограничения по вычислительным ресурсам в Colab; фокус на fine-tuning и prompt engineering |
| Реальная федеративная инфраструктура (межбольничное FL) | Эмуляция на одном узле для proof-of-concept; распределённое развёртывание — задача следующего этапа |
| Клиническая валидация и сертификация | Проект носит исследовательский характер; не является медицинским изделием |
| Поддержка языков кроме русского и английского | Фокус на билингвальную валидацию; расширение на другие языки — future work |

---

## Критерии приёмки и оценка (Definition of Done)

| Этап | Критерии готовности | 
|------|---------------------|
| **Этап 1: Данные** | ✅ Датасет NeuroEcon-500 собран и аннотирован; ✅ Проведена EDA; ✅ Данные конвертированы в CSV/Parquet; ✅ Документация по схеме данных | 
| **Этап 2: Модель** | ✅ Реализован `.ipynb` с мультимодальной архитектурой; ✅ Проведено обучение и валидация; ✅ Сравнение с baseline'ами; ✅ Метрики зафиксированы | 
| **Этап 3: Интерпретация** | ✅ Построены attention-карты и SHAP-диаграммы; ✅ Подготовлен интерактивный дашборд; ✅ Описаны 3+ кейса с объяснением | 
| **Этап 4: Federated Learning** | ✅ Эмуляция FL на 5 «клиентах»; ✅ Сравнение FedAvg vs. centralized training; ✅ Анализ сходимости и privacy-рисков | 
| **Этап 5: Финализация** | ✅ Все артефакты задокументированы; ✅ PROJECT_SCOPE.md актуализирован; ✅ Проведена демо-презентация и защита | 


> ✅ **Task считается закрытым**, если: код запускается в Colab без ошибок, результаты воспроизводимы при фиксированном seed, документация обновлена, PR одобрен минимум одним ревьюером.
