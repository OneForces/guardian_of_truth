# 🛡 Guardian of Truth — Detection of Hallucinations in LLM

Проект определяет вероятность галлюцинаций в ответах LLM на основе внутренних признаков модели (энтропия, вероятности токенов, скрытые состояния) и обученного классификатора.

---

## 🚀 Быстрый старт

### 1. Установка зависимостей

pip install -r requirements.txt

---

### 2. Проверка (быстрый режим)

python predict.py --input data/raw/knowledge_bench_public.csv --limit 3

✔ Обрабатывает 3 строки (10–20 секунд)

---

### 3. Полный запуск

python predict.py --input data/raw/knowledge_bench_public.csv --output outputs/submission.csv

✔ Результат сохраняется в:

outputs/submission.csv

---

## 📂 Формат входного файла

CSV должен содержать:

- prompt — вопрос / запрос  
- model_answer — ответ модели  

Пример:

prompt,model_answer
Кто написал "Войну и мир"?,Лев Толстой

---

## 📤 Формат выхода

id,score
0,0.73
1,0.21

- score — вероятность галлюцинации (0–1)

---

## ⚙️ Аргументы запуска

python predict.py --input <path> --output <path> --limit <N>

---

## 🧠 Как работает система

1. Загружается LLM  
2. Выполняется forward pass  
3. Извлекаются признаки (энтропия, вероятности, hidden states)  
4. Классификатор предсказывает вероятность галлюцинации  

---

## ⚠️ Важно

- Работает на CPU (GPU не обязателен)
- При ошибках используется fallback (score = 0.5)

---

## ✅ Минимальный тест для комиссии

pip install -r requirements.txt
python predict.py --input data/raw/knowledge_bench_public.csv --limit 3
