# 🎬 Movie Rating Prediction

Projekt polega na przewidywaniu ocen filmów na podstawie ich cech treściowych, takich jak gatunki oraz tagi przypisane przez użytkowników. Model regresyjny uczy się na danych z serwisu MovieLens i stara się oszacować, jaką ocenę użytkownik mógłby wystawić konkretnemu filmowi.

---

## 📁 Zawartość repozytorium

- `report.md` – [szczegółowy raport eksperymentu](report.md), opisujący dane, metodologię, wyniki i wnioski.
- `task` – [opis zadania](task), które było podstawą do realizacji projektu.
- `notebooks/` – notebooki Jupyter z kodem do analizy danych, trenowania modeli i wizualizacji.
- `scripts/` - skrypty Pythona z datasetem i trenowaniem modeli

---

## ⚙️ Uruchomienie projektu

1. Zainstaluj wymagane biblioteki:
   ```bash
   pip install -r scripts/requirements.txt
   ```
2.	Uruchom skrypt trenujący model:
    ```bash
    python scripts/train_model.py
    ```
3. (Opcjonalnie) Otwórz notebook z eksperymentem:
    ```
    jupyter notebook notebooks/random_forest_evaluation.ipynb
    ```

## 🧾 Licencja

Projekt edukacyjny realizowany w ramach zajęć z analizy dużych danych.