# 🎬 Raport z eksperymentu: Predykcja ocen filmów na podstawie tagów i gatunków

## 1. Wstęp i cel badania

Celem badania było stworzenie modelu regresyjnego, który na podstawie informacji o filmie (gatunki oraz przypisane tagi użytkowników) potrafi przewidywać ocenę, jaką użytkownik mógłby wystawić danemu filmowi. Projekt ten może znaleźć zastosowanie w systemach rekomendacji, gdzie precyzyjne przewidywanie ocen jest kluczowe dla jakości rekomendacji.


## 2. Opis danych

### Źródła danych:

Wykorzystano [MovieLens Latest Datasets](https://grouplens.org/datasets/movielens/latest/).

- movies.csv – metadane filmów, w tym tytuły i gatunki,
- tags.csv – tagi przypisane przez użytkowników do filmów,
- ratings.csv – rzeczywiste oceny użytkowników.

### Cechy danych:

- Liczba filmów: ~9 000
- Liczba rekordów z ocenami: ~100 000
- Liczba gatunków filmowych: 20+
- Liczba tagów (po ekstrakcji TF-IDF): 300
- Liczba końcowych cech (połączenie gatunków i tagów): ok. 320


### Rozkład zmiennej decyzyjnej (oceny):

- Skala ocen: od 0.5 do 5.0 (w krokach co 0.5)
- Typ predykcji: regresja (zmienna ciągła)



## 3. Metodologia i rozwiązanie

1.	**Przygotowanie danych:**
- Wydzielono gatunki filmowe jako cechy binarne (get_dummies).
- Zgrupowano tagi dla każdego filmu i przetworzono je za pomocą TF-IDF (300 najczęstszych słów).
- Połączono cechy w jedną ramkę danych final_df.

2. **Budowa zbioru uczącego:**
- Połączono oceny użytkowników z cechami filmów.
- Usunięto cechy niezwiązane z treścią filmu (userId, movieId, title, timestamp).
- Podzielono dane na zbiór treningowy i testowy (80/20).

3. **Modelowanie:**
- Użyto modelu Random Forest Regressor z 100 drzewami decyzyjnymi (n_estimators=100).
- Model trenowano na zestawie treningowym.

## 4. Metoda oceny jakości modelu

Do oceny jakości modelu wykorzystano standardowe metryki regresji:
- **RMSE (Root Mean Squared Error):** miara błędu średniokwadratowego – premiuje duże błędy.
- **MAE (Mean Absolute Error):** średni błąd bezwzględny – bardziej odporny na outliery.

## 5. Wyniki eksperymentalne

### Wyniki:

- ✅ **RMSE:** 0.984
- ✅ **MAE:** 0.772

### Wizualizacje:

**🔍 Rozkład reszt (różnic między prawdziwymi a przewidywanymi ocenami):**

![image](charts/distribution.png)

**📊 Błąd bezwzględny względem prawdziwej oceny:**

![image](charts/error.png)

**🌟 15 najważniejszych cech wpływających na ocenę filmu:**

![image](charts/features.png)

## 6. Podsumowanie i wnioski

- Model **Random Forest** uzyskał **dobrą jakość predykcji** ocen filmów wyłącznie na podstawie ich treści (gatunki i tagi), bez uwzględniania informacji o użytkowniku.
- TF-IDF okazało się skuteczne w przetwarzaniu tagów i tworzeniu reprezentacji tekstowej.
- Analiza ważności cech może posłużyć do interpretacji, które gatunki i tagi najbardziej wpływają na przewidywane oceny.
- Możliwości dalszej pracy:
    - Uwzględnienie informacji o użytkownikach (filtry kolaboracyjne).
    - Użycie nowszych modeli, np. XGBoost, LightGBM lub sieci neuronowych.
    - Redukcja liczby cech przy zachowaniu jakości predykcji (np. PCA, selekcja cech).