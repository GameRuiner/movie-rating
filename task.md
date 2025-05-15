# Projekt z ADD – Temat 3
**Ocena filmowa (max 3 osoby)**
1. **Dane do analizy**: składają się z ocen (*ratings*) filmów (w skali od 1-5 gwiazd) oraz
krótkich recenzji (*tagi*) napisanych przez widzów. Informacje te zostały zbierane przez
portal **MovieLens** – portal do celu rekomendacji filmowej.
  
  Dane zawierają około100000 ratings i ponad 1000 tagów dla ponad 9000
filmów. Dane zostały stworzone przez użytkowników w okresie 1995-2016.
Użytkownicy są wybrani w sposób losowy. Każdy użytkownik ocenia co
najmniej 20 filmów, i ma unikany identyfikator.

3. **Zadanie**:
  - **Przygotowanie danych do analizy**:
     
     Na podstawie **tagów i opisu** filmu tworzyć odpowiedni **zbiór atrybutów** i **zbiór
rekordów** do analizy.
  - **Przewidywanie ratings**:
    
      a) Używać **dowolnego klasyfikatora** (np. sieci neuronowe, drzewo
decyzyjne, klasyfikator Bayesowski, zespół klasyfikatorów,...) do predykcji
oceny nowego filmu.

      b) Odkrywać **istotne cechy** do oceniania filmu.
4. **Model uczenia**:
    - Zbiór uczący: losowo 80% zbioru danych
    - Zbiór testowy: pozostała część.
5. **Zródło danych**:
  - Udostępnione w **Datasets**.
  - Więcej informacji: https://grouplens.org/datasets/movielens/latest/
