# 🎬 Beetle TV Movie Stream | Matrix Completion Recommender

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Machine--Learning-SVD-orange.svg" alt="ML">
  <img src="https://img.shields.io/badge/Machine--Learning-SoftImpute-ff69b4.svg" alt="SoftImpute">
  <img src="https://img.shields.io/badge/Data--Science-Pandas-150458.svg" alt="Pandas">
  <img src="https://img.shields.io/badge/University-UDEM-yellow.svg" alt="UDEM">
</p>

### 🚀 Access the Live Application / Acesse o App / Acceso Directo:

<p align="left">
  <a href="https://beetletv-movie-recommender.streamlit.app/" target="_blank">
    Beetle TV Movie Recommender
  </a>
</p>

---

## 🌎 Choose your language / Seleccione su idioma / Escolha seu idioma

<details>
<summary><b>English (EN-US) 🇺🇸</b></summary>

### 📌 Project Overview

Developed as an academic project during an exchange program at **Universidad de Monterrey (UDEM)**, Beetle TV is a movie recommendation engine based on **Matrix Completion**. Using the MovieLens 100k dataset, the app allows users to interact with collaborative filtering algorithms in real-time.

### 🔬 Technical Core

1. **Interactive Feedback:** Users rate 20 random movies to build an initial profile.
2. **Data Sparsity Management:** Implementation of a popularity filter (>50 ratings) to ensure recommendation quality.
3. **Dual Mathematical Engines:**
   - **SVD:** Fast matrix factorization for immediate results.
   - **SoftImpute:** Iterative algorithm for high-precision imputation.
4. **Real-Time Movie Posters:** Integration with the TMDB API to display official movie covers during rating and in results.
5. **Profile Metrics:** After rating, the app shows number of rated films, average score, and detected favorite genre.
6. **Algorithm Comparison:** Optional side-by-side SVD vs SoftImpute view, with films recommended by both highlighted in green.
7. **Environment Constraints:** Optimized for Streamlit Cloud with specific memory handling.

</details>

<details>
<summary><b>Español (ES-MX) 🇲🇽</b></summary>

### 📌 Descripción del Proyecto

Proyecto desarrollado en la **Universidad de Monterrey (UDEM)** que consiste en un sistema de recomendación de películas basado en **Matrix Completion**. Utilizando el dataset MovieLens 100k, la aplicación permite al usuario experimentar con algoritmos de filtrado colaborativo.

### 📊 Aspectos Técnicos

- **Interfaz Interactiva:** Calificación de 20 películas aleatorias mediante un sistema de estrellas personalizado.
- **Filtro de Popularidad:** Conservación de películas con más de 50 calificaciones para asegurar la confiabilidad.
- **Motores de Cálculo:** Elección entre **SVD** (rapidez) y **SoftImpute** (precisión matemática en datos dispersos).
- **Portadas en Tiempo Real:** Integración con la API de TMDB para mostrar la imagen oficial de cada película durante la calificación y en los resultados.
- **Métricas del Perfil:** Al finalizar, se muestran el total de películas calificadas, la nota media y el género favorito detectado.
- **Comparación de Algoritmos:** Vista opcional lado a lado de SVD vs SoftImpute, con las coincidencias resaltadas en verde.
- **Optimización:** Configurado específicamente para entornos de nube con Python 3.11.

</details>

<details>
<summary><b>Português (PT-BR) 🇧🇷</b></summary>

### 📌 Visão Geral do Projeto

Desenvolvido durante um intercâmbio na **Universidad de Monterrey (UDEM)**, o Beetle TV é um motor de recomendação de filmes baseado em **Matrix Completion**. O sistema utiliza o dataset MovieLens 100k e aplica álgebra linear para prever as preferências do usuário.

### 🔬 Metodologia e Funcionalidades

- **Ciclo de Feedback:** Avaliação de 20 filmes para criar um perfil de preferências em tempo real.
- **Tratamento de Dados:** Filtro de popularidade para remover ruídos e garantir sugestões relevantes.
- **Algoritmos:** Comparação entre **SVD** (decomposição de valor singular) e **SoftImpute** (preenchimento iterativo).
- **Capas em Tempo Real:** Integração com a API do TMDB para exibir a capa oficial de cada filme durante a avaliação e nos resultados.
- **Métricas do Perfil:** Ao finalizar, são exibidos o total de filmes avaliados, a nota média e o género favorito detetado.
- **Comparação de Algoritmos:** Vista opcional lado a lado de SVD vs SoftImpute, com as coincidências destacadas a verde.
- **UX Customizada:** Interface multi-idioma com design focado na experiência de streaming.

</details>

---

## 🚀 Technical Journey | Trayectoria | Jornada

| Phase             | Component        | Key Visual                  | Goal                                                        |
| ----------------- | ---------------- | --------------------------- | ----------------------------------------------------------- |
| **1. UI Setup**   | Streamlit        | `beetletv-motor-engine.png` | Personalized user entry and data session initialization.    |
| **2. Selection**  | Sidebar          | `beetletv-menu.png`         | Dynamic language switching and mathematical model choice.   |
| **3. Filtering**  | Pandas Engine    | *Code Logic*                | Applied threshold of >50 ratings to fix data sparsity.      |
| **4. Imputation** | SVD / SoftImpute | *Processing*                | Completing the User-Item matrix to predict unknown ratings. |
| **5. Delivery**   | Recommendation   | *Result UI*                 | Top 10 movie suggestions with posters, profile metrics and algorithm comparison. |

---

## 🖼️ Visual Tour | Recorrido Visual | Tour Visual

| **1. Welcome & Entry** <br> ![](https://github.com/douglasbarbosaoliveira/BeetleTV-Movie-Recommender/raw/main/beetletv-motor-engine.png) <br> *Entry point for personalization.* | **2. Model Control** <br> ![](https://github.com/douglasbarbosaoliveira/BeetleTV-Movie-Recommender/raw/main/beetletv-menu.png) <br> *Engine and Language selection.* |
| --- | --- |

---

## 💡 Technical Notes | Notas Técnicas

**EN:** **SoftImpute** is memory-intensive. The app uses a threshold filter and specific model tuning (`max_iters=20`) to maintain stability on Streamlit Cloud. To display movie posters, add your TMDB API key to Streamlit Secrets as `TMDB_API_KEY`. The app works without it — posters are replaced by a placeholder. **Python 3.11 is required.**

**ES:** **SoftImpute** consume mucha memoria. La aplicación utiliza un filtro de umbral y ajustes específicos (`max_iters=20`) para mantener la estabilidad. Para mostrar portadas, agrega tu clave de TMDB en Streamlit Secrets como `TMDB_API_KEY`. La app funciona sin ella — las portadas se reemplazan por un placeholder. **Se requiere Python 3.11.**

**PT-BR:** O **SoftImpute** exige alto processamento. O app utiliza filtros de popularidade e ajuste de hiperparâmetros (`max_iters=20`) para garantir estabilidade no deploy. Para exibir capas, adiciona a tua chave do TMDB nos Streamlit Secrets como `TMDB_API_KEY`. O app funciona sem ela — as capas são substituídas por um placeholder. **Requer Python 3.11.**

---

## 🛠️ Tech Stack | Stack Tecnológico

- **Python 3.11** (`Streamlit`, `Pandas`, `Numpy`, `Scikit-learn`, `SciPy`, `Fancyimpute`, `Requests`).
- **TMDB API** — real-time movie posters.

## 👤 Author / Autor

**Douglas Barbosa de Oliveira**

- Multiplatform Software Development Student (FATEC/UDEM).
- Career transition to Data and BI.

## ⚙️ Setup

```
git clone https://github.com/douglasbarbosaoliveira/BeetleTV-Movie-Recommender.git
pip install -r requirements.txt
streamlit run app.py
```
