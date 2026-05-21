import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
import requests

# 1. CONFIGURAÇÕES VISUALES E INTERFAZ
st.set_page_config(page_title="Beetle TV Movie Stream", layout="centered")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@300;400;500;600&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Quicksand', sans-serif !important;
        background-color: #0f171e;
        color: #e1e4e8;
    }
    .stApp { background-color: #0f171e; }
    div[data-testid="stStatusWidget"] {
        display: flex; justify-content: center; align-items: center; width: 100%; text-align: center;
    }
    .stSpinner > div { margin: 0 auto !important; text-align: center !important; }
    .movie-card {
        background-color: #161b22; border-radius: 12px;
        border: 1px solid #30363d; margin-bottom: 16px;
        display: flex; align-items: stretch; overflow: hidden;
    }
    .movie-card-poster {
        width: 110px; min-width: 110px;
        object-fit: cover; display: block;
    }
    .movie-card-poster-placeholder {
        width: 110px; min-width: 110px;
        background-color: #0d1117;
        display: flex; align-items: center; justify-content: center;
        font-size: 2.5rem;
    }
    .movie-card-info {
        padding: 20px 24px;
        display: flex; flex-direction: column; justify-content: center;
    }
    h1, h2, h3 { font-family: 'Quicksand', sans-serif !important; font-weight: 600 !important; color: #ffffff !important; }
    .subtitle { font-size: 0.9rem; margin-top: -20px; color: #8b949e; font-weight: 300; letter-spacing: 0.5px; }
    [data-testid="InputInstructions"] { display: none; }
    div.stButton > button {
        background-color: #00a8e1 !important; color: white !important;
        border: none !important; width: 150px !important;
        margin-left: auto !important; margin-right: 0 !important;
        display: block; border-radius: 8px !important;
    }
    .fullwidth-btn div.stButton > button {
        width: 100% !important; margin-left: 0 !important;
    }
    [data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #30363d; }
    footer {visibility: hidden;}
    div[role="radiogroup"] > label { font-size: 26px !important; gap: 10px; }
    .metric-card {
        background-color: #161b22; border-radius: 10px; border: 1px solid #30363d;
        padding: 14px 18px; text-align: center;
    }
    .metric-value { font-size: 1.6rem; font-weight: 600; color: #00a8e1; }
    .metric-label { font-size: 0.75rem; color: #8b949e; margin-top: 2px; }
    .hero-header {
        position: relative;
        padding: 32px 0 24px 0;
        margin-bottom: 8px;
        overflow: hidden;
    }
    .hero-title {
        font-family: 'Quicksand', sans-serif !important;
        font-size: 3rem; font-weight: 700;
        color: #ffffff; line-height: 1.1;
        margin: 0 0 10px 0;
    }
    .hero-subtitle {
        font-size: 1rem; color: #00a8e1;
        letter-spacing: 1px; margin: 0 0 16px 0;
        font-weight: 400;
    }
    .hero-divider {
        width: 260px; height: 2px;
        background-color: #1e2a3a; margin-bottom: 14px;
    }
    .hero-tagline {
        font-size: 0.82rem; color: #4a6080;
        letter-spacing: 1px; margin: 0;
    }
    .hero-icon {
        position: absolute; right: 0; top: 16px;
        font-size: 5rem; opacity: 0.18;
        line-height: 1;
    }
    .fav-genre-banner {
        background-color: #0d1f2d; border-left: 3px solid #00a8e1;
        border-radius: 8px; padding: 10px 16px; margin-bottom: 16px;
        font-size: 0.9rem; color: #a0c8e0;
    }
    .movie-card-match {
        background-color: #0d2e1a !important;
        border-color: #1a5c35 !important;
    }
    .match-badge {
        display: inline-block; font-size: 0.7rem; font-weight: 600;
        background-color: #1a5c35; color: #4ecb7a;
        padding: 2px 8px; border-radius: 20px; margin-top: 4px;
    }
    .compare-col-title {
        font-size: 0.85rem; font-weight: 600; color: #8b949e;
        text-transform: uppercase; letter-spacing: 1px;
        margin-bottom: 10px; text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. CARGA DE DADOS Y MOTORES
@st.cache_data
def load_data():
    df_ratings = pd.read_csv('u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    genre_names = ['unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 
                   'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 
                   'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    col_names = ['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + genre_names
    df_items = pd.read_csv('u.item', sep='|', encoding='latin-1', names=col_names)
    df_items['genre_list'] = df_items.apply(lambda row: ", ".join([g for g in genre_names if row[g] == 1]), axis=1)
    return df_ratings, df_items[['item_id', 'title', 'genre_list', 'release_date']]

@st.cache_data(show_spinner=False)
def run_soft_impute(matrix):
    from fancyimpute import SoftImpute
    solver = SoftImpute(max_iters=20, max_rank=20, init_fill_method="zero", verbose=False)
    return solver.fit_transform(matrix)

# ── TMDB: busca poster ──
@st.cache_data(show_spinner=False, ttl=86400)
def fetch_poster_url(title: str, year: str = None):
    try:
        api_key = st.secrets["TMDB_API_KEY"]
    except Exception:
        return None
    try:
        clean_title = title.split(" (")[0].strip()
        params = {"api_key": api_key, "query": clean_title, "language": "en-US", "page": 1}
        if year:
            try: params["year"] = int(str(year)[:4])
            except Exception: pass
        r = requests.get("https://api.themoviedb.org/3/search/movie", params=params, timeout=5)
        r.raise_for_status()
        results = r.json().get("results", [])
        if results and results[0].get("poster_path"):
            return "https://image.tmdb.org/t/p/w185" + results[0]["poster_path"]
    except Exception:
        pass
    return None

# ── Motor de recomendação reutilizável ──
def compute_recs(my_ratings, algo):
    min_ratings = 50
    popular_movies = df_ratings['item_id'].value_counts()
    popular_movies = popular_movies[popular_movies >= min_ratings].index
    df_ratings_filtered = df_ratings[df_ratings['item_id'].isin(popular_movies)]

    user_item_matrix = df_ratings_filtered.pivot(index='user_id', columns='item_id', values='rating')
    new_user_data = pd.Series(my_ratings, name=9999)
    user_item_matrix = pd.concat([user_item_matrix, new_user_data.to_frame().T])

    user_means = user_item_matrix.mean(axis=1)
    matrix_centered = user_item_matrix.sub(user_means, axis=0)

    if algo == "SVD":
        u, s, vt = svds(matrix_centered.fillna(0).values, k=20)
        completed_matrix = np.dot(np.dot(u, np.diag(s)), vt)
    else:
        if 'completed_matrix_soft' not in st.session_state:
            input_matrix = matrix_centered.values.astype(np.float64)
            st.session_state.completed_matrix_soft = run_soft_impute(input_matrix)
        completed_matrix = st.session_state.completed_matrix_soft

    final_preds = pd.DataFrame(completed_matrix, columns=user_item_matrix.columns, index=user_item_matrix.index)
    # ── CORREÇÃO: fallback para 0 se user_means[9999] for NaN ──
    mean_new_user = user_means[9999] if not pd.isna(user_means[9999]) else 0.0
    user_preds = (final_preds.loc[9999] + mean_new_user).sort_values(ascending=False)

    recs_df = user_preds.reset_index()
    recs_df.columns = ['item_id', 'score']
    recs = pd.merge(recs_df, df_movies, on='item_id')
    recs = recs[~recs['item_id'].isin(my_ratings.keys())].head(10)
    return recs

df_ratings, df_movies = load_data()

# 3. GESTÃO DO ESTADO DE LA SESIÓN
if 'view' not in st.session_state: st.session_state.view = 'home'
if 'my_ratings' not in st.session_state: st.session_state.my_ratings = {}
if 'current_idx' not in st.session_state: st.session_state.current_idx = 0
if 'started' not in st.session_state: st.session_state.started = False
if 'name_input' not in st.session_state: st.session_state.name_input = ""
if 'random_sample' not in st.session_state:
    st.session_state.random_sample = df_movies.sample(20).reset_index(drop=True)
if 'last_algo' not in st.session_state: st.session_state.last_algo = None

# --- BARRA LATERAL ---
st.sidebar.title("Beetle TV Stream")
selected_lang = st.sidebar.selectbox("Idioma / Language", ["Español", "Português", "English"])
algo_choice = st.sidebar.radio("Modelo / Model", ["SVD", "SoftImpute"])


# ── CORREÇÃO: invalida cache do SoftImpute se o modelo mudou ──
if st.session_state.last_algo != algo_choice:
    if 'completed_matrix_soft' in st.session_state:
        del st.session_state.completed_matrix_soft
    st.session_state.last_algo = algo_choice

texts = {
    "English": {
        "welcome": "BeetleTV", "subtitle": "Movie Discovery Engine", "header": "Rate 20 movies",
        "res": "Recommendations", "name": "Name", "age": "Age", "prev": "Previous", "next": "Next",
        "restart": "Restart", "genre_list": "Genres", "release_date": "Release Date:",
        "warning": "Enter your name.", "excellent": "Excellent, {name}!", "proceed": "Proceed",
        "loading": f"Calculating with {algo_choice}...",
        "loading_soft": "SoftImpute is more thorough — this may take a few seconds…",
        "metrics_rated": "Movies rated", "metrics_avg": "Your avg. rating", "metrics_genre": "Favorite genre",
        "compare_btn": "Compare SVD vs SoftImpute", "compare_loading": "Running both algorithms…",
        "compare_svd": "SVD", "compare_soft": "SoftImpute",
        "fav_genre_msg": "Based on your taste for {genre}…",
        "rating_hint": "💡 Tip: low ratings are just as important as high ones — be honest!",
        "hero_tagline": "Collaborative Filtering · SVD · SoftImpute · TMDB API",
        "compare_section": "Algorithm Comparison",
    },
    "Español": {
        "welcome": "BeetleTV", "subtitle": "Recomendación de Películas", "header": "Califica 20 películas",
        "res": "Recomendaciones", "name": "Nombre", "age": "Edad", "prev": "Anterior", "next": "Siguiente",
        "restart": "Reiniciar", "genre_list": "Géneros", "release_date": "Fecha de Estreno:",
        "warning": "Ingresa tu nombre.", "excellent": "¡Excelente, {name}!", "proceed": "Continuar",
        "loading": f"Calculando con {algo_choice}...",
        "loading_soft": "SoftImpute es más profundo — esto puede tardar unos segundos…",
        "metrics_rated": "Películas calificadas", "metrics_avg": "Tu nota promedio", "metrics_genre": "Género favorito",
        "compare_btn": "Comparar SVD vs SoftImpute", "compare_loading": "Ejecutando ambos algoritmos…",
        "compare_svd": "SVD", "compare_soft": "SoftImpute",
        "fav_genre_msg": "Basado en tu gusto por {genre}…",
        "rating_hint": "💡 Tip: las notas bajas son tan importantes como las altas — ¡sé honesto!",
        "hero_tagline": "Filtrado Colaborativo · SVD · SoftImpute · TMDB API",
        "compare_section": "Comparación de Algoritmos",
    },
    "Português": {
        "welcome": "BeetleTV", "subtitle": "Recomendação de Filmes", "header": "Avalie 20 filmes",
        "res": "Recomendações", "name": "Nome", "age": "Idade", "prev": "Anterior", "next": "Próximo",
        "restart": "Reiniciar", "genre_list": "Gêneros", "release_date": "Data de Lançamento:",
        "warning": "Insira seu nome.", "excellent": "Excelente, {name}!", "proceed": "Prosseguir",
        "loading": f"Calculando com {algo_choice}...",
        "loading_soft": "SoftImpute é mais completo — isso pode levar alguns segundos…",
        "metrics_rated": "Filmes avaliados", "metrics_avg": "Sua nota média", "metrics_genre": "Gênero favorito",
        "compare_btn": "Comparar SVD vs SoftImpute", "compare_loading": "Rodando os dois algoritmos…",
        "compare_svd": "SVD", "compare_soft": "SoftImpute",
        "fav_genre_msg": "Baseado no seu gosto por {genre}…",
        "rating_hint": "💡 Dica: notas baixas são tão importantes quanto altas — seja honesto!",
        "hero_tagline": "Filtragem Colaborativa · SVD · SoftImpute · TMDB API",
        "compare_section": "Comparação de Algoritmos",
    },
}
t = texts[selected_lang]

# --- VISTA: EVALUACIÓN (HOME) ---
if st.session_state.view == 'home':
    st.markdown(f'''<div class="hero-header">
        <div class="hero-icon">🎬</div>
        <div class="hero-title">{t["welcome"]}</div>
        <div class="hero-subtitle">{t["subtitle"]}</div>
        <div class="hero-divider"></div>
        <div class="hero-tagline">{t["hero_tagline"]}</div>
    </div>''', unsafe_allow_html=True)

    if not st.session_state.started:
        c1, c2 = st.columns([3, 1])
        input_name = c1.text_input(t["name"], placeholder="", key="temp_name_widget")
        c2.number_input(t["age"], 0, 120, 34, key="age_input")
        
        if st.button(t['proceed'], type="primary"):
            if input_name.strip():
                st.session_state.name_input = input_name 
                st.session_state.started = True
                st.rerun()
            else: st.warning(t['warning'])
    else:
        st.subheader(t['header'])
        st.progress((st.session_state.current_idx + 1) / 20)
        st.write(f"📊 {st.session_state.current_idx + 1} / 20")
        current_movie = st.session_state.random_sample.iloc[st.session_state.current_idx]

        release = current_movie.get('release_date', '')
        year = str(release)[:4] if release and str(release) != 'nan' else None
        poster_url = fetch_poster_url(current_movie['title'], year)

        with st.container():
            poster_html = f'<img src="{poster_url}" class="movie-card-poster" />' if poster_url else '<div class="movie-card-poster-placeholder">🎬</div>'
            st.markdown(f"""<div class="movie-card">
                {poster_html}
                <div class="movie-card-info">
                    <h3 style="margin:0 0 8px 0;">{current_movie['title']}</h3>
                    <p style='color: #8b949e; margin:0;'>📅 {t['release_date']} {current_movie['release_date']} | 🎭 {t['genre_list']}: {current_movie['genre_list']}</p>
                </div>
            </div>""", unsafe_allow_html=True)

            rating = st.radio(
                "",
                ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"],
                key=f"star_{st.session_state.current_idx}",
                horizontal=True,
                index=None
            )

            st.caption(t['rating_hint'])

            if rating is not None:
                st.session_state.my_ratings[current_movie['item_id']] = len(rating)
                if st.session_state.current_idx < 19:
                    st.session_state.current_idx += 1
                    st.rerun()
                else:
                    st.session_state.view = 'results'
                    st.rerun()

        col_prev, col_next = st.columns([1, 1])
        if col_prev.button(f"⬅️ {t['prev']}") and st.session_state.current_idx > 0:
            st.session_state.current_idx -= 1
            st.rerun()
        if col_next.button(f"{t['next']} ➡️") and st.session_state.current_idx < 19:
            st.session_state.current_idx += 1
            st.rerun()

# --- VISTA: RESULTADOS ---
elif st.session_state.view == 'results':
    st.title(t['res'])
    display_name = st.session_state.get('name_input', 'Douglas')

    # ── CORREÇÃO: avisa se nenhum filme foi avaliado ──
    if not st.session_state.my_ratings:
        st.warning("Nenhum filme foi avaliado. Volte e avalie pelo menos um filme!" if selected_lang == "Português" else "¡No calificaste ninguna película. ¡Vuelve y califica al menos una!" if selected_lang == "Español" else "No movies were rated. Go back and rate at least one!")
        if st.button("← Voltar" if selected_lang == "Português" else "← Volver" if selected_lang == "Español" else "← Back"):
            st.session_state.view = 'home'
            st.rerun()
        st.stop()

    # ── NOVO 1: loading informativo para SoftImpute ──
    loading_msg = t['loading_soft'] if algo_choice == "SoftImpute" else t['loading']
    with st.spinner(loading_msg):
        recs = compute_recs(st.session_state.my_ratings, algo_choice)

    st.success(t['excellent'].format(name=display_name))

    # ── NOVO 2: métricas do perfil do usuário ──
    rated_ids = list(st.session_state.my_ratings.keys())
    rated_scores = list(st.session_state.my_ratings.values())
    avg_rating = round(sum(rated_scores) / len(rated_scores), 1)

    # detecta gênero favorito (mais frequente nos filmes bem avaliados >= 4)
    good_ids = [iid for iid, s in st.session_state.my_ratings.items() if s >= 4]
    fav_genre = "—"
    if good_ids:
        good_movies = df_movies[df_movies['item_id'].isin(good_ids)]
        all_genres = [g.strip() for gl in good_movies['genre_list'] for g in gl.split(',') if g.strip() and g.strip() != 'unknown']
        if all_genres:
            fav_genre = max(set(all_genres), key=all_genres.count)

    m1, m2, m3 = st.columns(3)
    m1.markdown(f'<div class="metric-card"><div class="metric-value">{len(rated_ids)}</div><div class="metric-label">{t["metrics_rated"]}</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="metric-card"><div class="metric-value">{"⭐" * round(avg_rating)}</div><div class="metric-label">{t["metrics_avg"]}: {avg_rating}</div></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="metric-card"><div class="metric-value" style="font-size:1rem;">{fav_genre}</div><div class="metric-label">{t["metrics_genre"]}</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── NOVO 3: highlight do gênero favorito ──
    if fav_genre != "—":
        st.markdown(f'<div class="fav-genre-banner">🎭 {t["fav_genre_msg"].format(genre=fav_genre)}</div>', unsafe_allow_html=True)

    # recomendações com poster
    for _, row in recs.iterrows():
        release = row.get('release_date', '')
        year = str(release)[:4] if release and str(release) != 'nan' else None
        poster_url = fetch_poster_url(row['title'], year)
        poster_html = f'<img src="{poster_url}" class="movie-card-poster" />' if poster_url else '<div class="movie-card-poster-placeholder">🎬</div>'
        st.markdown(f"""<div class="movie-card">
            {poster_html}
            <div class="movie-card-info">
                <b style="font-size:1rem;">{row['title']}</b><br>
                <small style='color: #8b949e;'>🎭 {row['genre_list']} | 📅 {row['release_date']}</small>
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Botão Reiniciar ──
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button(t['restart'], use_container_width=True):
        st.session_state.view = 'home'
        st.session_state.my_ratings = {}
        st.session_state.current_idx = 0
        st.session_state.started = False
        st.session_state.show_compare = False
        if 'completed_matrix_soft' in st.session_state:
            del st.session_state.completed_matrix_soft
        st.session_state.random_sample = df_movies.sample(20).reset_index(drop=True)
        st.rerun()

    # ── NOVO 4: comparação SVD vs SoftImpute ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---", unsafe_allow_html=False)
    st.subheader(t['compare_section'])
    st.markdown('<div class="fullwidth-btn">', unsafe_allow_html=True)
    if st.button(t['compare_btn'], use_container_width=True):
        st.session_state.show_compare = True
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.get('show_compare', False):
        with st.spinner(t['compare_loading']):
            recs_svd = compute_recs(st.session_state.my_ratings, "SVD")
            recs_soft = compute_recs(st.session_state.my_ratings, "SoftImpute")

        # filmes em comum entre os dois algoritmos
        common_ids = set(recs_svd['item_id'].tolist()) & set(recs_soft['item_id'].tolist())
        match_label = "✓ Match" if selected_lang == "English" else "✓ Coincidencia" if selected_lang == "Español" else "✓ Coincidência"

        col_svd, col_soft = st.columns(2)
        with col_svd:
            st.markdown(f'<div class="compare-col-title">🔵 {t["compare_svd"]}</div>', unsafe_allow_html=True)
            for _, row in recs_svd.iterrows():
                is_match = row['item_id'] in common_ids
                card_class = "movie-card movie-card-match" if is_match else "movie-card"
                badge = f'<span class="match-badge">{match_label}</span>' if is_match else ""
                st.markdown(f'<div class="{card_class}"><div class="movie-card-info"><b style="font-size:0.9rem;">{row["title"]}</b><br><small style="color:#8b949e;">{row["genre_list"]}</small>{badge}</div></div>', unsafe_allow_html=True)
        with col_soft:
            st.markdown(f'<div class="compare-col-title">🟠 {t["compare_soft"]}</div>', unsafe_allow_html=True)
            for _, row in recs_soft.iterrows():
                is_match = row['item_id'] in common_ids
                card_class = "movie-card movie-card-match" if is_match else "movie-card"
                badge = f'<span class="match-badge">{match_label}</span>' if is_match else ""
                st.markdown(f'<div class="{card_class}"><div class="movie-card-info"><b style="font-size:0.9rem;">{row["title"]}</b><br><small style="color:#8b949e;">{row["genre_list"]}</small>{badge}</div></div>', unsafe_allow_html=True)