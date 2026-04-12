import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds


# 1. CONFIGURACIONES VISUALES E INTERFAZ
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
        background-color: #161b22; padding: 24px; border-radius: 12px;
        border: 1px solid #30363d; margin-bottom: 16px;
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
    [data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #30363d; }
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# 2. CARGA DE DATOS Y MOTORES
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

# ✅ Ajuste aqui (cache_data)
@st.cache_data(show_spinner=False)
def run_soft_impute(matrix):
    from fancyimpute import SoftImpute
    solver = SoftImpute(verbose=False)
    return solver.fit_transform(matrix)

df_ratings, df_movies = load_data()

# 3. GESTIÓN DEL ESTADO DE LA SESIÓN
if 'view' not in st.session_state: st.session_state.view = 'home'
if 'my_ratings' not in st.session_state: st.session_state.my_ratings = {}
if 'current_idx' not in st.session_state: st.session_state.current_idx = 0
if 'started' not in st.session_state: st.session_state.started = False
if 'name_input' not in st.session_state: st.session_state.name_input = ""
if 'random_sample' not in st.session_state:
    st.session_state.random_sample = df_movies.sample(20).reset_index(drop=True)

# --- BARRA LATERAL ---
st.sidebar.title("Beetle TV Stream")
selected_lang = st.sidebar.selectbox("Idioma / Language", ["Español", "Português", "English"])
algo_choice = st.sidebar.radio("Modelo / Model", ["SVD", "SoftImpute"])

texts = {
    "English": {"welcome": "Welcome", "subtitle": "Movie Discovery Engine", "header": "Rate 20 movies", "res": "Recommendations", "name": "Name", "age": "Age", "prev": "Previous", "next": "Next", "restart": "Restart","genre_list": "Genres","release_date": "Release Date:", "warning": "Enter your name.", "excellent": "Excellent, {name}!", "proceed": "Proceed", "loading": f"Calculating with {algo_choice}..."},
    "Español": {"welcome": "Bienvenido", "subtitle": "Recomendación de Películas", "header": "Califica 20 películas", "res": "Recomendaciones", "name": "Nombre", "age": "Edad", "prev": "Anterior", "next": "Siguiente", "restart": "Reiniciar","genre_list": "Géneros","release_date": "Fecha de Estreno:", "warning": "Ingresa tu nombre.", "excellent": "¡Excelente, {name}!", "proceed": "Continuar", "loading": f"Calculando con {algo_choice}..."},
    "Português": {"welcome": "Bem-vindo", "subtitle": "Recomendação de Filmes", "header": "Avalie 20 filmes", "res": "Recomendações", "name": "Nome", "age": "Idade", "prev": "Anterior", "next": "Próximo", "restart": "Reiniciar","genre_list": "Gêneros","release_date": "Data de Lançamento:", "warning": "Insira seu nome.", "excellent": "Excelente, {name}!", "proceed": "Prosseguir", "loading": f"Calculando com {algo_choice}..."}
}
t = texts[selected_lang]

# --- VISTA: EVALUACIÓN (HOME) ---
if st.session_state.view == 'home':
    st.title(t['welcome'])
    st.markdown(f'<p class="subtitle">{t["subtitle"]}</p>', unsafe_allow_html=True)

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
        
        with st.container():
            st.markdown(f"""<div class="movie-card">
                <h3>{current_movie['title']}</h3>
                <p style='color: #8b949e;'>📅 {t['release_date']} {current_movie['release_date']} | 🎭 {t['genre_list']}: {current_movie['genre_list']}</p>
                </div>""", unsafe_allow_html=True)
            rating = st.feedback("stars", key=f"star_{st.session_state.current_idx}")
            if rating is not None:
                st.session_state.my_ratings[current_movie['item_id']] = rating + 1
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

    with st.spinner(t['loading']):
        user_item_matrix = df_ratings.pivot(index='user_id', columns='item_id', values='rating')
        new_user_data = pd.Series(st.session_state.my_ratings, name=9999)
        user_item_matrix = pd.concat([user_item_matrix, new_user_data.to_frame().T])

        user_means = user_item_matrix.mean(axis=1)
        matrix_centered = user_item_matrix.sub(user_means, axis=0)

        if algo_choice == "SVD":
            u, s, vt = svds(matrix_centered.fillna(0).values, k=20)
            completed_matrix = np.dot(np.dot(u, np.diag(s)), vt)
        else:
            if 'completed_matrix_soft' not in st.session_state:
                input_matrix = matrix_centered.values.astype(np.float64)
                st.session_state.completed_matrix_soft = run_soft_impute(input_matrix)

            completed_matrix = st.session_state.completed_matrix_soft

        final_preds = pd.DataFrame(completed_matrix, columns=user_item_matrix.columns, index=user_item_matrix.index)
        user_preds = (final_preds.loc[9999] + user_means[9999]).sort_values(ascending=False)
        
        recs_df = user_preds.reset_index()
        recs_df.columns = ['item_id', 'score']
        recs = pd.merge(recs_df, df_movies, on='item_id')
        recs = recs[~recs['item_id'].isin(st.session_state.my_ratings.keys())].head(10)

    st.success(t['excellent'].format(name=display_name))

    for _, row in recs.iterrows():
        st.markdown(f"""<div class="movie-card">
            <b>{row['title']}</b><br>
            <small style='color: #8b949e;'>🎭 {row['genre_list']} | 📅 {row['release_date']}</small>
            </div>""", unsafe_allow_html=True)

    if st.button(t['restart'], use_container_width=True):
        st.session_state.view = 'home'
        st.session_state.my_ratings = {}
        st.session_state.current_idx = 0
        st.session_state.started = False
        if 'completed_matrix_soft' in st.session_state:
            del st.session_state.completed_matrix_soft
        st.session_state.random_sample = df_movies.sample(20).reset_index(drop=True)
        st.rerun()