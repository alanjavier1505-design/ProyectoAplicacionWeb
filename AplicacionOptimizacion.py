import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sympy as sp
from scipy.optimize import line_search
import pandas as pd
import time

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Calculadora de Optimización", page_icon="🌌", layout="wide")

# --- ESTILOS GLOBALES ---
st.markdown("""
<style>
    .stApp { background-color: #0D1B2A; }
    [data-testid="stSidebar"] { background-color: #1A2F4A; }
    [data-testid="stSidebar"] * { color: #E8D5A3 !important; }
    html, body, [class*="css"], p, span, div, label { color: #F0F4F8; }
    h1, h2, h3, h4 { color: #E8C547 !important; }

    .stButton > button {
        background-color: #1A3A5C; color: #E8C547;
        border: 1px solid #E8C547; border-radius: 8px;
        font-weight: 600; transition: all 0.2s;
    }
    .stButton > button:hover { background-color: #E8C547; color: #0D1B2A; }

    input, textarea,
    [data-baseweb="input"] input,
    [data-baseweb="base-input"] input,
    .stTextInput input, .stNumberInput input {
        color: #0D1B2A !important;
        font-weight: 500 !important;
        caret-color: #0D1B2A !important;
    }
    [data-baseweb="select"] *,
    [data-baseweb="select"] span,
    [data-baseweb="select"] div,
    [data-baseweb="select"] p,
    .stSelectbox * { color: #0D1B2A !important; }
    [data-baseweb="popover"] *,
    [data-baseweb="menu"] *,
    [role="listbox"] *,
    [role="option"] * {
        color: #0D1B2A !important;
        background-color: #F0F4F8 !important;
    }
    [data-baseweb="menu"] [aria-selected="true"],
    [data-baseweb="menu"] [aria-selected="true"] * {
        background-color: #A8C4E0 !important;
        color: #0D1B2A !important;
    }
    .stNumberInput button,
    .stNumberInput button *,
    button[data-testid="stNumberInput-StepDown"],
    button[data-testid="stNumberInput-StepUp"] {
        background-color: #2E5077 !important;
        color: #F0F4F8 !important;
        border: none !important;
    }
    button[data-testid="stNumberInput-StepDown"]:hover,
    button[data-testid="stNumberInput-StepUp"]:hover {
        background-color: #E8C547 !important;
        color: #0D1B2A !important;
    }
    [data-testid="stMetric"] {
        background-color: #1A2F4A; border: 1px solid #2E5077; border-radius: 10px; padding: 12px;
    }
    [data-testid="stMetricLabel"] { color: #A8C4E0 !important; }
    [data-testid="stMetricValue"] { color: #E8C547 !important; }

    /* Métricas de rendimiento en azul claro */
    .metric-rendimiento [data-testid="stMetric"] {
        background-color: #0F2D45 !important;
        border: 1px solid #5BA4CF !important;
    }
    .metric-rendimiento [data-testid="stMetricValue"] {
        color: #5BA4CF !important;
    }

    .streamlit-expanderHeader { background-color: #1A2F4A; color: #E8C547 !important; border-radius: 8px; }
    .streamlit-expanderContent { background-color: #132338; border: 1px solid #2E5077; }
    hr { border-color: #2E5077; }
    [data-testid="stImage"] img {
        max-height: 220px; width: 100%;
        object-fit: cover; object-position: center 40%;
        border-radius: 12px; margin-bottom: 0.5rem;
    }
    .stCaption { color: #A8C4E0 !important; }
    .stAlert { background-color: #1A2F4A; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# --- BANNER ---
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg", use_container_width=True)

# --- TÍTULO ---
st.markdown("""
<h1 style='text-align: center; color: #E8C547; font-size: 2.4rem; margin-top: 0.2rem; margin-bottom: 1.2rem;'>
    🌌 Calculadora de Optimización
</h1>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'n_vars' not in st.session_state:
    st.session_state.n_vars = 2
if 'func_str' not in st.session_state:
    st.session_state.func_str = "x1**2 + x2**2"
if 'p0_str' not in st.session_state:
    st.session_state.p0_str = "5.0, 5.0"

def cargar_ejemplo(n, f, p):
    st.session_state.n_vars = n
    st.session_state.func_str = f
    st.session_state.p0_str = p

# --- MANUAL DE USUARIO ---
with st.expander("📖 Manual de Uso: ¿Cómo escribir las funciones matemáticas?"):
    st.markdown("""
    Para que la calculadora entienda correctamente tu función objetivo, debes usar la notación estándar asignada a esta calculadora. Aquí tienes una guía rápida:

    * **Variables:** Usa `x1`, `x2`, `x3`, etc.
    * **Sumas y restas:** Usa `+` y `-` (Ej: `x1 + x2`)
    * **Multiplicación explícita:** Usa siempre el asterisco `*`. **No** escribas `2x1`, debes escribir obligatoriamente `2*x1`.
    * **Potencias:** Usa doble asterisco `**`. **No** escribas el número al lado (ej: `x12`). Escribe `x1**2` para referirte a $x_1^2$.
    * **Exponencial (Euler):** Usa `exp()`. Ej: `exp(-x1**2)` para referirte a $e^{-x_1^2}$.
    * **Trigonometría:** Usa `sin(x)`, `cos(x)`, `tan(x)`.
    * **Logaritmos:**
        * Logaritmo natural (base $e$): usa `log(x)`. Ej: `log(x1)` para referirte a $\\ln(x_1)$.
        * Logaritmo base 10: usa `log(x, 10)`. Ej: `log(x1, 10)` para referirte a $\\log_{10}(x_1)$.
        * Logaritmo base 2: usa `log(x, 2)`. Ej: `log(x1, 2)` para referirte a $\\log_2(x_1)$.
        * ⚠️ El argumento del logaritmo debe ser **estrictamente positivo**. Asegúrate de que el punto de partida y la trayectoria no pasen por valores ≤ 0.
    * **Raíces:**
        * Raíz cuadrada: usa `sqrt(x)`. Ej: `sqrt(x1)` para referirte a $\\sqrt{x_1}$.
        * Raíz n-ésima: usa `x**(1/n)`. Ej: `x1**(1/3)` para referirte a $\\sqrt[3]{x_1}$.
        * ⚠️ El argumento de `sqrt()` y de cualquier raíz con exponente fraccionario debe ser **estrictamente positivo**. Si el algoritmo evalúa un punto negativo durante la trayectoria, el resultado será un número complejo y la optimización fallará. **No uses** `raiz()` ni `√` ya que no son sintaxis válidas.
    """)
    st.markdown("---")
    st.markdown("**Ejemplos de funciones de prueba clásicas (Haz clic para cargar en el panel izquierdo):**")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("🔹 **Función Cuadrática simple:** `x1**2 + x2**2`")
    with col2:
        st.button("Probar Cuadrática", on_click=cargar_ejemplo, args=(2, "x1**2 + x2**2", "5.0, 5.0"), use_container_width=True)

    col3, col4 = st.columns([3, 1])
    with col3:
        st.write("🔹 **Función de Rosenbrock:** `(1 - x1)**2 + 100 * (x2 - x1**2)**2`")
    with col4:
        st.button("Probar Rosenbrock", on_click=cargar_ejemplo, args=(2, "(1 - x1)**2 + 100 * (x2 - x1**2)**2", "-1.2, 1.0"), use_container_width=True)

    col5, col6 = st.columns([3, 1])
    with col5:
        st.write("🔹 **Función de Booth:** `(x1 + 2*x2 - 7)**2 + (2*x1 + x2 - 5)**2`")
    with col6:
        st.button("Probar Booth", on_click=cargar_ejemplo, args=(2, "(x1 + 2*x2 - 7)**2 + (2*x1 + x2 - 5)**2", "0.0, 0.0"), use_container_width=True)

    col7, col8 = st.columns([3, 1])
    with col7:
        st.write("🔹 **Pozo de Gauss (con Euler):** `-exp(-x1**2 - x2**2)`")
    with col8:
        st.button("Probar Gauss", on_click=cargar_ejemplo, args=(2, "-exp(-x1**2 - x2**2)", "1.0, 1.0"), use_container_width=True)

# --- SIDEBAR ---
st.sidebar.markdown("<h2 style='color:#E8C547;font-size:1.2rem;'>🛠️ Parámetros de Entrada</h2>", unsafe_allow_html=True)
num_vars = st.sidebar.number_input("Número de variables", min_value=1, max_value=10, key="n_vars")
metodo = st.sidebar.selectbox("Método de Optimización",
                              ["Gradiente Descendente", "Gradiente Conjugado", "Método de Newton"])
st.sidebar.markdown("<small style='color:#A8C4E0;'>Ej: x1**2 + x2**2, exp(-x1**2)</small>", unsafe_allow_html=True)
funcion_str = st.sidebar.text_input("Función objetivo f(x)", key="func_str")
punto_inicio = st.sidebar.text_input("Punto de partida (separado por comas)", key="p0_str")

st.sidebar.markdown("<h3 style='color:#E8C547;font-size:1rem;margin-top:1rem;'>🎯 Criterios de Parada</h3>", unsafe_allow_html=True)
max_iter = st.sidebar.number_input("Número máximo de iteraciones", min_value=1, value=100)
tolerancia = st.sidebar.number_input("Tolerancia de convergencia", value=1e-5, format="%.1e")

st.sidebar.markdown("<h3 style='color:#E8C547;font-size:1rem;margin-top:1rem;'>📏 Búsqueda de Línea y Wolfe</h3>", unsafe_allow_html=True)
alpha_init = st.sidebar.number_input("Alfa Inicial (α0)", min_value=0.01, max_value=10.0, value=1.0, step=0.1)
alpha_min = st.sidebar.number_input("Alfa Mínimo (α min)", min_value=1e-9, max_value=0.1, value=1e-4, format="%.1e")
c1 = st.sidebar.slider("Parámetro c1 (Armijo)", 0.0001, 0.5, 1e-4, format="%.4f")
c2 = st.sidebar.slider("Parámetro c2 (Curvatura)", c1, 0.99, 0.9)

st.sidebar.markdown("<h3 style='color:#E8C547;font-size:1rem;margin-top:1rem;'>👁️ Visualización de Tabla</h3>", unsafe_allow_html=True)
rango_tabla = st.sidebar.slider(
    "Rango de iteraciones a mostrar:",
    min_value=0, max_value=int(max_iter),
    value=(0, min(20, int(max_iter)))
)
min_iter_visual = rango_tabla[0]
max_iter_visual = rango_tabla[1]

# --- EJECUCIÓN ---
if st.button("🚀 Ejecutar Optimización"):
    try:
        x0 = np.array([float(x.strip()) for x in punto_inicio.split(',')])
        if len(x0) != num_vars:
            st.error(f"El punto de partida debe tener {num_vars} coordenadas.")
            st.stop()

        with st.spinner("Calculando derivadas y ejecutando optimización..."):
            vars_sym = sp.symbols(f'x1:{num_vars+1}')
            func_str_clean = funcion_str.replace('sen', 'sin').replace('^', '**')
            f_sym = sp.sympify(func_str_clean)
            grad_sym = [sp.diff(f_sym, var) for var in vars_sym]
            hessian_sym = [[sp.diff(g, var) for var in vars_sym] for g in grad_sym]
            f_num = sp.lambdify(vars_sym, f_sym, 'numpy')
            grad_num = sp.lambdify(vars_sym, grad_sym, 'numpy')
            hessian_num = sp.lambdify(vars_sym, hessian_sym, 'numpy')

            # Contadores de evaluaciones
            conteo_f    = 0
            conteo_grad = 0
            conteo_hess = 0

            def f_eval(x):
                nonlocal conteo_f
                conteo_f += 1
                return float(f_num(*x))

            def grad_eval(x):
                nonlocal conteo_grad
                conteo_grad += 1
                return np.array(grad_num(*x), dtype=float)

            def hess_eval(x):
                nonlocal conteo_hess
                conteo_hess += 1
                return np.array(hessian_num(*x), dtype=float)

            x_k = x0.copy()
            historial_datos = []
            trayectoria_x = []
            p_old = None
            g_old = None
            iters_realizadas = 0
            error_final = 0.0

            # --- INICIO CRONÓMETRO ---
            t_inicio = time.perf_counter()

            for k in range(max_iter):
                g_k = grad_eval(x_k)
                f_k = f_eval(x_k)
                error_actual = np.linalg.norm(g_k)
                trayectoria_x.append(x_k.copy())
                historial_datos.append({"Iteración": k, "x": x_k.copy(), "f(x)": f_k, "Error": error_actual})

                if error_actual < tolerancia:
                    error_final = error_actual
                    break

                if metodo == "Gradiente Descendente":
                    p_k = -g_k
                elif metodo == "Método de Newton":
                    H_k = hess_eval(x_k)
                    try:
                        p_k = np.linalg.solve(H_k, -g_k)
                    except np.linalg.LinAlgError:
                        p_k = np.dot(np.linalg.pinv(H_k), -g_k)
                elif metodo == "Gradiente Conjugado":
                    if k == 0 or k % num_vars == 0:
                        p_k = -g_k
                    else:
                        denom = np.dot(g_old, g_old)
                        beta_k = 0.0 if denom < 1e-30 else max(0.0, np.dot(g_k, g_k - g_old) / denom)
                        p_k = -g_k + beta_k * p_old

                res_alpha = line_search(f_eval, grad_eval, x_k, p_k, gfk=g_k, old_fval=f_k, c1=c1, c2=c2)
                alpha_k = res_alpha[0]
                if alpha_k is None or alpha_k < alpha_min:
                    alpha_k = alpha_min

                p_old = p_k
                g_old = g_k
                x_k = x_k + alpha_k * p_k
                iters_realizadas += 1
                error_final = error_actual

            # --- FIN CRONÓMETRO ---
            t_fin = time.perf_counter()
            tiempo_total = t_fin - t_inicio
            tiempo_por_iter = tiempo_total / iters_realizadas if iters_realizadas > 0 else 0.0

            trayectoria_x.append(x_k.copy())

        # --- RESULTADOS ---
        st.success("✅ Optimización finalizada con éxito.")
        st.balloons()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Punto mínimo (x*)", f"[{', '.join([f'{x:.4f}' for x in x_k])}]")
        col2.metric("Valor en el mínimo f(x*)", f"{f_eval(x_k):.4e}")
        col3.metric("Iteraciones", iters_realizadas)
        col4.metric("Error Final", f"{error_final:.2e}")

        df_historial = pd.DataFrame(historial_datos)

        # --- RENDIMIENTO COMPUTACIONAL ---
        st.markdown("---")
        st.subheader("⚡ Rendimiento Computacional")

        st.markdown("<div class='metric-rendimiento'>", unsafe_allow_html=True)
        rc1, rc2, rc3, rc4, rc5 = st.columns(5)
        rc1.metric("⏱️ Tiempo total", f"{tiempo_total*1000:.2f} ms")
        rc2.metric("⏱️ Tiempo por iteración", f"{tiempo_por_iter*1000:.3f} ms")
        rc3.metric("📊 Evaluaciones f(x)", conteo_f)
        rc4.metric("📐 Evaluaciones ∇f", conteo_grad)
        rc5.metric(
            "🔢 Evaluaciones H",
            conteo_hess if metodo == "Método de Newton" else "— (no aplica)"
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if metodo != "Método de Newton":
            st.caption("ℹ️ El Hessiano solo se evalúa en el Método de Newton. Gradiente Descendente y Conjugado son métodos de primer orden.")

        # --- GRÁFICO DE CONVERGENCIA ---
        st.markdown("---")
        st.subheader("📈 Gráfico de Convergencia")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_historial["Iteración"], y=df_historial["Error"],
            mode='lines+markers', name='||∇f(x)||',
            marker=dict(color='#E8C547'), line=dict(color='#E8C547')
        ))
        fig.update_layout(
            title="Error vs Número de Iteraciones",
            xaxis_title="Iteración", yaxis_title="Error", yaxis_type="log",
            paper_bgcolor='#0D1B2A', plot_bgcolor='#132338',
            font=dict(color='#F0F4F8'), title_font=dict(color='#E8C547'),
            xaxis=dict(gridcolor='#2E5077'), yaxis=dict(gridcolor='#2E5077')
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- MAPA DE CONTORNO ---
        if num_vars == 2:
            st.markdown("---")
            st.subheader("🗺️ Mapa de Contorno con Trayectoria")

            tray = np.array(trayectoria_x)
            x1_tray = tray[:, 0]
            x2_tray = tray[:, 1]

            margen = max(abs(x0[0] - x_k[0]), abs(x0[1] - x_k[1])) * 0.4 + 1.5
            x1_grid = np.linspace(min(x1_tray.min(), x_k[0]) - margen, max(x1_tray.max(), x_k[0]) + margen, 120)
            x2_grid = np.linspace(min(x2_tray.min(), x_k[1]) - margen, max(x2_tray.max(), x_k[1]) + margen, 120)
            X1, X2 = np.meshgrid(x1_grid, x2_grid)

            Z = np.zeros_like(X1)
            for i in range(120):
                for j in range(120):
                    try:
                        Z[i, j] = f_eval([X1[i, j], X2[i, j]])
                    except Exception:
                        Z[i, j] = np.nan

            z_finite = Z[np.isfinite(Z)]
            if len(z_finite) > 0:
                Z = np.clip(Z, np.percentile(z_finite, 2), np.percentile(z_finite, 98))

            fig_contour = go.Figure()
            fig_contour.add_trace(go.Contour(
                x=x1_grid, y=x2_grid, z=Z,
                colorscale='Blues',
                contours=dict(showlabels=True, labelfont=dict(size=10, color='#E8C547')),
                colorbar=dict(
                    title=dict(text='f(x)', font=dict(color='#E8C547')),
                    thickness=14, tickfont=dict(color='#F0F4F8')
                ),
                name='f(x1, x2)'
            ))
            fig_contour.add_trace(go.Scatter(
                x=x1_tray, y=x2_tray, mode='lines+markers',
                line=dict(color='#E8C547', width=2, dash='dot'),
                marker=dict(color='#E8C547', size=6, symbol='circle', line=dict(color='#0D1B2A', width=1)),
                name='Trayectoria'
            ))
            fig_contour.add_trace(go.Scatter(
                x=[x0[0]], y=[x0[1]], mode='markers+text',
                marker=dict(color='#5BA4CF', size=14, symbol='diamond', line=dict(color='white', width=2)),
                text=['Inicio'], textposition='top right',
                textfont=dict(color='#5BA4CF', size=12), name='Punto de inicio'
            ))
            fig_contour.add_trace(go.Scatter(
                x=[x_k[0]], y=[x_k[1]], mode='markers+text',
                marker=dict(color='#E8C547', size=16, symbol='star', line=dict(color='white', width=2)),
                text=['x*'], textposition='top right',
                textfont=dict(color='#E8C547', size=13), name='Mínimo encontrado'
            ))
            fig_contour.update_layout(
                title=f"Curvas de nivel de f(x) — trayectoria {metodo}",
                xaxis_title='x1', yaxis_title='x2',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                            font=dict(color='#F0F4F8')),
                height=520,
                paper_bgcolor='#0D1B2A', plot_bgcolor='#132338',
                font=dict(color='#F0F4F8'), title_font=dict(color='#E8C547'),
                xaxis=dict(gridcolor='#2E5077'), yaxis=dict(gridcolor='#2E5077')
            )
            st.plotly_chart(fig_contour, use_container_width=True)
            st.caption("⬦ Inicio  ★ Mínimo encontrado  · · · Trayectoria del algoritmo")
        else:
            st.info("ℹ️ El mapa de contorno está disponible solo para funciones de 2 variables.")

        # --- CLASIFICACIÓN DEL PUNTO CRÍTICO ---
        st.markdown("---")
        st.subheader("🔬 Clasificación del Punto Crítico")

        H_final = hess_eval(x_k)
        eigenvalues_real = np.real(np.linalg.eigvals(H_final))
        det_H = np.linalg.det(H_final)
        traza_H = np.trace(H_final)

        todos_positivos = np.all(eigenvalues_real > 1e-10)
        todos_negativos = np.all(eigenvalues_real < -1e-10)
        hay_mixtos      = np.any(eigenvalues_real > 1e-10) and np.any(eigenvalues_real < -1e-10)
        hay_cero        = np.any(np.abs(eigenvalues_real) <= 1e-10)

        if todos_positivos:
            tipo, color_box = "✅ Mínimo local", "success"
            explicacion = "Todos los valores propios del Hessiano son **positivos**, lo que indica que la función es localmente convexa en x*. El punto encontrado es un **mínimo local** (o global si la función es convexa)."
        elif todos_negativos:
            tipo, color_box = "⚠️ Máximo local", "warning"
            explicacion = "Todos los valores propios del Hessiano son **negativos**, lo que indica que la función es localmente cóncava en x*. El punto encontrado es un **máximo local**, no un mínimo."
        elif hay_mixtos:
            tipo, color_box = "❌ Punto de silla", "error"
            explicacion = "Los valores propios del Hessiano tienen **signos mixtos** (positivos y negativos). El punto encontrado es un **punto de silla**: mínimo en algunas direcciones y máximo en otras. No es un mínimo real de la función."
        elif hay_cero:
            tipo, color_box = "⚠️ Punto crítico degenerado", "warning"
            explicacion = "Al menos un valor propio del Hessiano es **cercano a cero**. El test de segunda derivada es **inconcluso**: se necesita análisis de orden superior para determinar si es mínimo, máximo o punto de silla."
        else:
            tipo, color_box = "❓ No determinado", "info"
            explicacion = "No se pudo clasificar el punto crítico con la información disponible."

        if color_box == "success":   st.success(f"**Tipo de punto crítico:** {tipo}")
        elif color_box == "warning": st.warning(f"**Tipo de punto crítico:** {tipo}")
        elif color_box == "error":   st.error(f"**Tipo de punto crítico:** {tipo}")
        else:                        st.info(f"**Tipo de punto crítico:** {tipo}")

        st.markdown(explicacion)

        col_h1, col_h2, col_h3 = st.columns(3)
        col_h1.metric("Determinante det(H)", f"{det_H:.4e}")
        col_h2.metric("Traza tr(H)", f"{traza_H:.4e}")
        col_h3.metric("Nº de variables", num_vars)

        st.dataframe(pd.DataFrame({
            "Valor propio λ": [f"{v:.6e}" for v in eigenvalues_real],
            "Signo": ["+" if v > 1e-10 else ("−" if v < -1e-10 else "≈0") for v in eigenvalues_real]
        }), hide_index=True, use_container_width=True)

        # --- MATEMÁTICA SIMBÓLICA ---
        st.markdown("---")
        st.subheader("🧬 Análisis Matemático Avanzado")
        st.write("La aplicación dedujo las siguientes expresiones utilizando diferenciación automática:")
        col_math1, col_math2 = st.columns(2)
        with col_math1:
            st.latex(r"\nabla f(x) = " + sp.latex(grad_sym))
        with col_math2:
            st.latex(r"H(x) = " + sp.latex(hessian_sym))

        # --- HISTORIAL ---
        st.markdown("---")
        st.subheader("📋 Historial de Iteraciones")
        for i in range(num_vars):
            df_historial[f"x{i+1}"] = df_historial["x"].apply(lambda coord: coord[i])
        df_historial = df_historial.drop(columns=["x"])
        tabla_filtrada = df_historial[
            (df_historial["Iteración"] >= min_iter_visual) &
            (df_historial["Iteración"] <= max_iter_visual)
        ]
        st.dataframe(tabla_filtrada, hide_index=True, use_container_width=True)

    except Exception as e:
        st.error(f"Ocurrió un error matemático o de sintaxis: {e}")
        st.info("💡 Asegúrate de que la función esté bien escrita. Puedes revisar el 'Manual de Uso' en la parte superior.")
