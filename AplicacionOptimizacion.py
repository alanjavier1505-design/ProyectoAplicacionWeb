import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sympy as sp
from scipy.optimize import line_search
import pandas as pd

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Optimizador de Funciones", page_icon="⚙️", layout="wide")
st.title("⚙️ Calculadora de Optimización No Lineal")

# --- MEMORIA DE LA PÁGINA (SESSION STATE) ---
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

# --- MANUAL DE USUARIO (PLEGABLE) ---
with st.expander("📖 Manual de Uso: ¿Cómo escribir las funciones matemáticas?"):
    st.markdown("""
    Para que la calculadora entienda correctamente tu función objetivo, debes usar la notación estándar asignada a esta calculadora. Aquí tienes una guía rápida:

    * **Variables:** Usa `x1`, `x2`, `x3`, etc.
    * **Sumas y restas:** Usa `+` y `-` (Ej: `x1 + x2`)
    * **Multiplicación explícita:** Usa siempre el asterisco `*`. **No** escribas `2x1`, debes escribir obligatoriamente `2*x1`.
    * **Potencias:** Usa doble asterisco `**`. **No** escribas el número al lado (ej: `x12`). Escribe `x1**2` para referirte a $x_1^2$.
    * **Exponencial (Euler):** Usa `exp()`. Ej: `exp(-x1**2)` para referirte a $e^{-x_1^2}$.
    * **Trigonometría:** Usa `sin(x)`, `cos(x)`, `tan(x)`.
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

# --- BARRA LATERAL: DATOS DE ENTRADA ---
st.sidebar.header("🛠️ Parámetros de Entrada")

num_vars = st.sidebar.number_input("Número de variables", min_value=1, max_value=10, key="n_vars")
metodo = st.sidebar.selectbox("Método de Optimización", 
                              ["Gradiente Descendente", "Gradiente Conjugado", "Método de Newton"])

st.sidebar.markdown("<small>Ej: x1**2 + x2**2, exp(-x1**2)</small>", unsafe_allow_html=True)
funcion_str = st.sidebar.text_input("Función objetivo f(x)", key="func_str")
punto_inicio = st.sidebar.text_input("Punto de partida (separado por comas)", key="p0_str")

st.sidebar.subheader("🎯 Criterios de Parada")
max_iter = st.sidebar.number_input("Número máximo de iteraciones", min_value=1, value=100)
tolerancia = st.sidebar.number_input("Tolerancia de convergencia", value=1e-5, format="%.1e")

st.sidebar.subheader("📏 Búsqueda de Línea y Wolfe")
alpha_init = st.sidebar.number_input("Alfa Inicial (α0)", min_value=0.01, max_value=10.0, value=1.0, step=0.1)
alpha_min = st.sidebar.number_input("Alfa Mínimo (α min)", min_value=1e-9, max_value=0.1, value=1e-4, format="%.1e")
c1 = st.sidebar.slider("Parámetro c1 (Armijo)", 0.0001, 0.5, 1e-4, format="%.4f")
c2 = st.sidebar.slider("Parámetro c2 (Curvatura)", c1, 0.99, 0.9)

# --- SLIDER EN LA BARRA LATERAL ---
st.sidebar.subheader("👁️ Visualización de Tabla")
rango_tabla = st.sidebar.slider(
    "Rango de iteraciones a mostrar:",
    min_value=0,
    max_value=int(max_iter), 
    value=(0, min(20, int(max_iter)))
)
min_iter_visual = rango_tabla[0]
max_iter_visual = rango_tabla[1]

# --- LÓGICA DE EJECUCIÓN ---
if st.button("Ejecutar Optimización"):
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
            
            def f_eval(x): return float(f_num(*x))
            def grad_eval(x): return np.array(grad_num(*x), dtype=float)
            def hess_eval(x): return np.array(hessian_num(*x), dtype=float)

            x_k = x0.copy()
            historial_datos = []
            trayectoria_x = []
            
            p_old = None
            g_old = None
            
            iters_realizadas = 0
            error_final = 0.0
            
            for k in range(max_iter):
                g_k = grad_eval(x_k)
                f_k = f_eval(x_k)
                error_actual = np.linalg.norm(g_k)
                
                trayectoria_x.append(x_k.copy())
                
                historial_datos.append({
                    "Iteración": k,
                    "x": x_k.copy(),
                    "f(x)": f_k,
                    "Error": error_actual
                })
                
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
                        if denom < 1e-30:
                            beta_k = 0.0
                        else:
                            beta_k = max(0.0, np.dot(g_k, g_k - g_old) / denom)
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

            # Agregar el punto final a la trayectoria
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

        # --- GRÁFICO DE CONVERGENCIA ---
        st.markdown("---")
        st.subheader("📈 Gráfico de Convergencia")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_historial["Iteración"], y=df_historial["Error"], 
                                 mode='lines+markers', name='||∇f(x)||', marker=dict(color='#F63366')))
        fig.update_layout(title="Error vs Número de Iteraciones", xaxis_title="Iteración", 
                          yaxis_title="Error", yaxis_type="log") 
        st.plotly_chart(fig, use_container_width=True)

        # ---------------------------------------------------------------
        # NUEVO: MAPA DE CONTORNO CON TRAYECTORIA (solo para 2 variables)
        # ---------------------------------------------------------------
        if num_vars == 2:
            st.markdown("---")
            st.subheader("🗺️ Mapa de Contorno con Trayectoria")

            tray = np.array(trayectoria_x)
            x1_tray = tray[:, 0]
            x2_tray = tray[:, 1]

            # Calcular rango del contorno con margen alrededor de la trayectoria
            margen = max(abs(x0[0] - x_k[0]), abs(x0[1] - x_k[1])) * 0.4 + 1.5
            x1_min = min(x1_tray.min(), x_k[0]) - margen
            x1_max = max(x1_tray.max(), x_k[0]) + margen
            x2_min = min(x2_tray.min(), x_k[1]) - margen
            x2_max = max(x2_tray.max(), x_k[1]) + margen

            grid_n = 120
            x1_grid = np.linspace(x1_min, x1_max, grid_n)
            x2_grid = np.linspace(x2_min, x2_max, grid_n)
            X1, X2 = np.meshgrid(x1_grid, x2_grid)

            Z = np.zeros_like(X1)
            for i in range(grid_n):
                for j in range(grid_n):
                    try:
                        Z[i, j] = f_eval([X1[i, j], X2[i, j]])
                    except Exception:
                        Z[i, j] = np.nan

            # Limitar valores extremos para que el contorno sea legible
            z_finite = Z[np.isfinite(Z)]
            if len(z_finite) > 0:
                z_p5  = np.percentile(z_finite, 2)
                z_p95 = np.percentile(z_finite, 98)
                Z = np.clip(Z, z_p5, z_p95)

            fig_contour = go.Figure()

            # Capa 1: superficie de contorno rellena
            fig_contour.add_trace(go.Contour(
                x=x1_grid, y=x2_grid, z=Z,
                colorscale='RdYlGn_r',
                contours=dict(showlabels=True, labelfont=dict(size=10, color='white')),
                colorbar=dict(title='f(x)', thickness=14),
                name='f(x1, x2)'
            ))

            # Capa 2: trayectoria del algoritmo
            fig_contour.add_trace(go.Scatter(
                x=x1_tray, y=x2_tray,
                mode='lines+markers',
                line=dict(color='white', width=2, dash='dot'),
                marker=dict(color='white', size=6, symbol='circle',
                            line=dict(color='#333', width=1)),
                name='Trayectoria'
            ))

            # Capa 3: punto de inicio
            fig_contour.add_trace(go.Scatter(
                x=[x0[0]], y=[x0[1]],
                mode='markers+text',
                marker=dict(color='royalblue', size=14, symbol='diamond',
                            line=dict(color='white', width=2)),
                text=['Inicio'],
                textposition='top right',
                textfont=dict(color='royalblue', size=12),
                name='Punto de inicio'
            ))

            # Capa 4: punto final (mínimo encontrado)
            fig_contour.add_trace(go.Scatter(
                x=[x_k[0]], y=[x_k[1]],
                mode='markers+text',
                marker=dict(color='#F63366', size=16, symbol='star',
                            line=dict(color='white', width=2)),
                text=['x*'],
                textposition='top right',
                textfont=dict(color='#F63366', size=13),
                name='Mínimo encontrado'
            ))

            fig_contour.update_layout(
                title=f"Curvas de nivel de f(x) — trayectoria {metodo}",
                xaxis_title='x1',
                yaxis_title='x2',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                height=520
            )
            st.plotly_chart(fig_contour, use_container_width=True)
            st.caption("⬦ Inicio  ★ Mínimo encontrado  · · · Trayectoria del algoritmo")

        else:
            st.info("ℹ️ El mapa de contorno está disponible solo para funciones de 2 variables.")

        # ---------------------------------------------------------------
        # NUEVO: CLASIFICACIÓN AUTOMÁTICA DEL PUNTO CRÍTICO
        # ---------------------------------------------------------------
        st.markdown("---")
        st.subheader("🔬 Clasificación del Punto Crítico")

        H_final = hess_eval(x_k)
        eigenvalues = np.linalg.eigvals(H_final)
        eigenvalues_real = np.real(eigenvalues)
        det_H = np.linalg.det(H_final)
        traza_H = np.trace(H_final)

        todos_positivos  = np.all(eigenvalues_real > 1e-10)
        todos_negativos  = np.all(eigenvalues_real < -1e-10)
        hay_mixtos       = np.any(eigenvalues_real > 1e-10) and np.any(eigenvalues_real < -1e-10)
        hay_cero         = np.any(np.abs(eigenvalues_real) <= 1e-10)

        if todos_positivos:
            tipo      = "✅ Mínimo local"
            color_box = "success"
            explicacion = (
                "Todos los valores propios del Hessiano son **positivos**, "
                "lo que indica que la función es localmente convexa en x*. "
                "El punto encontrado es un **mínimo local** (o global si la función es convexa)."
            )
        elif todos_negativos:
            tipo      = "⚠️ Máximo local"
            color_box = "warning"
            explicacion = (
                "Todos los valores propios del Hessiano son **negativos**, "
                "lo que indica que la función es localmente cóncava en x*. "
                "El punto encontrado es un **máximo local**, no un mínimo."
            )
        elif hay_mixtos:
            tipo      = "❌ Punto de silla"
            color_box = "error"
            explicacion = (
                "Los valores propios del Hessiano tienen **signos mixtos** (positivos y negativos). "
                "El punto encontrado es un **punto de silla**: mínimo en algunas direcciones y máximo en otras. "
                "No es un mínimo real de la función."
            )
        elif hay_cero:
            tipo      = "⚠️ Punto crítico degenerado"
            color_box = "warning"
            explicacion = (
                "Al menos un valor propio del Hessiano es **cercano a cero**. "
                "El test de segunda derivada es **inconcluso**: se necesita análisis de orden superior "
                "para determinar si es mínimo, máximo o punto de silla."
            )
        else:
            tipo      = "❓ No determinado"
            color_box = "info"
            explicacion = "No se pudo clasificar el punto crítico con la información disponible."

        # Mostrar clasificación
        if color_box == "success":
            st.success(f"**Tipo de punto crítico:** {tipo}")
        elif color_box == "warning":
            st.warning(f"**Tipo de punto crítico:** {tipo}")
        elif color_box == "error":
            st.error(f"**Tipo de punto crítico:** {tipo}")
        else:
            st.info(f"**Tipo de punto crítico:** {tipo}")

        st.markdown(explicacion)

        # Tabla de valores propios y métricas del Hessiano
        col_h1, col_h2, col_h3 = st.columns(3)
        col_h1.metric("Determinante det(H)", f"{det_H:.4e}")
        col_h2.metric("Traza tr(H)", f"{traza_H:.4e}")
        col_h3.metric("Nº de variables", num_vars)

        df_eig = pd.DataFrame({
            "Valor propio λ": [f"{v:.6e}" for v in eigenvalues_real],
            "Signo": ["+" if v > 1e-10 else ("−" if v < -1e-10 else "≈0") for v in eigenvalues_real]
        })
        st.dataframe(df_eig, hide_index=True, use_container_width=True)

        # --- MATEMÁTICA SIMBÓLICA ---
        st.markdown("---")
        st.subheader("🧬 Análisis Matemático Avanzado")
        st.write("La aplicación dedujo las siguientes expresiones utilizando diferenciación automática:")
        col_math1, col_math2 = st.columns(2)
        with col_math1:
            st.latex(r"\nabla f(x) = " + sp.latex(grad_sym))
        with col_math2:
            st.latex(r"H(x) = " + sp.latex(hessian_sym))
            
        # --- HISTORIAL FILTRABLE ---
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
