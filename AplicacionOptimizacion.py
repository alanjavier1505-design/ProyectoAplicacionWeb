import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sympy as sp
from scipy.optimize import line_search
import pandas as pd

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Optimizador de Funciones", layout="wide")
st.title("Calculadora de Optimización No Lineal")

# --- BARRA LATERAL: DATOS DE ENTRADA ---
st.sidebar.header("Parámetros de Entrada")

num_vars = st.sidebar.number_input("Número de variables", min_value=1, max_value=10, value=2)
metodo = st.sidebar.selectbox("Método de Optimización", 
                              ["Gradiente Descendente", "Gradiente Conjugado", "Método de Newton"])

# Pequeña ayuda visual en el input
st.sidebar.markdown("<small>Ej: x1**2 + x2**2, sin(x1) + cos(x2)</small>", unsafe_allow_html=True)
funcion_str = st.sidebar.text_input("Función objetivo f(x)", value="x1**2 + x2**2")
punto_inicio = st.sidebar.text_input("Punto de partida (separado por comas)", value="5.0, 5.0")

st.sidebar.subheader("Criterios de Parada")
max_iter = st.sidebar.number_input("Número máximo de iteraciones", min_value=1, value=100)
tolerancia = st.sidebar.number_input("Tolerancia de convergencia", value=1e-5, format="%.1e")

st.sidebar.subheader("Parámetros de Wolfe")
c1 = st.sidebar.slider("Parámetro c1 (Armijo)", 0.0001, 0.5, 1e-4, format="%.4f")
c2 = st.sidebar.slider("Parámetro c2 (Curvatura)", c1, 0.99, 0.9)

# --- NUEVO: SLIDER EN LA BARRA LATERAL ---
st.sidebar.subheader("Visualización de Tabla")
rango_tabla = st.sidebar.slider(
    "Rango de iteraciones a mostrar:",
    min_value=0,
    max_value=int(max_iter), # El máximo ahora es el límite de iteraciones
    value=(0, min(20, int(max_iter)))
)
min_iter_visual = rango_tabla[0]
max_iter_visual = rango_tabla[1]

# --- LÓGICA DE EJECUCIÓN ---
if st.button("Ejecutar Optimización"):
    try:
        # 1. Parsear los inputs numéricos
        x0 = np.array([float(x.strip()) for x in punto_inicio.split(',')])
        if len(x0) != num_vars:
            st.error(f"El punto de partida debe tener {num_vars} coordenadas.")
            st.stop()
            
        with st.spinner("Calculando derivadas y ejecutando optimización..."):
            
            # 2. MOTOR MATEMÁTICO (SymPy)
            # Definir variables simbólicas (x1, x2, ..., xn)
            vars_sym = sp.symbols(f'x1:{num_vars+1}')
            
            # Limpiar la función ingresada para evitar errores comunes
            func_str_clean = funcion_str.replace('sen', 'sin').replace('^', '**')
            f_sym = sp.sympify(func_str_clean)
            
            # Calcular Gradiente y Hessiano simbólicamente
            grad_sym = [sp.diff(f_sym, var) for var in vars_sym]
            hessian_sym = [[sp.diff(g, var) for var in vars_sym] for g in grad_sym]
            
            # Convertir a funciones numéricas de NumPy (Mucho más rápidas)
            f_num = sp.lambdify(vars_sym, f_sym, 'numpy')
            grad_num = sp.lambdify(vars_sym, grad_sym, 'numpy')
            hessian_num = sp.lambdify(vars_sym, hessian_sym, 'numpy')
            
            # Envoltorios para manejar los arrays correctamente
            def f_eval(x): return float(f_num(*x))
            def grad_eval(x): return np.array(grad_num(*x), dtype=float)
            def hess_eval(x): return np.array(hessian_num(*x), dtype=float)

            # 3. BUCLE DE OPTIMIZACIÓN
            x_k = x0.copy()
            historial_datos = []
            
            p_old = None
            g_old = None
            
            iters_realizadas = 0
            error_final = 0.0
            
            for k in range(max_iter):
                g_k = grad_eval(x_k)
                f_k = f_eval(x_k)
                error_actual = np.linalg.norm(g_k)
                
                # Guardar datos para la tabla y gráfico
                historial_datos.append({
                    "Iteración": k,
                    "x": x_k.copy(),
                    "f(x)": f_k,
                    "Error": error_actual
                })
                
                if error_actual < tolerancia:
                    error_final = error_actual
                    break
                    
                # Calcular Dirección de Búsqueda (P_k) según el método
                if metodo == "Gradiente Descendente":
                    p_k = -g_k
                    
                elif metodo == "Método de Newton":
                    H_k = hess_eval(x_k)
                    try:
                        # Resolver sistema lineal H*p = -g (Más estable que calcular la inversa directa)
                        p_k = np.linalg.solve(H_k, -g_k)
                    except np.linalg.LinAlgError:
                        # Si la matriz es singular, usamos pseudoinversa o bajamos a gradiente
                        p_k = np.dot(np.linalg.pinv(H_k), -g_k)
                        
                elif metodo == "Gradiente Conjugado":
                    if k == 0:
                        p_k = -g_k
                    else:
                        # Fórmula de Fletcher-Reeves para Beta
                        beta_k = np.dot(g_k, g_k) / np.dot(g_old, g_old)
                        p_k = -g_k + beta_k * p_old
                
                # Búsqueda de línea con Condiciones de Wolfe (SciPy)
                res_alpha = line_search(f_eval, grad_eval, x_k, p_k, gfk=g_k, old_fval=f_k, c1=c1, c2=c2)
                alpha_k = res_alpha[0]
                
                # Si Wolfe falla en encontrar un paso, usamos uno pequeño para no romper el bucle
                if alpha_k is None:
                    alpha_k = 1e-4
                    
                # Actualizar variables para la siguiente iteración
                p_old = p_k
                g_old = g_k
                x_k = x_k + alpha_k * p_k
                iters_realizadas += 1
                error_final = error_actual

        # --- RESULTADOS ESPERADOS ---
        st.success("Optimización finalizada con éxito.")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Punto mínimo (x*)", f"[{', '.join([f'{x:.4f}' for x in x_k])}]")
        col2.metric("Valor en el mínimo f(x*)", f"{f_eval(x_k):.4e}")
        col3.metric("Iteraciones", iters_realizadas)
        col4.metric("Error Final", f"{error_final:.2e}")
        
        # Preparar datos para gráficos y tablas
        df_historial = pd.DataFrame(historial_datos)
        
        # --- GRÁFICO DE CONVERGENCIA ---
        st.subheader("Gráfico de Convergencia")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_historial["Iteración"], y=df_historial["Error"], 
                                 mode='lines+markers', name='||∇f(x)||', marker=dict(color='#F63366')))
        fig.update_layout(title="Error vs Número de Iteraciones", xaxis_title="Iteración", 
                          yaxis_title="Error", yaxis_type="log") # Escala logarítmica es mejor para el error
        st.plotly_chart(fig, use_container_width=True)
        
        # --- VALOR AGREGADO 1: MATEMÁTICA SIMBÓLICA ---
        st.markdown("---")
        st.subheader("Análisis Matemático")
        st.write("La aplicación dedujo las siguientes expresiones utilizando diferenciación automática:")
        col_math1, col_math2 = st.columns(2)
        with col_math1:
            st.latex(r"\nabla f(x) = " + sp.latex(grad_sym))
        with col_math2:
            st.latex(r"H(x) = " + sp.latex(hessian_sym))
            
        # --- VALOR AGREGADO 2: HISTORIAL FILTRABLE ---
        st.markdown("---")
        st.subheader("Historial de Iteraciones")
        
        # Desglosar el array 'x' en columnas separadas x1, x2... para la tabla
        for i in range(num_vars):
            df_historial[f"x{i+1}"] = df_historial["x"].apply(lambda coord: coord[i])
        df_historial = df_historial.drop(columns=["x"]) # Quitamos la columna original del array
        
        # Aplicamos el filtro usando las variables obtenidas del slider en la barra lateral
        tabla_filtrada = df_historial[(df_historial["Iteración"] >= min_iter_visual) & (df_historial["Iteración"] <= max_iter_visual)]
        
        # Mostrar la tabla
        st.dataframe(tabla_filtrada, hide_index=True, use_container_width=True)

    except Exception as e:
        st.error(f"Ocurrió un error matemático o de sintaxis: {e}")
        st.info("Asegúrate de que la función esté bien escrita (ej. usa * para multiplicar: 2*x1).")
