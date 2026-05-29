import streamlit as st
import numpy as np
import plotly.graph_objects as go
# import sympy as sp # Para el valor agregado de derivadas automáticas

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Optimizador de Funciones", layout="wide")
st.title("Calculadora de Optimización No Lineal")

# --- BARRA LATERAL: DATOS DE ENTRADA ---
st.sidebar.header("Parámetros de Entrada")

num_vars = st.sidebar.number_input("Número de variables", min_value=1, max_value=10, value=2)
metodo = st.sidebar.selectbox("Método de Optimización", 
                              ["Gradiente Descendente", "Gradiente Conjugado", "Método de Newton"])
funcion_str = st.sidebar.text_input("Función objetivo f(x)", value="x1**2 + x2**2")
punto_inicio = st.sidebar.text_input("Punto de partida (separado por comas)", value="5.0, 5.0")

st.sidebar.subheader("Criterios de Parada")
max_iter = st.sidebar.number_input("Número máximo de iteraciones", min_value=1, value=100)
tolerancia = st.sidebar.number_input("Tolerancia de convergencia", value=1e-5, format="%.1e")

st.sidebar.subheader("Parámetros de Wolfe")
c1 = st.sidebar.slider("Parámetro c1 (Armijo)", 0.0001, 0.5, 1e-4, format="%.4f")
c2 = st.sidebar.slider("Parámetro c2 (Curvatura)", c1, 0.99, 0.9)

# --- LÓGICA DE EJECUCIÓN ---
if st.button("Ejecutar Optimización"):
    try:
        # 1. Parsear los inputs
        x0 = np.array([float(x.strip()) for x in punto_inicio.split(',')])
        if len(x0) != num_vars:
            st.error(f"El punto de partida debe tener {num_vars} coordenadas.")
            st.stop()
            
        # 2. Aquí iría tu lógica matemática (sympy para la función, y tu bucle de optimización)
        # Por ahora usaremos datos simulados para mostrar cómo se verían los resultados esperados
        
        # DATOS SIMULADOS (Reemplazar con los algoritmos reales)
        x_min_encontrado = np.zeros(num_vars)
        f_min = 0.0
        iters_realizadas = 25
        error_final = 1e-6
        historial_errores = [10.0 * (0.8 ** i) for i in range(iters_realizadas)] # Curva simulada
        
        # --- RESULTADOS ESPERADOS ---
        st.success("Optimización finalizada con éxito.")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Punto mínimo (x*)", f"[{', '.join([f'{x:.4f}' for x in x_min_encontrado])}]")
        col2.metric("Valor en el mínimo f(x*)", f"{f_min:.4e}")
        col3.metric("Iteraciones", iters_realizadas)
        col4.metric("Error Final", f"{error_final:.2e}")
        
        # --- GRÁFICO DE CONVERGENCIA ---
        st.subheader("Gráfico de Convergencia")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(iters_realizadas)), y=historial_errores, mode='lines+markers', name='Error'))
        fig.update_layout(title="Error vs Número de Iteraciones", xaxis_title="Iteración", yaxis_title="Error ||∇f(x)||")
        st.plotly_chart(fig, use_container_width=True)
        
        # --- VALOR AGREGADO ---
        st.subheader("Valor Agregado")
        st.info("Aquí puedes mostrar la matriz Hessiana calculada automáticamente, o el gráfico 2D de la trayectoria de los puntos.")

    except Exception as e:
        st.error(f"Ocurrió un error al procesar los datos: {e}")