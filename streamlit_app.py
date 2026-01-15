import streamlit as st
import cv2
import numpy as np
import networkx as nx
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label, regionprops
from scipy.interpolate import UnivariateSpline
from PIL import Image
import tempfile
import os

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Coleóptilos",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Parámetros fijos para Hue y Value
FIXED_LOWER_H = 15
FIXED_UPPER_H = 40
FIXED_LOWER_V = 50
FIXED_UPPER_V = 255

# ---------------------------------------------------
# Funciones de procesamiento
# ---------------------------------------------------

def detectar_placa_hough(imagen_bgr):
    """Detecta la placa circular (90 mm) - versión probada y funcional"""
    gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
    gris = cv2.medianBlur(gris, 5)
    circles = cv2.HoughCircles(
        gris,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=gris.shape[0]//4,
        param1=50,
        param2=30,
        minRadius=0,
        maxRadius=0
    )
    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        circles = sorted(circles, key=lambda c: c[2], reverse=True)
        x, y, r = circles[0]
        return x, y, r
    return None

def obtener_trayectoria_central(skel):
    """Obtiene la trayectoria central (longest path) del esqueleto"""
    coords = np.column_stack(np.nonzero(skel))
    
    if len(coords) < 2:
        return coords, float(len(coords))
    
    G = nx.Graph()
    for (r, c) in map(tuple, coords):
        G.add_node((r, c))
    
    # Conectividad 8
    for (r, c) in map(tuple, coords):
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if (nr, nc) in G:
                    dist = np.hypot(dr, dc)
                    G.add_edge((r, c), (nr, nc), weight=dist)
    
    endpoints = [n for n, d in G.degree() if d == 1]
    if len(endpoints) < 2:
        path_coords = coords
        path_length = float(len(coords))
        return path_coords, path_length
    
    max_length = 0
    best_path = []
    for i in range(len(endpoints)):
        for j in range(i+1, len(endpoints)):
            try:
                path = nx.dijkstra_path(G, endpoints[i], endpoints[j])
                length = nx.dijkstra_path_length(G, endpoints[i], endpoints[j])
                if length > max_length:
                    max_length = length
                    best_path = path
            except nx.NetworkXNoPath:
                pass
    
    path_coords = np.array(best_path) if best_path else coords
    return path_coords, max_length

def calcular_curvatura_parametrica(fitted_points):
    """Calcula curvatura y radio de curvatura de la curva suavizada"""
    N = len(fitted_points)
    if N < 3:
        return np.array([9999])
    
    dt = 1.0 / (N - 1)
    y = fitted_points[:, 0]
    x = fitted_points[:, 1]
    
    y_prime = np.zeros(N)
    x_prime = np.zeros(N)
    y_second = np.zeros(N)
    x_second = np.zeros(N)
    
    for i in range(1, N - 1):
        y_prime[i] = (y[i + 1] - y[i - 1]) / (2 * dt)
        x_prime[i] = (x[i + 1] - x[i - 1]) / (2 * dt)
        y_second[i] = (y[i + 1] - 2 * y[i] + y[i - 1]) / (dt ** 2)
        x_second[i] = (x[i + 1] - 2 * x[i] + x[i - 1]) / (dt ** 2)
    
    y_prime[0] = y_prime[1]
    x_prime[0] = x_prime[1]
    y_prime[-1] = y_prime[-2]
    x_prime[-1] = x_prime[-2]
    
    y_second[0] = y_second[1]
    x_second[0] = x_second[1]
    y_second[-1] = y_second[-2]
    x_second[-1] = x_second[-2]
    
    numerador = np.abs(x_prime * y_second - y_prime * x_second)
    denominador = (x_prime ** 2 + y_prime ** 2) ** 1.5
    
    curvatura = np.divide(numerador, denominador, where=denominador > 1e-10, out=np.zeros_like(numerador))
    radio = np.divide(1, curvatura, where=curvatura > 1e-10, out=np.full_like(curvatura, 9999))
    
    return radio

def suavizar_curva_parametrica(path_points, degree=3, n_samples=200):
    """Ajuste polinomial paramétrico y cálculo de longitud"""
    if len(path_points) < degree + 1:
        return path_points, float(len(path_points))
    
    diffs = np.diff(path_points, axis=0)
    seg_lengths = np.hypot(diffs[:,0], diffs[:,1])
    dist_acum = np.concatenate([[0], np.cumsum(seg_lengths)])
    dist_total = dist_acum[-1]
    
    if dist_total <= 0:
        return path_points, 0.0
    
    t_vals = dist_acum / dist_total
    y_coords = path_points[:,0]
    x_coords = path_points[:,1]
    
    poly_y = np.poly1d(np.polyfit(t_vals, y_coords, degree))
    poly_x = np.poly1d(np.polyfit(t_vals, x_coords, degree))
    
    t_samples = np.linspace(0, 1, n_samples)
    y_fit = poly_y(t_samples)
    x_fit = poly_x(t_samples)
    fitted_points = np.column_stack((y_fit, x_fit))
    
    diffs_fit = np.diff(fitted_points, axis=0)
    length_smooth = np.sum(np.hypot(diffs_fit[:,0], diffs_fit[:,1]))
    
    return fitted_points, length_smooth

def contar_segmentos(imagen_bgr, lower_s, upper_s):
    """Cuenta los segmentos detectados con los parámetros dados"""
    mask = segmentar_coleoptilos(imagen_bgr, lower_s, upper_s)
    mask_bool = mask.astype(bool)
    mask_bool = remove_small_objects(mask_bool, min_size=50)
    etiquetas = label(mask_bool)
    num_segmentos = len(np.unique(etiquetas)) - 1  # Restar 1 por el fondo (0)
    return num_segmentos

def ajustar_parametros_saturacion(imagen_bgr, num_esperado, lower_s=50, upper_s=255):
    """Ajusta automáticamente los parámetros de saturación para detectar el número esperado de segmentos"""
    mejor_lower_s = lower_s
    mejor_upper_s = upper_s
    mejor_diferencia = float('inf')
    
    # Probar diferentes combinaciones de saturación
    for l_s in range(0, 256, 20):
        for u_s in range(max(l_s + 50, 50), 256, 20):
            num_actual = contar_segmentos(imagen_bgr, l_s, u_s)
            diferencia = abs(num_actual - num_esperado)
            
            if diferencia < mejor_diferencia:
                mejor_diferencia = diferencia
                mejor_lower_s = l_s
                mejor_upper_s = u_s
            
            # Si encontramos coincidencia exacta, terminamos
            if num_actual == num_esperado:
                return l_s, u_s, num_actual
    
    # Si no encontramos exactamente, retornar los mejores
    num_final = contar_segmentos(imagen_bgr, mejor_lower_s, mejor_upper_s)
    return mejor_lower_s, mejor_upper_s, num_final

def segmentar_coleoptilos(imagen_bgr, lower_s=50, upper_s=255):
    """Segmenta los coleóptilos usando el espacio de color HSV"""
    imagen_hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([FIXED_LOWER_H, lower_s, FIXED_LOWER_V])
    upper = np.array([FIXED_UPPER_H, upper_s, FIXED_UPPER_V])
    mask = cv2.inRange(imagen_hsv, lower, upper)
    
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return mask

def medir_coleoptilos(imagen_bgr, lower_s=50, upper_s=255, placa_manual=None):
    """Detecta y mide los coleóptilos sobre la imagen original"""
    if imagen_bgr is None:
        return None, None
    
    # Detectar la placa para calcular el factor de conversión (90 mm reales)
    imagen_procesada = imagen_bgr.copy()
    altura, ancho = imagen_bgr.shape[:2]
    centro_x_img = ancho // 2
    centro_y_img = altura // 2
    
    # Usar placa manual si está disponible, si no, detectar automáticamente
    if placa_manual is not None:
        offset_x, offset_y, r = placa_manual
        # Convertir offsets a coordenadas reales
        x = centro_x_img + offset_x
        y = centro_y_img + offset_y
        placa = (x, y, r)
    else:
        placa = detectar_placa_hough(imagen_bgr)
    
    factor = None
    if placa is not None:
        x, y, r = placa
        x, y, r = int(x), int(y), int(r)
        diametro_px = 2 * r
        factor = 90 / diametro_px
        cv2.circle(imagen_procesada, (x, y), r, (0, 0, 255), 2)
        # Asegurar que las coordenadas del texto estén dentro de los límites
        texto_x = max(10, x - r)
        texto_y = max(20, y - r - 10)
        cv2.putText(imagen_procesada, "Placa (90mm)", (texto_x, texto_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Segmentar coleóptilos
    mask = segmentar_coleoptilos(imagen_bgr, lower_s, upper_s)
    mask_bool = mask.astype(bool)
    mask_bool = remove_small_objects(mask_bool, min_size=50)
    etiquetas = label(mask_bool)
    
    mediciones = []
    
    for region in regionprops(etiquetas):
        if region.area < 50:
            continue
        
        mask_objeto = (etiquetas == region.label)
        skel = skeletonize(mask_objeto)
        
        # Obtener trayectoria central
        path_points, raw_length = obtener_trayectoria_central(skel)
        
        # Suavizar la curva paramétricamente
        fitted_points, length_smooth = suavizar_curva_parametrica(path_points, degree=3, n_samples=200)
        
        # Calcular radio de curvatura a partir de la curva suavizada
        radios = calcular_curvatura_parametrica(fitted_points)
        radio_min_px = np.min(radios)
        
        longitud_mm = length_smooth * factor if factor is not None else None
        radio_min_mm = radio_min_px * factor if factor is not None else None
        
        # Dibujar la curva suavizada en azul
        pts = np.int32(fitted_points)
        for i in range(len(pts) - 1):
            p1 = (pts[i, 1], pts[i, 0])
            p2 = (pts[i + 1, 1], pts[i + 1, 0])
            cv2.line(imagen_procesada, p1, p2, (255, 0, 0), 2)
        
        # Dibujar etiqueta en color negro en el centroide
        cy, cx = region.centroid
        cv2.putText(imagen_procesada, str(region.label), (int(cx), int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        
        mediciones.append({
            "ID": int(region.label),
            "Longitud (px)": round(length_smooth, 2),
            "Longitud (mm)": round(longitud_mm, 2) if longitud_mm is not None else "N/A",
            "Radio min (px)": round(radio_min_px, 2),
            "Radio min (mm)": round(radio_min_mm, 2) if radio_min_mm is not None else "N/A"
        })
    
    return imagen_procesada, mediciones

# ---------------------------------------------------
# Interfaz de usuario con Streamlit
# ---------------------------------------------------

st.title("🌱 Análisis de Coleóptilos")
st.markdown("---")

# Sidebar para opciones
with st.sidebar:
    st.header("Opciones de entrada")
    opcion_entrada = st.radio(
        "¿Cómo deseas capturar la imagen?",
        ["📸 Captura desde cámara", "📁 Cargar archivo"]
    )
    
    st.markdown("---")
    st.header("Parámetros de segmentación")
    
    # Campo para número esperado de segmentos
    num_segmentos_esperados = st.number_input(
        "Número de segmentos esperados",
        min_value=1,
        max_value=50,
        value=5,
        help="Ingresa el número de coleóptilos que esperas detectar"
    )
    st.session_state.num_segmentos_esperados = num_segmentos_esperados
    
    col1, col2 = st.columns(2)
    with col1:
        lower_s = st.slider("Saturación mínima", 0, 255, 50)
    with col2:
        upper_s = st.slider("Saturación máxima", 0, 255, 255)
    
    # Botón para auto-ajustar parámetros
    auto_adjust = st.checkbox("Auto-ajustar parámetros de saturación", value=False)
    
    st.markdown("---")
    st.header("Calibración de placa")
    
    st.write("Ajusta el círculo de la placa (offsets desde el centro de la imagen):")
    
    # Obtener valores por defecto del slider basados en detección
    placa_detectada = st.session_state.get('placa_detectada', None)
    
    # Asumir dimensiones estándar para calcular offsets
    ancho_ref = 1280
    altura_ref = 960
    centro_x_ref = ancho_ref // 2
    centro_y_ref = altura_ref // 2
    
    if placa_detectada:
        x_detectado, y_detectado, r_default = placa_detectada
        offset_x_default = x_detectado - centro_x_ref
        offset_y_default = y_detectado - centro_y_ref
    else:
        offset_x_default = 0
        offset_y_default = 0
        r_default = 100
    
    # Sliders para offsets (desplazamiento desde el centro)
    offset_x = st.slider("Desplazamiento X (píxeles)", -600, 600, offset_x_default, key="sidebar_x")
    offset_y = st.slider("Desplazamiento Y (píxeles)", -450, 450, offset_y_default, key="sidebar_y")
    radio_slider = st.slider("Radio (píxeles)", 10, 500, r_default, key="sidebar_r")
    
    # Guardar posición real del círculo (centro_imagen + offset)
    st.session_state.placa_manual = (offset_x, offset_y, radio_slider)

# Procesamiento según la opción seleccionada
imagen_para_procesar = None

if opcion_entrada == "📸 Captura desde cámara":
    st.subheader("Captura desde cámara")
    st.info("Se requiere permiso de acceso a la cámara del dispositivo")
    
    # Usar streamlit-webrtc si está disponible, si no, usar input_camera
    try:
        from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
        
        rtc_configuration = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        
        webrtc_ctx = webrtc_streamer(
            key="webcam",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
            sendback_audio=False,
            video_frame_callback=None,
            media_stream_constraints_string='{"video": true}',
        )
        
        if webrtc_ctx.state.playing:
            st.write("Cámara en vivo - Captura habilitada")
    except ImportError:
        st.warning("streamlit-webrtc no está instalado. Usando captura simple.")
        camera_image = st.camera_input("Captura una foto")
        if camera_image is not None:
            imagen_pil = Image.open(camera_image)
            imagen_para_procesar = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)
            # Detectar placa automáticamente
            placa_detectada = detectar_placa_hough(imagen_para_procesar)
            if placa_detectada:
                st.session_state.placa_detectada = placa_detectada

elif opcion_entrada == "📁 Cargar archivo":
    st.subheader("Cargar imagen desde archivo")
    archivo_cargado = st.file_uploader(
        "Selecciona una imagen",
        type=["jpg", "jpeg", "png", "bmp"]
    )
    
    if archivo_cargado is not None:
        imagen_pil = Image.open(archivo_cargado)
        imagen_para_procesar = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)
        st.success("Imagen cargada correctamente")
        # Detectar placa automáticamente
        placa_detectada = detectar_placa_hough(imagen_para_procesar)
        if placa_detectada:
            st.session_state.placa_detectada = placa_detectada

# Procesamiento de la imagen
if imagen_para_procesar is not None:
    st.markdown("---")
    st.subheader("Procesamiento")
    
    # Mostrar imagen original
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Imagen original**")
        imagen_rgb = cv2.cvtColor(imagen_para_procesar, cv2.COLOR_BGR2RGB)
        st.image(imagen_rgb)
    
    with col2:
        st.write("**Imagen con detecciones (Backbone central suavizado)**")
        
        # Procesar imagen con actualizaciones en tiempo real
        # Asegurar que num_segmentos_esperados esté definido
        num_segmentos_esperados = st.session_state.get('num_segmentos_esperados', 5)
        
        # Auto-ajustar parámetros si está habilitado
        params_lower_s = lower_s
        params_upper_s = upper_s
        
        if auto_adjust and num_segmentos_esperados > 0:
            params_lower_s, params_upper_s, num_detectado = ajustar_parametros_saturacion(
                imagen_para_procesar, 
                num_segmentos_esperados,
                lower_s,
                upper_s
            )
        else:
            # Contar segmentos con parámetros actuales
            num_detectado = contar_segmentos(imagen_para_procesar, params_lower_s, params_upper_s)
        
        placa_usar = st.session_state.get('placa_manual', None)
        imagen_procesada, mediciones = medir_coleoptilos(
            imagen_para_procesar,
            lower_s=params_lower_s,
            upper_s=params_upper_s,
            placa_manual=placa_usar
        )
        
        if imagen_procesada is not None:
            imagen_procesada_rgb = cv2.cvtColor(imagen_procesada, cv2.COLOR_BGR2RGB)
            st.image(imagen_procesada_rgb)
            st.caption("Línea azul: eje central suavizado | Punto rojo: placa de calibración (90mm)")
        else:
            st.warning("No se pudo procesar la imagen")
        
        # Mostrar información de segmentación
        if auto_adjust and num_segmentos_esperados > 0:
            st.success(f"✅ Parámetros ajustados: {num_detectado} segmentos detectados (esperados: {num_segmentos_esperados})")
        else:
            st.info(f"ℹ️ Segmentos detectados: {num_detectado} (esperados: {num_segmentos_esperados})")
    
    # Mostrar resultados
    st.markdown("---")
    st.subheader("📊 Resultados de mediciones")
    
    if mediciones is not None and len(mediciones) > 0:
        # Convertir a tabla para visualización
        df_mediciones = []
        for med in mediciones:
            df_mediciones.append({
                "ID": med['ID'],
                "Longitud (px)": med['Longitud (px)'],
                "Longitud (mm)": med['Longitud (mm)'],
                "Radio mín (px)": med['Radio min (px)'],
                "Radio mín (mm)": med['Radio min (mm)']
            })
        
        # Mostrar tabla interactiva
        st.write("#### Tabla de mediciones:")
        
        # Crear columnas para mostrar la tabla como HTML
        tabla_html = "<table style='border-collapse: collapse; width: 100%;'>"
        tabla_html += "<tr style='background-color: #2a2a2a; color: #ffffff;'>"
        tabla_html += "<th style='border: 1px solid #555; padding: 8px;'>ID</th>"
        tabla_html += "<th style='border: 1px solid #555; padding: 8px;'>Longitud (px)</th>"
        tabla_html += "<th style='border: 1px solid #555; padding: 8px;'>Longitud (mm)</th>"
        tabla_html += "<th style='border: 1px solid #555; padding: 8px;'>Radio mín (px)</th>"
        tabla_html += "<th style='border: 1px solid #555; padding: 8px;'>Radio mín (mm)</th>"
        tabla_html += "</tr>"
        
        for med in mediciones:
            tabla_html += "<tr>"
            tabla_html += f"<td style='border: 1px solid #555; padding: 8px;'>{med['ID']}</td>"
            tabla_html += f"<td style='border: 1px solid #555; padding: 8px;'>{med['Longitud (px)']}</td>"
            tabla_html += f"<td style='border: 1px solid #555; padding: 8px;'>{med['Longitud (mm)']}</td>"
            tabla_html += f"<td style='border: 1px solid #555; padding: 8px;'>{med['Radio min (px)']}</td>"
            tabla_html += f"<td style='border: 1px solid #555; padding: 8px;'>{med['Radio min (mm)']}</td>"
            tabla_html += "</tr>"
        
        tabla_html += "</table>"
        st.markdown(tabla_html, unsafe_allow_html=True)
        
        # Estadísticas
        st.write("#### Estadísticas:")
        longitudes_px = [m['Longitud (px)'] for m in mediciones]
        radios_px = [m['Radio min (px)'] for m in mediciones]
        longitudes_mm = [m['Longitud (mm)'] for m in mediciones if isinstance(m['Longitud (mm)'], float)]
        radios_mm = [m['Radio min (mm)'] for m in mediciones if isinstance(m['Radio min (mm)'], float)]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total", len(mediciones))
        with col2:
            st.metric("Longitud promedio (px)", f"{np.mean(longitudes_px):.2f}")
        with col3:
            if longitudes_mm:
                st.metric("Longitud promedio (mm)", f"{np.mean(longitudes_mm):.2f}")
        with col4:
            if radios_mm:
                st.metric("Radio mín promedio (mm)", f"{np.mean(radios_mm):.2f}")
        
        # Descargar resultados en formato CSV
        csv_content = "ID,Longitud (px),Longitud (mm),Radio min (px),Radio min (mm)\n"
        for med in mediciones:
            csv_content += f"{med['ID']},{med['Longitud (px)']},{med['Longitud (mm)']},{med['Radio min (px)']},{med['Radio min (mm)']}\n"
        
        st.download_button(
            label="📥 Descargar resultados (CSV)",
            data=csv_content,
            file_name="coleoptilo_mediciones.csv",
            mime="text/csv"
        )
    else:
        st.warning("No se detectaron coleóptilos en la imagen. Intenta ajustar los parámetros de saturación.")

else:
    st.info("👈 Selecciona una opción en el panel izquierdo para comenzar")# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
    Aplicación de análisis de coleóptilos | 2026
    </div>
    """,
    unsafe_allow_html=True
)
