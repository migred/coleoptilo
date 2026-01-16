import cv2
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label, regionprops
import streamlit as st
from PIL import Image
import io

# Función para segmentar los coleóptilos usando el espacio de color HSV
def segmentar_coleoptilos(imagen_bgr, lower_hue=15, upper_hue=40, lower_sat=50, upper_sat=255, lower_val=50, upper_val=255):
    # Convertir la imagen al espacio HSV
    imagen_hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)
    # Definir rangos ajustables para la segmentación
    lower = np.array([lower_hue, lower_sat, lower_val])
    upper = np.array([upper_hue, upper_sat, upper_val])
    mask = cv2.inRange(imagen_hsv, lower, upper)
    # Aplicar operaciones morfológicas para limpiar la máscara
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask

# Función para detectar y medir los coleóptilos sobre la imagen original
def medir_coleoptilos(imagen_bgr, lower_hue=15, upper_hue=40, lower_sat=50, upper_sat=255, lower_val=50, upper_val=255, min_area=50):
    if imagen_bgr is None:
        return None, None

    # Usar segmentación en HSV para obtener la máscara de los coleóptilos
    mask = segmentar_coleoptilos(imagen_bgr, lower_hue, upper_hue, lower_sat, upper_sat, lower_val, upper_val)
    mask_bool = mask.astype(bool)
    mask_bool = remove_small_objects(mask_bool, max_size=min_area-1)
    etiquetas = label(mask_bool)
    
    mediciones = []
    # Copia de la imagen original para superponer las detecciones
    imagen_con_detecciones = imagen_bgr.copy()
    
    for region in regionprops(etiquetas):
        if region.area < min_area:
            continue
        # Calcular el esqueleto para medir la longitud (en píxeles)
        mask_objeto = (etiquetas == region.label)
        skeleton = skeletonize(mask_objeto)
        longitud_px = np.count_nonzero(skeleton)
        
        mediciones.append({
            "ID": region.label,
            "Área (px)": region.area,
            "Longitud (px)": longitud_px
        })
        # Obtener el bounding box (y1, x1, y2, x2)
        y1, x1, y2, x2 = region.bbox
        cv2.rectangle(imagen_con_detecciones, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(imagen_con_detecciones, str(region.label), (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    df = pd.DataFrame(mediciones)
    return imagen_con_detecciones, df

# Configuración de la página
st.set_page_config(
    page_title="Detección de Coleóptilos",
    page_icon="🌱",
    layout="wide"
)

# Título principal
st.title("🌱 Detección de Coleóptilos (Segmentación HSV)")
st.markdown("---")

# Crear dos columnas para mejor organización
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📷 Captura de Imagen")
    
    # Opción de seleccionar método de captura
    opcion = st.radio(
        "Selecciona el método de entrada:",
        ["📱 Cámara (Móvil/Web)", "📁 Subir archivo"],
        horizontal=True
    )
    
    imagen_capturada = None
    
    if opcion == "📱 Cámara (Móvil/Web)":
        st.info("💡 En dispositivos móviles, esto te permitirá usar la cámara del teléfono.")
        # camera_input permite capturar desde la cámara del dispositivo
        camera_image = st.camera_input("Toma una foto de los coleóptilos")
        
        if camera_image is not None:
            imagen_capturada = camera_image
            
    else:  # Subir archivo
        uploaded_file = st.file_uploader(
            "Selecciona una imagen",
            type=["jpg", "jpeg", "png"],
            help="Sube una imagen en formato JPG, JPEG o PNG"
        )
        
        if uploaded_file is not None:
            imagen_capturada = uploaded_file

with col2:
    st.subheader("⚙️ Parámetros de Detección")
    
    # Selector de modo de entrada
    modo_entrada = st.selectbox(
        "Modo de entrada de parámetros:",
        ["Sliders", "Campos de texto"],
        help="Elige cómo quieres introducir los valores"
    )
    
    if modo_entrada == "Sliders":
        st.write("**Rango de Matiz (Hue)**")
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            lower_hue = st.slider("Mín Matiz", 0, 179, 15)
        with col_h2:
            upper_hue = st.slider("Máx Matiz", 0, 179, 40)
        
        st.write("**Rango de Saturación**")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            lower_sat = st.slider("Mín Saturación", 0, 255, 50)
        with col_s2:
            upper_sat = st.slider("Máx Saturación", 0, 255, 255)
        
        st.write("**Rango de Valor (Brillo)**")
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            lower_val = st.slider("Mín Valor", 0, 255, 50)
        with col_v2:
            upper_val = st.slider("Máx Valor", 0, 255, 255)
        
        min_area = st.slider("Área mínima (px)", 10, 500, 50, help="Objetos menores a este valor serán ignorados")
    
    else:  # Campos de texto
        st.write("**Rango de Matiz (Hue) [0-179]**")
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            lower_hue = st.number_input("Mín Matiz", 0, 179, 15, key="text_lower_hue")
        with col_h2:
            upper_hue = st.number_input("Máx Matiz", 0, 179, 40, key="text_upper_hue")
        
        st.write("**Rango de Saturación [0-255]**")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            lower_sat = st.number_input("Mín Saturación", 0, 255, 50, key="text_lower_sat")
        with col_s2:
            upper_sat = st.number_input("Máx Saturación", 0, 255, 255, key="text_upper_sat")
        
        st.write("**Rango de Valor (Brillo) [0-255]**")
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            lower_val = st.number_input("Mín Valor", 0, 255, 50, key="text_lower_val")
        with col_v2:
            upper_val = st.number_input("Máx Valor", 0, 255, 255, key="text_upper_val")
        
        min_area = st.number_input("Área mínima (px)", 10, 500, 50, help="Objetos menores a este valor serán ignorados", key="text_min_area")
    
    # Validación de rangos
    if lower_hue >= upper_hue:
        st.error("⚠️ El valor mínimo de matiz debe ser menor que el máximo")
    if lower_sat >= upper_sat:
        st.error("⚠️ El valor mínimo de saturación debe ser menor que el máximo")
    if lower_val >= upper_val:
        st.error("⚠️ El valor mínimo de valor debe ser menor que el máximo")
    
    # Botón para restablecer valores por defecto
    if st.button("🔄 Restablecer valores por defecto"):
        st.rerun()

# Procesar la imagen si existe
if imagen_capturada is not None:
    # Convertir la imagen a formato OpenCV (BGR)
    image_pil = Image.open(imagen_capturada)
    image_np = np.array(image_pil)
    
    # Convertir RGB a BGR para OpenCV
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        imagen_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    else:
        imagen_bgr = image_np
    
    # Procesar la imagen
    with st.spinner('🔍 Analizando coleóptilos...'):
        imagen_procesada, df_mediciones = medir_coleoptilos(
            imagen_bgr, lower_hue, upper_hue, lower_sat, upper_sat, lower_val, upper_val, min_area
        )
    
    # Mostrar resultados
    st.markdown("---")
    col_img1, col_img2 = st.columns(2)
    
    with col_img1:
        st.subheader("🖼️ Imagen Original")
        st.image(image_pil, use_container_width=True)
    
    with col_img2:
        st.subheader("✅ Imagen con Detecciones")
        if imagen_procesada is not None:
            # Convertir BGR a RGB para mostrar correctamente
            imagen_procesada_rgb = cv2.cvtColor(imagen_procesada, cv2.COLOR_BGR2RGB)
            st.image(imagen_procesada_rgb, use_container_width=True)
    
    # Mostrar tabla de mediciones debajo
    st.markdown("---")
    st.subheader("📊 Mediciones de Coleóptilos")
    
    if df_mediciones is not None and not df_mediciones.empty:
        # Mostrar estadísticas resumen
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        
        with col_stat1:
            st.metric("Total de Coleóptilos", len(df_mediciones))
        
        with col_stat2:
            st.metric("Longitud Promedio (px)", f"{df_mediciones['Longitud (px)'].mean():.1f}")
        
        with col_stat3:
            st.metric("Área Promedio (px)", f"{df_mediciones['Área (px)'].mean():.1f}")
        
        # Mostrar tabla de datos
        st.dataframe(
            df_mediciones,
            use_container_width=True,
            hide_index=True
        )
        
        # Opción para descargar los datos
        csv = df_mediciones.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar datos en CSV",
            data=csv,
            file_name='mediciones_coleoptilos.csv',
            mime='text/csv',
        )
    else:
        st.warning("⚠️ No se detectaron coleóptilos en la imagen. Intenta ajustar la iluminación o la configuración.")

else:
    # Mensaje de bienvenida cuando no hay imagen
    st.info("👆 Selecciona un método de captura arriba para comenzar el análisis.")
    
    with st.expander("ℹ️ Instrucciones de uso"):
        st.markdown("""
        ### Cómo usar esta aplicación:
        
        1. **Método de captura**:
           - **📱 Cámara**: Usa la cámara de tu móvil o webcam para tomar una foto en tiempo real
           - **📁 Subir archivo**: Sube una imagen existente desde tu dispositivo
        
        2. **Captura la imagen**: Asegúrate de que los coleóptilos sean visibles y estén bien iluminados
        
        3. **Análisis automático**: La aplicación detectará y medirá automáticamente los coleóptilos
        
        4. **Resultados**: Revisa las detecciones en la imagen y las mediciones en la tabla
        
        5. **Descarga**: Puedes descargar los datos en formato CSV para análisis posterior
        
        ### Consejos para mejores resultados:
        - 🔆 Usa buena iluminación uniforme
        - 📏 Coloca los coleóptilos sobre un fondo contrastante
        - 📸 Asegúrate de que la imagen esté enfocada y sin desenfoque
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Aplicación de análisis de coleóptilos | Desarrollado con Streamlit</div>",
    unsafe_allow_html=True
)
