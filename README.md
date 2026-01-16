# 🌱 Análisis de Coleóptilos

Aplicación web interactiva para capturar, segmentar y analizar coleóptilos (tallos de plántulas de gramíneas) mediante visión por computadora.

## ✨ Características principales

- **📸 Múltiples opciones de entrada**:
  - Captura en tiempo real desde cámara web
  - Carga de imágenes desde archiv

- **🎯 Calibración automática y manual**:
  - Detección automática de la placa de referencia (90mm)
  - Ajuste manual mediante sliders (offsets desde el centro)
  - Visualización en tiempo real del círculo ajustado

- **🔬 Análisis avanzado**:
  - Segmentación HSV de coleóptilos
  - Extracción del esqueleto (backbone central suavizado)
  - Cálculo de longitud en píxeles y milímetros
  - Medición de curvatura y radio de curvatura

- **⚙️ Parámetros ajustables**:
  - Control de saturación (mínima y máxima)
  - Auto-ajuste automático de parámetros
  - Número de segmentos esperados configurable

- **📊 Resultados**:
  - Tabla interactiva con mediciones
  - Estadísticas agregadas (promedios, totales)
  - Exportación de resultados a CSV
  - Visualización de detecciones en imagen procesada

## 🚀 Inicio rápido

### Requisitos previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Instalación local

1. **Clona o descarga el repositorio**
```bash
git clone <URL-del-repositorio>
cd coleoptilo
```

2. **Crea un entorno virtual (opcional pero recomendado)**
```bash
python -m venv venv
# En Windows:
venv\Scripts\activate
# En macOS/Linux:
source venv/bin/activate
```

3. **Instala las dependencias**
```bash
pip install -r requirements.txt
```

4. **Ejecuta la aplicación**
```bash
streamlit run streamlit_app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

## 🎮 Cómo usar la aplicación

### 1. Selecciona una opción de entrada
En la barra lateral, elige cómo deseas capturar la imagen:
- **📸 Captura desde cámara**: Toma una foto directamente
- **📁 Cargar archivo**: Selecciona una imagen guardada
- **🎥 Grabación de video**: Extrae el primer fotograma de un video

### 2. Ajusta los parámetros de segmentación
- **Número de segmentos esperados**: Indica cuántos coleóptilos esperas detectar
- **Saturación mínima y máxima**: Controla qué tonalidades se detectan
- **Auto-ajustar parámetros**: Busca automáticamente los mejores valores

### 3. Calibra la placa de referencia
Los sliders permiten ajustar el círculo de la placa:
- **Desplazamiento X**: Movimiento horizontal desde el centro
- **Desplazamiento Y**: Movimiento vertical desde el centro
- **Radio**: Tamaño del círculo

### 4. Visualiza los resultados
- Imagen original (izquierda)
- Imagen con detecciones y mediciones (derecha)
- Tabla de mediciones con valores en píxeles y milímetros
- Estadísticas agregadas

### 5. Descarga los resultados
Haz clic en "📥 Descargar resultados (CSV)" para guardar las mediciones

## 📋 Parámetros de segmentación

| Parámetro | Rango | Valor por defecto | Descripción |
|-----------|-------|-------------------|------------|
| Saturación mínima | 0-255 | 50 | Umbral inferior de saturación HSV |
| Saturación máxima | 0-255 | 255 | Umbral superior de saturación HSV |
| Segmentos esperados | 1-50 | 5 | Número de coleóptilos a detectar |

## 🔧 Requisitos del sistema

### Dependencias Python

```
numpy>=1.21.0           # Cálculos numéricos
scipy>=1.7.0            # Interpolación y análisis científico
opencv-python>=4.6.0    # Procesamiento de imágenes
scikit-image>=0.19.0    # Segmentación y análisis de imágenes
streamlit>=1.28.0       # Marco web interactivo
pillow>=8.0.0           # Manejo de imágenes
networkx>=2.6.0         # Análisis de grafos
```

### Requisitos de hardware

- **Cámara web** (opcional, solo para captura en vivo)
- **Memoria RAM**: Mínimo 512 MB
- **Procesador**: Cualquier procesador moderno (sin aceleración GPU necesaria)

## 📐 Especificaciones técnicas

### Formato de entrada
- **Imágenes**: JPG, JPEG, PNG, BMP
- **Videos**: MP4, AVI, MOV
- **Dimensiones**: Sin restricción (se adapta automáticamente)

### Espacio de color
- **Entrada**: BGR (OpenCV)
- **Procesamiento**: HSV
- **Salida**: RGB

### Calibración
- **Diámetro de placa de referencia**: 90 mm
- **Unidades de salida**: Píxeles y milímetros

### Mediciones

1. **Longitud del coleóptilo**:
   - Extraída del esqueleto suavizado
   - Calculada mediante interpolación polinomial paramétrica
   - Convertida a mm usando factor de calibración

2. **Radio de curvatura**:
   - Calculado a partir de la primera y segunda derivada
   - Medida de la curvatura local en cada punto
   - Radio mínimo reportado para cada coleóptilo

## 📊 Formato de salida (CSV)

```csv
ID,Longitud (px),Longitud (mm),Radio min (px),Radio min (mm)
1,245.67,2.45,125.34,1.25
2,198.45,1.98,98.76,0.99
...
```

## 🌐 Desplegar en Streamlit Cloud

1. **Sube el repositorio a GitHub** con estos archivos:
   - `streamlit_app.py`
   - `requirements.txt`
   - `packages.txt` (importante para librerías del sistema)
   - `.gitignore` (opcional)

2. **Ve a [Streamlit Cloud](https://share.streamlit.io/)**

3. **Crea una nueva app**:
   - Selecciona tu repositorio
   - Rama: `main`
   - Main file: `streamlit_app.py`

4. **Espera a que se instale** (puede tomar 3-5 minutos)

5. **Si hay error de OpenCV**:
   - Haz clic en "Manage app" → "Reboot app"
   - Si persiste, ve a "Settings" → "Advanced settings" → "Client error details" para ver logs
   - Verifica que `packages.txt` y `requirements.txt` estén en el repositorio

6. **¡Listo!** La app se desplegará automáticamente

## 🐛 Solución de problemas

### Error: `ImportError: cv2 cannot open shared object file`

**Solución:**
1. Asegúrate de que `packages.txt` esté en el repositorio con:
   ```
   libsm6
   libxext6
   libxrender-dev
   libgomp1
   ```
2. Verifica que `requirements.txt` use `opencv-python-headless` (no `opencv-python`)
3. Haz clic en "Manage app" → "Reboot app" en Streamlit Cloud
4. Si persiste, elimina y crea la app nuevamente

### La placa no se detecta correctamente
- Mejora la iluminación de la imagen
- Asegúrate de que la placa esté completamente visible
- Ajusta manualmente usando los sliders

### No se detectan suficientes coleóptilos
- Ajusta los valores de saturación mínima y máxima
- Usa la opción "Auto-ajustar parámetros"
- Verifica el contraste de la imagen

### La app es lenta
- Reduce la resolución de la imagen
- Simplifica los valores de parámetros

## 📝 Notas de uso

- La placa debe estar completamente visible en la imagen
- El fondo debe tener buen contraste con los coleóptilos
- Para mejor precisión, usa luz uniforme
- Las mediciones en mm requieren calibración correcta de la placa

## 👨‍💻 Desarrollo

Para modificar o extender la aplicación:

1. Clona el repositorio
2. Crea una rama: `git checkout -b feature/mi-mejora`
3. Realiza los cambios
4. Sube la rama: `git push origin feature/mi-mejora`
5. Crea un Pull Request

## 📄 Licencia

Este proyecto es de código abierto y está disponible bajo licencia [Especificar licencia]

## 👥 Contacto

Para reportar problemas o sugerencias:
- Email: [tu-email]
- GitHub Issues: [enlace al repositorio]

---

**Última actualización**: Enero 2026  
**Versión**: 1.0.0

## 🎓 Uso académico

Esta herramienta ha sido desarrollada para fines de investigación y educación en el análisis de crecimiento de plantas.

**Citar como:**
```
Análisis de Coleóptilos v1.0. Universidad Autónoma de Madrid. 2026.
```

## 📚 Referencias

- OpenCV: https://opencv.org/
- Streamlit: https://streamlit.io/
- Scikit-image: https://scikit-image.org/

