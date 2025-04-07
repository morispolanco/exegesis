import streamlit as st
import google.generativeai as genai
import os

# --- Configuración Inicial y Carga de API Key ---

# Configuración de la página de Streamlit
st.set_page_config(page_title="Exégesis Bíblica con Gemini", page_icon="✝️", layout="wide")

st.title("📜 Aplicación de Exégesis Bíblica con IA")
st.caption("Generada usando Google Gemini 1.5 Flash")

# Variable global para saber si la API está configurada
api_configured = False
gemini_api_key = None # Variable para almacenar la clave

# --- Funciones de la Lógica Principal ---

def configurar_api(api_key_to_configure):
    """Configura la API de Gemini con la clave proporcionada. Devuelve True/False."""
    global api_configured
    if not api_key_to_configure:
        # Ya no mostramos error aquí, la verificación se hará fuera
        api_configured = False
        return False
    try:
        genai.configure(api_key=api_key_to_configure)
        # Pequeña prueba para verificar la clave (opcional pero útil)
        # _ = genai.list_models() # Descomentar si quieres una validación más estricta
        api_configured = True
        print("API de Gemini configurada correctamente.") # Mensaje para logs/consola
        return True # Éxito
    except Exception as e:
        # Mostramos el error en la interfaz principal más adelante
        st.error(f"❌ Error al configurar la API de Gemini: {e}. Verifica tu clave en st.secrets.")
        api_configured = False
        return False

def construir_prompt(libro, capitulo):
    """Construye el prompt detallado para la API de Gemini."""
    # (Misma función que antes - sin cambios)
    prompt = f"""
    **Tarea:** Realiza una exégesis académica exhaustiva del capítulo {capitulo} del libro de {libro}.

    **Instrucciones Obligatorias:**

    1.  **Enfoque Metodológico:** Utiliza principalmente un enfoque histórico-crítico y/o histórico-gramatical. Considera el contexto original del texto.
    2.  **Contextualización:**
        *   **Histórico-Cultural:** Describe el trasfondo histórico y cultural relevante para entender el capítulo (época, audiencia original, costumbres, situación política/social si aplica).
        *   **Literario:** Analiza el lugar del capítulo dentro del libro completo y su relación con los capítulos circundantes. Identifica el género literario predominante del capítulo (narrativa, poesía, ley, profecía, epístola, etc.).
    3.  **Análisis del Texto:**
        *   **Estructura:** Describe la estructura interna del capítulo, identificando secciones o unidades de pensamiento principales.
        *   **Contenido:** Resume el contenido principal del capítulo.
        *   **Análisis Detallado:** Examina versículos, frases o palabras clave que sean cruciales para la interpretación. Si es pertinente, menciona términos importantes en el idioma original (hebreo/arameo/griego), explicando su significado y relevancia en el contexto. Analiza aspectos gramaticales o sintácticos notables si afectan significativamente la interpretación.
    4.  **Temas Teológicos:** Identifica y discute los principales temas teológicos o doctrinales que surgen del texto. ¿Qué revela el capítulo sobre Dios, la humanidad, la salvación, la alianza, etc.?
    5.  **FUENTES ACADÉMICAS (IMPRESCINDIBLE Y CRÍTICO):**
        *   **CITA OBLIGATORIA:** **DEBES** citar fuentes académicas *verificables* para respaldar tus análisis, interpretaciones y afirmaciones sobre el contexto histórico o lingüístico.
        *   **Tipos de Fuentes Aceptables:** Comentarios bíblicos críticos reconocidos (ej. series como Anchor Yale Bible, Hermeneia, NICNT/NICOT, Word Biblical Commentary), artículos de revistas académicas especializadas (ej. Journal of Biblical Literature, New Testament Studies, Vetus Testamentum, Catholic Biblical Quarterly), monografías académicas publicadas por editoriales universitarias o académicas de prestigio (ej. Eerdmans, Brill, Mohr Siebeck, Oxford University Press, Cambridge University Press).
        *   **Tipos de Fuentes a EVITAR (a menos que se usen críticamente y se señale):** Fuentes puramente devocionales, sermones sin base académica, sitios web no académicos, comentarios populares sin rigor crítico.
        *   **Formato de Citas:** Usa un formato claro y consistente en el texto (ej. Apellido, Año, p. X) y proporciona una bibliografía breve al final con las referencias completas de las fuentes citadas. **La verificabilidad de las fuentes es esencial.**
    6.  **Lenguaje y Tono:** Mantén un lenguaje claro, objetivo y académico.
    7.  **Extensión:** Proporciona un análisis razonablemente detallado, no una simple descripción superficial.

    **Texto a Analizar:** {libro} {capitulo}

    **Idioma de la Respuesta:** Español
    """
    return prompt

def generar_exegesis(libro, capitulo):
    """Llama a la API de Gemini para generar la exégesis. Asume que la API ya está configurada."""
    if not api_configured:
         st.error("Error interno: La función generar_exegesis fue llamada sin que la API estuviera configurada.")
         return None

    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        prompt_texto = construir_prompt(libro, capitulo)

        generation_config = genai.GenerationConfig(
            temperature=0.6,
        )
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]

        with st.spinner(f'🤖 Analizando {libro} {capitulo}... Esto puede tardar unos momentos...'):
            response = model.generate_content(
                prompt_texto,
                generation_config=generation_config,
                safety_settings=safety_settings
            )

        if response.parts:
            return response.text
        else:
            try:
                feedback = response.prompt_feedback
                block_reason = feedback.block_reason if feedback else "No especificada"
                st.error(f"Error: La API no devolvió contenido. Posible razón: {block_reason}.")
                return None
            except Exception as feedback_err:
                 st.error(f"Error: La API no devolvió contenido y no se pudo obtener feedback adicional: {feedback_err}")
                 return None

    except Exception as e:
        st.error(f"❌ Error durante la generación con la API de Gemini: {e}")
        return None

# --- INTENTO DE CONFIGURACIÓN AUTOMÁTICA DE LA API ---
# Esta parte se ejecuta *antes* de renderizar la interfaz principal
try:
    gemini_api_key = st.secrets.get("GEMINI_API_KEY")
    if gemini_api_key:
        # Intentar configurar la API si se encontró la clave
        configurar_api(gemini_api_key)
    else:
        # La clave no está en secrets, api_configured seguirá siendo False
        pass # No hacer nada más aquí, el mensaje de error se mostrará abajo
except Exception as e:
    # Manejar error si st.secrets no está disponible (ej. entorno local sin .streamlit/secrets.toml)
    # api_configured seguirá siendo False
    # st.warning(f"No se pudo acceder a st.secrets. Asegúrate de configurar secrets.toml localmente o en la plataforma de despliegue. Error: {e}")
    pass # El mensaje principal de error será suficiente

# --- Sidebar (Simplificada) ---
with st.sidebar:
    st.header("Instrucciones de Uso")
    st.markdown("""
    1.  Asegúrate de que tu API Key de Google (Gemini) esté configurada correctamente en los **Secretos de Streamlit** (clave `GEMINI_API_KEY`).
        *   **Local:** Archivo `.streamlit/secrets.toml`.
        *   **Desplegado:** Sección de Secretos de la plataforma.
    2.  Escribe el **Libro** y **Capítulo** en la pantalla principal.
    3.  Haz clic en **"Generar Exégesis"**.
    """)
    st.markdown("---")
    st.warning("""
    **Importante:** La IA es una herramienta. Verifica siempre las fuentes y consulta múltiples recursos académicos.
    """)
    st.markdown("---")
    # Puedes añadir aquí otros elementos no relacionados con la API Key si lo deseas


# --- Interfaz Principal de Streamlit ---

# Mostrar mensaje de error si la API no se pudo configurar
if not api_configured:
    st.error("🛑 **Error de Configuración:** No se encontró o no es válida la API Key de Gemini (`GEMINI_API_KEY`) en los Secretos de Streamlit. La aplicación no puede funcionar.")
    st.info("Por favor, configura la clave API en los secretos (localmente en `.streamlit/secrets.toml` o en la configuración de despliegue) y refresca la página.")

# Columnas para los inputs (se mostrarán siempre, pero el botón estará deshabilitado si no hay API)
col1, col2 = st.columns(2)

with col1:
    libro_input = st.text_input("Libro de la Biblia:", placeholder="Ej. Génesis", key="libro", disabled=not api_configured)

with col2:
    capitulo_input = st.number_input("Capítulo:", min_value=1, step=1, format="%d", placeholder="Ej. 1", key="capitulo", disabled=not api_configured)

# Botón para iniciar la generación, deshabilitado si la API no está configurada
generar_btn = st.button(
    "✨ Generar Exégesis ✨",
    type="primary",
    use_container_width=True,
    disabled=not api_configured # El botón se activa solo si api_configured es True
)

# --- Lógica de Ejecución al Presionar el Botón ---
if generar_btn and api_configured: # Doble chequeo por si acaso
    # Validaciones de entrada
    if not libro_input:
        st.warning("⚠️ Por favor, introduce el nombre del libro.")
    elif not capitulo_input or capitulo_input < 1:
        st.warning("⚠️ Por favor, introduce un número de capítulo válido (mayor que 0).")
    else:
        # Llamar a la función de generación
        resultado = generar_exegesis(libro_input.strip(), int(capitulo_input))

        if resultado:
            st.subheader(f"📖 Exégesis de {libro_input.strip()} {capitulo_input}")
            st.markdown("---")
            st.markdown(resultado, unsafe_allow_html=True) # Usar markdown para formato
            st.success("✅ Exégesis generada con éxito.")
        # Los errores durante la generación ya se muestran dentro de generar_exegesis
elif generar_btn and not api_configured:
    # Esto no debería suceder porque el botón está deshabilitado, pero como salvaguarda
    st.error("La API Key no está configurada. No se puede generar la exégesis.")
