import streamlit as st
import google.generativeai as genai
import os
# Ya no necesitamos python-dotenv explícitamente aquí si usamos st.secrets

# --- Configuración Inicial y Carga de API Key ---

# Configuración de la página de Streamlit
st.set_page_config(page_title="Exégesis Bíblica con Gemini", page_icon="✝️", layout="wide")

st.title("📜 Aplicación de Exégesis Bíblica con IA")
st.caption("Generada usando Google Gemini 1.5 Flash")

# Variable global para saber si la API está configurada
api_configured = False
gemini_api_key = None # Variable para almacenar la clave

# --- Funciones de la Lógica Principal (Adaptadas) ---

def configurar_api(api_key_to_configure):
    """Configura la API de Gemini con la clave proporcionada."""
    global api_configured
    if not api_key_to_configure:
        # Este caso no debería ocurrir si llamamos correctamente, pero por seguridad
        st.error("❌ No se proporcionó una clave API para configurar.")
        api_configured = False
        return False
    try:
        genai.configure(api_key=api_key_to_configure)
        # Pequeña prueba para verificar la clave (opcional pero útil)
        # _ = genai.list_models()
        api_configured = True
        return True # Éxito
    except Exception as e:
        st.error(f"❌ Error al configurar la API de Gemini con la clave proporcionada: {e}")
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
         # Este chequeo es una segunda capa de seguridad. La configuración debe hacerse antes.
         st.error("Error interno: La función generar_exegesis fue llamada sin que la API estuviera configurada.")
         return None

    try:
        # Ya no necesitamos configurar el modelo aquí si genai está configurado globalmente
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

# --- Sidebar para Configuración y Ayuda ---
with st.sidebar:
    st.header("Configuración de API Key")

    # Instrucciones claras sobre st.secrets
    st.markdown("""
    La forma recomendada y segura de configurar tu API Key de Google (Gemini) es usando los **Secretos de Streamlit**:
    1.  **Localmente:** Crea un archivo `.streamlit/secrets.toml` y añade `GEMINI_API_KEY = "TU_CLAVE"`.
    2.  **Desplegado (Streamlit Community Cloud):** Pega el contenido de tu `secrets.toml` en la configuración de Secretos de tu aplicación.
    """)
    st.markdown("---")

    # Intentar obtener la clave de st.secrets
    try:
        gemini_api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception as e:
         st.warning(f"No se pudo leer st.secrets (puede ser normal si no está configurado): {e}")
         gemini_api_key = None

    # Input manual como *alternativa* si no se encuentra en secrets
    api_key_input_manual = st.text_input(
        "O introduce tu API Key manualmente (menos seguro):",
        type="password",
        help="Es preferible usar st.secrets.",
        key="manual_api_key" # Añadir una clave única para el widget
    )

    # Botón explícito para intentar configurar si se usa la entrada manual
    if api_key_input_manual and not gemini_api_key: # Mostrar solo si no hay secreto y hay texto
        if st.button("Configurar con clave manual"):
            if configurar_api(api_key_input_manual):
                st.success("✅ API Key configurada manualmente.")
                gemini_api_key = api_key_input_manual # Guardar la clave que funcionó
            else:
                # El error ya se muestra en configurar_api
                pass
    elif gemini_api_key and not api_configured:
         # Si la clave se encontró en secrets, intentar configurar automáticamente
         if configurar_api(gemini_api_key):
              st.success("✅ API Key cargada y configurada desde st.secrets.")
         else:
              # El error ya se muestra en configurar_api
              st.warning("Clave encontrada en st.secrets, pero falló la configuración.")
    elif api_configured:
        st.success("✅ API Key ya configurada.") # Indicar que ya está lista
    else:
        st.info("Esperando configuración de API Key...")


    st.markdown("---")
    st.subheader("Instrucciones de Uso")
    st.markdown("""
    1.  Asegúrate de que la **API Key esté configurada** (preferiblemente vía `st.secrets`).
    2.  Escribe el **Libro** y **Capítulo**.
    3.  Haz clic en **"Generar Exégesis"**.
    """)
    st.markdown("---")
    st.warning("""
    **Importante:** La IA es una herramienta. Verifica siempre las fuentes y consulta múltiples recursos académicos.
    """)


# --- Interfaz Principal de Streamlit ---

# Columnas para los inputs
col1, col2 = st.columns(2)

with col1:
    libro_input = st.text_input("Libro de la Biblia:", placeholder="Ej. Génesis", key="libro")

with col2:
    capitulo_input = st.number_input("Capítulo:", min_value=1, step=1, format="%d", placeholder="Ej. 1", key="capitulo")

# Botón para iniciar la generación, deshabilitado si la API no está configurada
generar_btn = st.button(
    "✨ Generar Exégesis ✨",
    type="primary",
    use_container_width=True,
    disabled=not api_configured # El botón se activa solo si api_configured es True
)

# --- Lógica de Ejecución al Presionar el Botón ---
if generar_btn:
    # Validaciones de entrada (la API ya debe estar configurada para que el botón esté activo)
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
