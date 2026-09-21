import streamlit as st
import numpy as np
import pandas as pd
import pickle
import re
import io
import datetime
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="GRU NER Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM LIGHT THEME
# ============================================================

st.markdown("""
<style>

    /* ==============================
       GLOBAL
       ============================== */

    .stApp {
        background-color: #f7f9fc;
    }

    .main .block-container {
        max-width: 1450px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    /* ==============================
       SIDEBAR
       ============================== */

    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }

    section[data-testid="stSidebar"] * {
        color: #172033;
    }

    /* ==============================
       HEADINGS
       ============================== */

    .main-title {
        font-size: 36px;
        font-weight: 800;
        color: #172033;
        margin-bottom: 4px;
    }

    .subtitle {
        color: #667085;
        font-size: 16px;
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 750;
        color: #172033;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    /* ==============================
       STATUS
       ============================== */

    .status-box {
        background-color: #ecfdf3;
        border: 1px solid #abefc6;
        color: #027a48;
        border-radius: 25px;
        padding: 8px 15px;
        font-weight: 700;
        text-align: center;
    }

    .offline-box {
        background-color: #fef3f2;
        border: 1px solid #fecdca;
        color: #b42318;
        border-radius: 25px;
        padding: 8px 15px;
        font-weight: 700;
        text-align: center;
    }

    /* ==============================
       METRIC CARDS
       ============================== */

    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e4e7ec;
        border-radius: 16px;
        padding: 20px;
        min-height: 125px;
        box-shadow: 0 3px 12px rgba(16, 24, 40, 0.05);
    }

    .metric-label {
        color: #667085;
        font-size: 14px;
        font-weight: 600;
    }

    .metric-value {
        color: #172033;
        font-size: 29px;
        font-weight: 800;
        margin-top: 8px;
    }

    .metric-description {
        color: #98a2b3;
        font-size: 12px;
        margin-top: 5px;
    }

    /* ==============================
       ENTITY HIGHLIGHTING
       ============================== */

    .highlight-box {
        background-color: #ffffff;
        border: 1px solid #e4e7ec;
        border-radius: 16px;
        padding: 25px;
        line-height: 2.3;
        font-size: 17px;
        box-shadow: 0 3px 12px rgba(16, 24, 40, 0.04);
    }

    .person {
        background-color: #dcfce7;
        color: #166534;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 700;
    }

    .location {
        background-color: #dbeafe;
        color: #1d4ed8;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 700;
    }

    .organization {
        background-color: #ffedd5;
        color: #c2410c;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 700;
    }

    /* ==============================
       INFO BOX
       ============================== */

    .info-box {
        background-color: #eef4ff;
        border: 1px solid #c7d7fe;
        border-radius: 15px;
        padding: 20px;
        color: #344054;
    }

    /* ==============================
       HISTORY ITEM
       ============================== */

    .history-item {
        background-color: #ffffff;
        border: 1px solid #e4e7ec;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }

    .history-time {
        color: #98a2b3;
        font-size: 12px;
    }

    /* ==============================
       FOOTER
       ============================== */

    .footer {
        text-align: center;
        color: #98a2b3;
        font-size: 13px;
        padding: 35px 0 10px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL FILES
# ============================================================

MODEL_PATH = "models/gru_ner_model.keras"
WORD2ID_PATH = "models/word2id.pkl"
ID2LABEL_PATH = "models/id2label.pkl"

MAX_LEN = 128
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model, True, None

    except Exception as exc:
        return None, False, str(exc)


# ============================================================
# LOAD VOCABULARY
# ============================================================

@st.cache_resource
def load_vocabulary():

    try:

        with open(WORD2ID_PATH, "rb") as file:
            word2id = pickle.load(file)

        with open(ID2LABEL_PATH, "rb") as file:
            id2label = pickle.load(file)

        return word2id, id2label, True, None

    except Exception as exc:

        return None, None, False, str(exc)


model, model_loaded, model_error = load_model()
word2id, id2label, vocabulary_loaded, vocab_error = load_vocabulary()

# Session-level history of analyses (kept in memory only)
if "history" not in st.session_state:
    st.session_state["history"] = []


# ============================================================
# TOKENIZER
# ============================================================

def tokenize_text(text):

    return re.findall(r"\w+|[^\w\s]", text)


# ============================================================
# PREDICTION
# ============================================================

def predict_entities(text):

    words = tokenize_text(text)

    if not words:
        return []

    if not model_loaded or not vocabulary_loaded:
        return []

    sequence = [
        word2id.get(
            word,
            word2id.get(UNK_TOKEN, 1)
        )
        for word in words
    ]

    sequence = sequence[:MAX_LEN]

    padded_sequence = sequence + [
        word2id.get(PAD_TOKEN, 0)
    ] * (MAX_LEN - len(sequence))

    input_array = np.array(
        [padded_sequence]
    )

    predictions = model.predict(
        input_array,
        verbose=0
    )

    predicted_ids = np.argmax(
        predictions,
        axis=-1
    )[0]

    results = []

    for i, word in enumerate(words[:MAX_LEN]):

        predicted_id = int(
            predicted_ids[i]
        )

        label = id2label.get(
            predicted_id,
            "O"
        )

        confidence = float(
            predictions[0][i][predicted_id]
        )

        results.append({
            "word": word,
            "label": label,
            "confidence": confidence
        })

    return results


# ============================================================
# ENTITY EXTRACTION
# ============================================================

def extract_entities(results):

    entities = []

    current_words = []
    current_type = None
    current_confidences = []

    for item in results:

        word = item["word"]
        label = item["label"]
        confidence = item["confidence"]

        # --------------------------------
        # Beginning of entity
        # --------------------------------

        if label.startswith("B-"):

            if current_words:

                entities.append({
                    "text": " ".join(current_words),
                    "type": current_type,
                    "confidence": float(
                        np.mean(current_confidences)
                    )
                })

            current_words = [word]
            current_type = label[2:]
            current_confidences = [confidence]

        # --------------------------------
        # Inside entity
        # --------------------------------

        elif label.startswith("I-"):

            entity_type = label[2:]

            if (
                current_words
                and entity_type == current_type
            ):

                current_words.append(word)
                current_confidences.append(
                    confidence
                )

            else:

                if current_words:

                    entities.append({
                        "text": " ".join(current_words),
                        "type": current_type,
                        "confidence": float(
                            np.mean(current_confidences)
                        )
                    })

                current_words = [word]
                current_type = entity_type
                current_confidences = [confidence]

        # --------------------------------
        # Outside
        # --------------------------------

        else:

            if current_words:

                entities.append({
                    "text": " ".join(current_words),
                    "type": current_type,
                    "confidence": float(
                        np.mean(current_confidences)
                    )
                })

            current_words = []
            current_type = None
            current_confidences = []

    # --------------------------------
    # Last entity
    # --------------------------------

    if current_words:

        entities.append({
            "text": " ".join(current_words),
            "type": current_type,
            "confidence": float(
                np.mean(current_confidences)
            )
        })

    return entities


# ============================================================
# HIGHLIGHT TEXT
# ============================================================

def highlight_text(results):

    html_output = ""

    for item in results:

        word = item["word"]
        label = item["label"]

        safe_word = (
            word
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

        if label.endswith("PER"):

            html_output += (
                f'<span class="person">'
                f'{safe_word}'
                f'</span> '
            )

        elif label.endswith("LOC"):

            html_output += (
                f'<span class="location">'
                f'{safe_word}'
                f'</span> '
            )

        elif label.endswith("ORG"):

            html_output += (
                f'<span class="organization">'
                f'{safe_word}'
                f'</span> '
            )

        else:

            html_output += (
                f'{safe_word} '
            )

    return html_output


# ============================================================
# METRIC CARD
# ============================================================

def metric_card(
    title,
    value,
    description
):

    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">{title}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-description">{description}</div>'
        f'</div>',
        unsafe_allow_html=True
    )


# ============================================================
# HEADER
# ============================================================

def render_header():

    left, right = st.columns(
        [5, 1]
    )

    with left:

        st.markdown(
            '<div class="main-title">'
            '🧠 GRU NER Intelligence'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="subtitle">'
            'Named Entity Recognition powered by '
            'a Bidirectional GRU model'
            '</div>',
            unsafe_allow_html=True
        )

    with right:

        if model_loaded and vocabulary_loaded:

            st.markdown(
                '<div class="status-box">'
                '● Model Ready'
                '</div>',
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                '<div class="offline-box">'
                '● Model Not Found'
                '</div>',
                unsafe_allow_html=True
            )


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():

    st.sidebar.markdown(
        "## 🧠 GRU NER"
    )

    st.sidebar.caption(
        "AI Named Entity Recognition"
    )

    st.sidebar.divider()

    page = st.sidebar.radio(
        "Navigation",
        [
            "📊 Dashboard",
            "🔎 Text Analyzer",
            "📈 Entity Analytics",
            "🎯 Model Performance",
            "🕓 History",
            "ℹ️ About Model"
        ]
    )

    st.sidebar.divider()

    st.sidebar.markdown(
        "### Model Status"
    )

    if model_loaded:

        st.sidebar.success(
            "Model loaded successfully"
        )

    else:

        st.sidebar.error(
            "Model file not found"
        )

        with st.sidebar.expander("Details"):
            st.code(model_error or "Unknown error")

    if vocabulary_loaded:

        st.sidebar.success(
            "Vocabulary loaded successfully"
        )

    else:

        st.sidebar.error(
            "Vocabulary files not found"
        )

        with st.sidebar.expander("Details"):
            st.code(vocab_error or "Unknown error")

    st.sidebar.markdown(
        "### Supported Entities"
    )

    st.sidebar.markdown(
        """
        🟢 **PERSON**

        🔵 **LOCATION**

        🟠 **ORGANIZATION**
        """
    )

    st.sidebar.divider()

    st.sidebar.caption(
        f"Analyses this session: {len(st.session_state['history'])}"
    )

    return page


# ============================================================
# PIPELINE CARD
# ============================================================

def pipeline_card(
    icon,
    title,
    description
):

    with st.container(border=True):

        st.markdown(
            f"<div style='text-align:center;'>"
            f"<div style='font-size:34px;'>{icon}</div>"
            f"<h4 style='margin-bottom:5px;'>{title}</h4>"
            f"<p style='color:#667085;font-size:13px;'>"
            f"{description}"
            f"</p>"
            f"</div>",
            unsafe_allow_html=True
        )


# ============================================================
# DASHBOARD
# ============================================================

def dashboard():

    st.markdown(
        '<div class="section-title">'
        'Model Overview'
        '</div>',
        unsafe_allow_html=True
    )

    cols = st.columns(5)

    with cols[0]:

        metric_card(
            "Architecture",
            "BiGRU",
            "Bidirectional GRU"
        )

    with cols[1]:

        metric_card(
            "Task",
            "NER",
            "Named Entity Recognition"
        )

    with cols[2]:

        metric_card(
            "Sequence Length",
            "128",
            "Maximum tokens"
        )

    with cols[3]:

        metric_card(
            "Embedding",
            "128",
            "Embedding dimensions"
        )

    with cols[4]:

        metric_card(
            "Entity Types",
            "3",
            "Person, Location, Organization"
        )

    # ========================================================
    # PERFORMANCE
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'Model Performance'
        '</div>',
        unsafe_allow_html=True
    )

    cols = st.columns(4)

    with cols[0]:

        metric_card(
            "Accuracy",
            "94.41%",
            "Test accuracy"
        )

    with cols[1]:

        metric_card(
            "Precision",
            "72.65%",
            "Entity precision"
        )

    with cols[2]:

        metric_card(
            "Recall",
            "63.36%",
            "Entity recall"
        )

    with cols[3]:

        metric_card(
            "F1 Score",
            "67.69%",
            "Overall F1"
        )

    # ========================================================
    # PROCESSING PIPELINE
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'NER Processing Pipeline'
        '</div>',
        unsafe_allow_html=True
    )

    pipeline_steps = [

        (
            "📝",
            "Input Text",
            "Raw text"
        ),

        (
            "✂️",
            "Tokenization",
            "Split text into tokens"
        ),

        (
            "🔢",
            "Encoding",
            "Convert words to IDs"
        ),

        (
            "🧠",
            "BiGRU",
            "Learn sequence context"
        ),

        (
            "🏷️",
            "NER Labels",
            "Classify entities"
        )

    ]

    cols = st.columns(5)

    for i, step in enumerate(
        pipeline_steps
    ):

        with cols[i]:

            pipeline_card(
                step[0],
                step[1],
                step[2]
            )

    # ========================================================
    # ENTITY TYPES
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'Supported Entity Classes'
        '</div>',
        unsafe_allow_html=True
    )

    cols = st.columns(3)

    with cols[0]:

        with st.container(border=True):

            st.markdown("### 🟢 PERSON")

            st.markdown(
                "Names of people"
            )

            st.metric(
                "Precision",
                "70%"
            )

    with cols[1]:

        with st.container(border=True):

            st.markdown("### 🔵 LOCATION")

            st.markdown(
                "Cities, countries and places"
            )

            st.metric(
                "Precision",
                "80%"
            )

    with cols[2]:

        with st.container(border=True):

            st.markdown("### 🟠 ORGANIZATION")

            st.markdown(
                "Companies and institutions"
            )

            st.metric(
                "Precision",
                "67%"
            )


# ============================================================
# TEXT ANALYZER
# ============================================================

def text_analyzer():

    st.markdown(
        '<div class="section-title">'
        '🔎 Text Analyzer'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter English text, paste it, or upload a .txt file. "
        "The trained GRU model will identify named entities."
    )

    samples = {

        "Sample 1":
            "Bill Gates founded Microsoft in the United States and later visited Paris.",

        "Sample 2":
            "Barack Obama met Angela Merkel in Berlin to discuss cooperation between the United Nations and the European Union.",

        "Sample 3":
            "Apple opened a new office in London, while Google expanded its operations in California.",

        "Sample 4":
            "Elon Musk announced that SpaceX will launch a new rocket from Florida. The company is working with NASA on future missions to the Moon and Mars."

    }

    source_choice = st.selectbox(
        "Choose a source",
        ["Custom Text", "Upload .txt file"] + list(samples.keys())
    )

    default_text = ""

    if source_choice in samples:
        default_text = samples[source_choice]

    uploaded_file = None

    if source_choice == "Upload .txt file":

        uploaded_file = st.file_uploader(
            "Upload a plain text file",
            type=["txt"]
        )

        if uploaded_file is not None:
            default_text = uploaded_file.read().decode(
                "utf-8", errors="ignore"
            )

    text = st.text_area(
        "Input Text",
        value=default_text,
        height=170,
        placeholder=(
            "Type or paste your text here..."
        )
    )

    # ========================================================
    # TEXT STATISTICS
    # ========================================================

    if text:

        words = len(
            text.split()
        )

        characters = len(text)

        sentences = len(
            re.findall(
                r"[.!?]+",
                text
            )
        )

        cols = st.columns(3)

        with cols[0]:

            metric_card(
                "Characters",
                characters,
                "Total characters"
            )

        with cols[1]:

            metric_card(
                "Words",
                words,
                "Words in input"
            )

        with cols[2]:

            metric_card(
                "Sentences",
                sentences,
                "Detected sentences"
            )

    # ========================================================
    # BUTTONS
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        analyze = st.button(
            "🔍 Analyze Text",
            type="primary",
            use_container_width=True
        )

    with col2:

        clear = st.button(
            "🗑️ Clear",
            use_container_width=True
        )

    if clear:

        if "results" in st.session_state:

            del st.session_state["results"]

        if "entities" in st.session_state:

            del st.session_state["entities"]

        st.rerun()

    # ========================================================
    # ANALYZE
    # ========================================================

    if analyze:

        if not text.strip():

            st.warning(
                "Please enter some text first."
            )

        elif not model_loaded:

            st.error(
                "The trained GRU model could not be loaded."
            )

        elif not vocabulary_loaded:

            st.error(
                "Vocabulary files could not be loaded."
            )

        else:

            with st.spinner("Running inference..."):

                results = predict_entities(
                    text
                )

                entities = extract_entities(
                    results
                )

            st.session_state[
                "results"
            ] = results

            st.session_state[
                "entities"
            ] = entities

            st.session_state["history"].append({
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "text": text[:160] + ("..." if len(text) > 160 else ""),
                "entity_count": len(entities),
                "persons": sum(1 for e in entities if e["type"] == "PER"),
                "locations": sum(1 for e in entities if e["type"] == "LOC"),
                "organizations": sum(1 for e in entities if e["type"] == "ORG"),
            })

    # ========================================================
    # RESULTS
    # ========================================================

    if (
        "results"
        in st.session_state
    ):

        results = st.session_state[
            "results"
        ]

        entities = st.session_state[
            "entities"
        ]

        # ====================================================
        # HIGHLIGHTED TEXT
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            'Highlighted Text'
            '</div>',
            unsafe_allow_html=True
        )

        highlighted = highlight_text(
            results
        )

        st.markdown(
            f'<div class="highlight-box">{highlighted}</div>',
            unsafe_allow_html=True
        )

        # ====================================================
        # ENTITY COUNTS
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            'Detected Entities'
            '</div>',
            unsafe_allow_html=True
        )

        person_count = sum(
            1
            for entity in entities
            if entity["type"] == "PER"
        )

        location_count = sum(
            1
            for entity in entities
            if entity["type"] == "LOC"
        )

        organization_count = sum(
            1
            for entity in entities
            if entity["type"] == "ORG"
        )

        cols = st.columns(3)

        with cols[0]:

            metric_card(
                "🟢 Persons",
                person_count,
                "People detected"
            )

        with cols[1]:

            metric_card(
                "🔵 Locations",
                location_count,
                "Places detected"
            )

        with cols[2]:

            metric_card(
                "🟠 Organizations",
                organization_count,
                "Organizations detected"
            )

        # ====================================================
        # ENTITY TABLE
        # ====================================================

        if entities:

            table_data = []

            for entity in entities:

                table_data.append({
                    "Entity":
                        entity["text"],

                    "Type":
                        entity["type"],

                    "Confidence":
                        f"{entity['confidence'] * 100:.2f}%"
                })

            df = pd.DataFrame(
                table_data
            )

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)

            st.download_button(
                "⬇️ Download entities as CSV",
                data=csv_buffer.getvalue(),
                file_name="ner_entities.csv",
                mime="text/csv",
                use_container_width=True
            )

            # =================================================
            # CONFIDENCE CHART
            # =================================================

            st.markdown(
                '<div class="section-title">'
                'Entity Confidence'
                '</div>',
                unsafe_allow_html=True
            )

            chart_df = pd.DataFrame({

                "Entity": [
                    e["text"]
                    for e in entities
                ],

                "Confidence": [
                    e["confidence"] * 100
                    for e in entities
                ],

                "Type": [
                    e["type"]
                    for e in entities
                ]

            })

            fig = px.bar(
                chart_df,
                x="Entity",
                y="Confidence",
                color="Type",
                text_auto=".1f"
            )

            fig.update_layout(
                template="plotly_white",
                height=430,
                yaxis_title="Confidence (%)",
                xaxis_title="Entity"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.info(
                "No PERSON, LOCATION or ORGANIZATION "
                "entities were detected."
            )

        # ====================================================
        # TOKEN ANALYSIS
        # ====================================================

        with st.expander(
            "🔬 View Token-Level Analysis"
        ):

            token_data = pd.DataFrame(
                results
            )

            token_data[
                "confidence"
            ] = (
                token_data["confidence"] * 100
            ).round(2)

            token_data.columns = [
                "Token",
                "NER Label",
                "Confidence (%)"
            ]

            st.dataframe(
                token_data,
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# ENTITY ANALYTICS
# ============================================================

def entity_analytics():

    st.markdown(
        '<div class="section-title">'
        '📈 Entity Analytics'
        '</div>',
        unsafe_allow_html=True
    )

    data = pd.DataFrame({

        "Entity Type": [
            "PERSON",
            "LOCATION",
            "ORGANIZATION"
        ],

        "Precision": [
            70,
            80,
            67
        ],

        "Recall": [
            55,
            74,
            61
        ],

        "F1 Score": [
            62,
            77,
            64
        ]

    })

    # ========================================================
    # PERFORMANCE CHART
    # ========================================================

    fig = px.bar(
        data,
        x="Entity Type",
        y=[
            "Precision",
            "Recall",
            "F1 Score"
        ],
        barmode="group",
        text_auto=True
    )

    fig.update_layout(
        template="plotly_white",
        height=480,
        yaxis_title="Score (%)",
        xaxis_title=""
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ========================================================
    # DISTRIBUTION
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'Entity Distribution'
        '</div>',
        unsafe_allow_html=True
    )

    distribution = pd.DataFrame({

        "Entity": [
            "PERSON",
            "LOCATION",
            "ORGANIZATION"
        ],

        "Count": [
            1617,
            1668,
            1661
        ]

    })

    col1, col2 = st.columns(2)

    with col1:

        fig = px.pie(
            distribution,
            names="Entity",
            values="Count",
            hole=0.45
        )

        fig.update_layout(
            template="plotly_white",
            height=430
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.bar(
            distribution,
            x="Entity",
            y="Count",
            text="Count"
        )

        fig.update_layout(
            template="plotly_white",
            height=430
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

def model_performance():

    st.markdown(
        '<div class="section-title">'
        '🎯 Model Performance'
        '</div>',
        unsafe_allow_html=True
    )

    # ========================================================
    # TOP METRICS
    # ========================================================

    cols = st.columns(4)

    performance_metrics = [

        (
            "Accuracy",
            "94.41%",
            "Test Accuracy"
        ),

        (
            "Precision",
            "72.65%",
            "Entity Precision"
        ),

        (
            "Recall",
            "63.36%",
            "Entity Recall"
        ),

        (
            "F1 Score",
            "67.69%",
            "Overall F1"
        )

    ]

    for col, metric in zip(
        cols,
        performance_metrics
    ):

        with col:

            metric_card(
                metric[0],
                metric[1],
                metric[2]
            )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'Classification Report'
        '</div>',
        unsafe_allow_html=True
    )

    report = pd.DataFrame({

        "Entity": [
            "PERSON",
            "LOCATION",
            "ORGANIZATION"
        ],

        "Precision": [
            "70%",
            "80%",
            "67%"
        ],

        "Recall": [
            "55%",
            "74%",
            "61%"
        ],

        "F1 Score": [
            "62%",
            "77%",
            "64%"
        ]

    })

    st.dataframe(
        report,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # PERFORMANCE CHART
    # ========================================================

    chart_report = pd.DataFrame({

        "Entity": [
            "PERSON",
            "LOCATION",
            "ORGANIZATION"
        ],

        "Precision": [
            70,
            80,
            67
        ],

        "Recall": [
            55,
            74,
            61
        ],

        "F1 Score": [
            62,
            77,
            64
        ]

    })

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Precision",
            x=chart_report["Entity"],
            y=chart_report["Precision"]
        )
    )

    fig.add_trace(
        go.Bar(
            name="Recall",
            x=chart_report["Entity"],
            y=chart_report["Recall"]
        )
    )

    fig.add_trace(
        go.Bar(
            name="F1 Score",
            x=chart_report["Entity"],
            y=chart_report["F1 Score"]
        )
    )

    fig.update_layout(
        barmode="group",
        template="plotly_white",
        height=450,
        yaxis_title="Score (%)",
        xaxis_title="Entity Type"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ========================================================
    # MODEL ARCHITECTURE
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'Model Architecture'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "The following visual represents the sequence "
        "of operations used by the trained GRU model."
    )

    architecture = [

        (
            "🔢",
            "Input",
            "128 tokens"
        ),

        (
            "📚",
            "Embedding",
            "128 dimensions"
        ),

        (
            "🧠",
            "BiGRU",
            "128 units"
        ),

        (
            "💧",
            "Dropout",
            "0.30"
        ),

        (
            "🎯",
            "Dense + Softmax",
            "7 NER labels"
        )

    ]

    cols = st.columns(5)

    for i, item in enumerate(
        architecture
    ):

        with cols[i]:

            with st.container(
                border=True
            ):

                st.markdown(
                    f"## {item[0]}"
                )

                st.markdown(
                    f"### {item[1]}"
                )

                st.caption(
                    item[2]
                )

    # ========================================================
    # ARCHITECTURE DETAILS
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'Architecture Details'
        '</div>',
        unsafe_allow_html=True
    )

    architecture_table = pd.DataFrame({

        "Layer": [
            "Input",
            "Embedding",
            "Bidirectional GRU",
            "Dropout",
            "Dense"
        ],

        "Configuration": [
            "128 tokens",
            "128 dimensions",
            "128 GRU units",
            "0.30",
            "7 output classes"
        ],

        "Purpose": [
            "Receive token IDs",
            "Convert IDs into vectors",
            "Learn forward and backward context",
            "Reduce overfitting",
            "Predict NER labels"
        ]

    })

    st.dataframe(
        architecture_table,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# HISTORY
# ============================================================

def history_page():

    st.markdown(
        '<div class="section-title">'
        '🕓 Analysis History'
        '</div>',
        unsafe_allow_html=True
    )

    history = st.session_state["history"]

    if not history:

        st.info(
            "No analyses yet this session. Run something in "
            "the Text Analyzer and it will show up here."
        )

        return

    col1, col2 = st.columns([1, 5])

    with col1:

        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state["history"] = []
            st.rerun()

    for item in reversed(history):

        st.markdown(
            f'<div class="history-item">'
            f'<div class="history-time">{item["time"]}</div>'
            f'<div style="margin:6px 0;">{item["text"]}</div>'
            f'<div style="font-size:13px;color:#344054;">'
            f'🟢 {item["persons"]} persons &nbsp; '
            f'🔵 {item["locations"]} locations &nbsp; '
            f'🟠 {item["organizations"]} organizations &nbsp; '
            f'· {item["entity_count"]} total entities'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    hist_df = pd.DataFrame(history)

    csv_buffer = io.StringIO()
    hist_df.to_csv(csv_buffer, index=False)

    st.download_button(
        "⬇️ Download full history as CSV",
        data=csv_buffer.getvalue(),
        file_name="ner_analysis_history.csv",
        mime="text/csv"
    )


# ============================================================
# ABOUT MODEL
# ============================================================

def about_model():

    st.markdown(
        '<div class="section-title">'
        'ℹ️ About the Model'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="info-box">'
        '<h3>🧠 GRU-Based Named Entity Recognition</h3>'
        '<p>This application uses a Bidirectional GRU neural '
        'network for sequence-based Named Entity Recognition.</p>'
        '<p>The model processes text token-by-token and assigns '
        'an NER label to each token.</p>'
        '</div>',
        unsafe_allow_html=True
    )

    # ========================================================
    # SUPPORTED ENTITIES
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'Supported Entity Categories'
        '</div>',
        unsafe_allow_html=True
    )

    cols = st.columns(3)

    with cols[0]:

        with st.container(border=True):

            st.markdown(
                "### 🟢 PERSON"
            )

            st.write(
                "Names of people."
            )

            st.info(
                "Example: Bill Gates"
            )

    with cols[1]:

        with st.container(border=True):

            st.markdown(
                "### 🔵 LOCATION"
            )

            st.write(
                "Cities, countries and places."
            )

            st.info(
                "Example: Paris"
            )

    with cols[2]:

        with st.container(border=True):

            st.markdown(
                "### 🟠 ORGANIZATION"
            )

            st.write(
                "Companies, institutions and organizations."
            )

            st.info(
                "Example: Microsoft"
            )

    # ========================================================
    # CONFIGURATION
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'Technical Configuration'
        '</div>',
        unsafe_allow_html=True
    )

    configuration = pd.DataFrame({

        "Component": [
            "Architecture",
            "Embedding Dimension",
            "GRU Units",
            "Sequence Length",
            "Dropout",
            "Optimizer",
            "Task",
            "Framework"
        ],

        "Configuration": [
            "Bidirectional GRU",
            "128",
            "128",
            "128",
            "0.30",
            "Adam",
            "Named Entity Recognition",
            "TensorFlow / Keras"
        ]

    })

    st.dataframe(
        configuration,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # NER LABELS
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'NER Labels'
        '</div>',
        unsafe_allow_html=True
    )

    labels = pd.DataFrame({

        "Label": [
            "O",
            "B-PER",
            "I-PER",
            "B-ORG",
            "I-ORG",
            "B-LOC",
            "I-LOC"
        ],

        "Meaning": [
            "Outside an entity",
            "Beginning of person",
            "Inside person",
            "Beginning of organization",
            "Inside organization",
            "Beginning of location",
            "Inside location"
        ]

    })

    st.dataframe(
        labels,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # REQUIRED FILES
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'Required Files'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Place these three files in the same folder as this "
        "app so the model can load:"
    )

    files_table = pd.DataFrame({
        "File": [MODEL_PATH, WORD2ID_PATH, ID2LABEL_PATH],
        "Purpose": [
            "Trained Keras BiGRU model",
            "Word-to-index vocabulary mapping",
            "Label index-to-name mapping"
        ],
        "Status": [
            "✅ Found" if model_loaded else "❌ Missing",
            "✅ Found" if vocabulary_loaded else "❌ Missing",
            "✅ Found" if vocabulary_loaded else "❌ Missing",
        ]
    })

    st.dataframe(
        files_table,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

render_header()

page = render_sidebar()

st.divider()


if page == "📊 Dashboard":

    dashboard()


elif page == "🔎 Text Analyzer":

    text_analyzer()


elif page == "📈 Entity Analytics":

    entity_analytics()


elif page == "🎯 Model Performance":

    model_performance()


elif page == "🕓 History":

    history_page()


elif page == "ℹ️ About Model":

    about_model()


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">GRU NER Intelligence '
    '• Bidirectional GRU Named Entity Recognition '
    '• TensorFlow + Streamlit</div>',
    unsafe_allow_html=True
)
