import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
import re
import os
from pathlib import Path
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

st.set_page_config(
    page_title="Fake News Detection with Explainability",
    page_icon="📰",
    layout="wide"
)

st.markdown("""
<style>
.main {
    padding-top: 1.5rem;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1100px;
}
.title-text {
    font-size: 3rem;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}
.subtitle-text {
    font-size: 1.05rem;
    color: #9aa4b2;
    margin-bottom: 1.5rem;
}
.card {
    background-color: #111827;
    border: 1px solid #1f2937;
    border-radius: 16px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
}
.metric-card {
    background: linear-gradient(135deg, #111827, #1f2937);
    border: 1px solid #2d3748;
    border-radius: 18px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-label {
    font-size: 0.95rem;
    color: #9aa4b2;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
}
.fake-box {
    background: rgba(239, 68, 68, 0.12);
    border: 1px solid rgba(239, 68, 68, 0.35);
    border-radius: 16px;
    padding: 0.9rem 1rem;
    margin-top: 0.5rem;
}
.real-box {
    background: rgba(34, 197, 94, 0.12);
    border: 1px solid rgba(34, 197, 94, 0.35);
    border-radius: 16px;
    padding: 0.9rem 1rem;
    margin-top: 0.5rem;
}
.small-note {
    color: #9aa4b2;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "saved_model"

if not MODEL_PATH.exists():
    st.error(f"saved_model folder not found at: {MODEL_PATH}")
    st.stop()

@st.cache_resource
def load_model():
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )
    tokenizer = DistilBertTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )
    model.to(device)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

def predict_news(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    label_map = {0: "REAL", 1: "FAKE"}
    return label_map[pred], confidence

def get_fake_score(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    return probs[0][1].item()

def perturbation_explanation(text, top_k=10):
    words = re.findall(r"\b\w+\b", text)
    if not words:
        return []

    original_score = get_fake_score(" ".join(words))
    importance = []

    for i in range(len(words)):
        new_text = " ".join(words[:i] + words[i+1:])
        new_score = get_fake_score(new_text)
        importance.append((words[i], original_score - new_score))

    ranked = sorted(importance, key=lambda x: x[1], reverse=True)

    stop_words = {
        "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
        "to", "of", "in", "on", "for", "with", "at", "by", "this", "that",
        "it", "as", "be", "from", "all", "who", "you", "your", "their",
        "they", "them", "we", "our", "his", "her", "he", "she", "no"
    }

    filtered = [
        (w, s) for w, s in ranked
        if w.lower() not in stop_words and len(w) > 2 and s > 0
    ]

    return filtered[:top_k]

def explain_words(words):
    explanations = []

    for word, score in words:
        w = word.lower()

        if w in ["breaking", "shocking", "unbelievable", "exclusive", "urgent"]:
            reason = "Sensational wording, often associated with exaggerated or misleading content."
        elif w in ["washington", "reuters", "senator", "government", "official", "court"]:
            reason = "Reference to a formal entity or institution, commonly seen in structured reporting."
        elif w in ["thank", "god", "finally", "unfair", "terrible", "amazing"]:
            reason = "Emotional or opinion-driven language, less common in neutral reporting."
        elif w in ["secret", "hidden", "truth", "exposed", "agenda", "conspiracy"]:
            reason = "Conspiracy-style or dramatic framing that can make text appear suspicious."
        elif w.istitle():
            reason = "Named entity or proper noun, which may reflect source- or topic-specific cues."
        else:
            reason = "This word influenced the model based on patterns learned during training."

        explanations.append((word, round(score, 6), reason))

    return explanations

def overall_explanation(label):
    if label == "FAKE":
        return "The model predicts this news as FAKE because the text appears to contain more emotional, dramatic, or informal language patterns."
    return "The model predicts this news as REAL because the text appears more structured, factual, and similar to formal reporting."

with st.sidebar:
    st.markdown("## 🧪 Try an Example")
    example_1 = "WASHINGTON  - The U.S. Senate approved a new infrastructure bill on Tuesday aimed at improving roads, bridges, and public transportation systems across the country."
    example_2 = "Breaking: Scientists have discovered a miracle cure that can instantly reverse aging, but governments are hiding it from the public."
    example_3 = "Thank God they finally caught him. People were so scared and nobody felt safe. This whole situation was insane and unbelievable."

    if st.button("Load Real-style Example"):
        st.session_state["input_text"] = example_1
    if st.button("Load Fake-style Example"):
        st.session_state["input_text"] = example_2
    if st.button("Load Emotional Example"):
        st.session_state["input_text"] = example_3

st.markdown('<div class="title-text">📰 Fake News Detection with Explainability</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle-text">Analyze a news passage, predict whether it looks real or fake, and understand which words influenced the decision.</div>',
    unsafe_allow_html=True
)

text = st.text_area(
    "Enter news text",
    value=st.session_state.get("input_text", ""),
    height=220,
    placeholder="Paste a news article, headline, or short passage here..."
)

col_btn1, col_btn2, _ = st.columns([1, 1, 4])
with col_btn1:
    predict_clicked = st.button("🔍 Predict", use_container_width=True)
with col_btn2:
    clear_clicked = st.button("🧹 Clear", use_container_width=True)

if clear_clicked:
    st.session_state["input_text"] = ""
    st.rerun()

if predict_clicked:
    if not text.strip():
        st.warning("Please enter some news text first.")
    else:
        label, confidence = predict_news(text)
        important_words = perturbation_explanation(text, top_k=8)
        explained_words = explain_words(important_words)

        if confidence < 0.6:
            st.warning("⚠️ Low confidence prediction. The model is uncertain about this input.")

        st.markdown("### Results")

        c1, c2 = st.columns(2)

        with c1:
            if label == "FAKE":
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Prediction</div>
                        <div class="metric-value">🚨 {label}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Prediction</div>
                        <div class="metric-value">✅ {label}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with c2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value">{confidence:.4f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        if label == "FAKE":
            st.markdown(
                f'<div class="fake-box"><b>Overall Explanation:</b> {overall_explanation(label)}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="real-box"><b>Overall Explanation:</b> {overall_explanation(label)}</div>',
                unsafe_allow_html=True
            )

        st.markdown("### Important Words and Reasons")

        if explained_words:
            explanation_df = pd.DataFrame(
                explained_words,
                columns=["Word", "Influence Score", "Why It Matters"]
            )
            st.dataframe(explanation_df, use_container_width=True, hide_index=True)
        else:
            st.info("No meaningful explanation words were extracted from this text.")

        st.markdown("### Input Preview")
        st.markdown(f'<div class="card">{text[:1200]}</div>', unsafe_allow_html=True)

        st.markdown(
            '<div class="small-note">Note: This system identifies patterns learned from the dataset. High confidence does not guarantee factual truth.</div>',
            unsafe_allow_html=True
        )
