# Multimodal Interview Assessment System
# Author: Shrishti
# GitHub: github.com/shristi146/interviewtest

import streamlit as st
import cv2
import whisper
from deepface import DeepFace
from textblob import TextBlob
import spacy
import librosa
from collections import Counter
import tempfile
import subprocess
import os
from fpdf import FPDF

# --- Cached model loaders ---
@st.cache_resource
def load_whisper():
    return whisper.load_model("tiny")

@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

# --- PDF Generator ---
def generate_pdf(final_score, emotion_score, speech_score, content_score,
                 transcript, sentiment, wpm, filler_result, keywords, emotion_counts, total_frames):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 15, "Interview Assessment Report", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, f"Overall Interview Score: {final_score:.1f}/100", ln=True)
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Score Breakdown:", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"  Emotion Score (30%):       {emotion_score:.1f}/100", ln=True)
    pdf.cell(0, 7, f"  Speech Quality (30%):      {speech_score:.1f}/100", ln=True)
    pdf.cell(0, 7, f"  Content Score (40%):       {content_score:.1f}/100", ln=True)
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Transcript:", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, transcript)
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Analysis Details:", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"  Sentiment: {sentiment[0]['label']} ({sentiment[0]['score']*100:.1f}%)", ln=True)
    pdf.cell(0, 7, f"  Speaking Speed: {wpm:.1f} WPM", ln=True)
    pdf.cell(0, 7, f"  Filler Words: {filler_result if filler_result else 'None detected'}", ln=True)
    pdf.cell(0, 7, f"  Keywords: {', '.join(keywords[:10])}", ln=True)
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Emotion Breakdown:", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for emo, cnt in emotion_counts.items():
        pct = (cnt / total_frames) * 100
        pdf.cell(0, 7, f"  {emo.capitalize()}: {pct:.1f}%", ln=True)
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Recommendations:", ln=True)
    pdf.set_font("Helvetica", "", 11)
    filler_penalty = sum(filler_result.values()) * 5
    pdf.cell(0, 7, "  Good filler word control" if filler_penalty <= 10 else "  Try reducing filler words", ln=True)
    pdf.cell(0, 7, "  Good speaking pace" if 110 <= wpm <= 160 else "  Adjust speaking pace (110-160 WPM)", ln=True)
    pdf.cell(0, 7, "  Positive facial expression overall" if emotion_score >= 60 else "  Try to appear more relaxed on camera", ln=True)

    return bytes(pdf.output())

# --- Lighting normalization (bias reduction) ---
def normalize_frame(frame):
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

# --- Page config ---
st.set_page_config(page_title="Interview Assessment System", page_icon="🎤", layout="centered")

st.title("🎤 Multimodal Interview Assessment System")
st.caption("Upload an interview video to get an AI-powered scorecard: facial emotion, speech quality, content, and pacing.")

# --- Bias disclosure ---
with st.expander("⚠️ Known Limitations & Bias Disclosure"):
    st.write("""
    - **Facial Emotion Bias:** The DeepFace model is trained on FER-2013 dataset, 
      which has known limitations under poor lighting, non-frontal angles, 
      and darker skin tones. Neutral expressions are sometimes misclassified as fear or sadness.
    - **Speech Bias:** Whisper performs best on clear English speech. 
      Accented speech or background noise may reduce transcription accuracy.
    - **Content Bias:** Keyword scoring favors technical vocabulary. 
      Non-technical but valid answers may score lower than they should.
    - **Sample Size:** Emotion scoring samples every 10th frame — 
      short videos may have limited frame coverage.
    
    This tool is intended as a supplementary feedback system, 
    not a definitive hiring decision tool.
    """)

uploaded_video = st.file_uploader("Upload your interview video", type=["mp4"])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_video.read())
    video_path = tfile.name
    st.video(uploaded_video)

    if st.button("Analyze Interview"):
        with st.spinner("Analyzing... this can take a minute or two on first run"):

            status = st.empty()

            status.info("🎵 Extracting audio...")

            # --- Extract audio from video ---
            audio_path = video_path.replace(".mp4", ".mp3")
            subprocess.run(
                ["ffmpeg", "-y", "-i", video_path, "-q:a", "0", "-map", "a", audio_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

            status.info("🗣️ Converting speech to text...")

            # --- Module 2: Speech-to-text ---
            whisper_model = load_whisper()
            transcript = whisper_model.transcribe(audio_path)["text"].strip()

            status.info("📝 Analyzing transcript...")

            # --- Module 3: Text analysis ---
            polarity = TextBlob(transcript).sentiment.polarity

            if polarity >= 0:
                sentiment = [{"label": "POSITIVE", "score": polarity}]
            else:
                sentiment = [{"label": "NEGATIVE", "score": abs(polarity)}]
            filler_words = ["um", "uh", "like", "you know", "actually", "basically", "literally"]
            transcript_lower = transcript.lower()
            filler_result = {w: transcript_lower.count(w) for w in filler_words if transcript_lower.count(w) > 0}

            nlp = load_spacy()
            doc = nlp(transcript)
            keywords = [t.text for t in doc if t.pos_ in ["NOUN", "PROPN"]]

            word_count = len(transcript.split())

            # --- Module 4: Voice analysis ---
            y, sr = librosa.load(audio_path)
            duration = librosa.get_duration(y=y, sr=sr)
            wpm = (word_count / duration) * 60 if duration > 0 else 0

            status.info("😊 Analyzing facial expressions...")

            # --- Module 1: Facial emotion (with bias reduction) ---

            cap = cv2.VideoCapture(video_path)
            emotions_list = []
            i = 0

            while True:
                ok, frame = cap.read()

                if not ok:
                    break

                if i % 60 == 0:  # analyze only every 60th frame
                    try:
                        frame = normalize_frame(frame)

                        r = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)

                        emotion_scores = r[0]["emotion"]
                        dominant = r[0]["dominant_emotion"]
                        confidence = emotion_scores[dominant]

                        if confidence > 60:
                            emotions_list.append(dominant)

                    except Exception:
                         pass

                    i += 1

                cap.release()
                
            emotion_counts = Counter(emotions_list)
            total_frames = len(emotions_list) if emotions_list else 1

            status.info("📊 Calculating final score...")

            # --- Module 5: Final weighted score ---
            positive_emotions = emotion_counts.get("happy", 0) + emotion_counts.get("neutral", 0)
            emotion_score = (positive_emotions / total_frames) * 100

            sentiment_score = (
                sentiment[0]["score"] * 100
                if sentiment[0]["label"] == "POSITIVE"
                else (1 - sentiment[0]["score"]) * 100
            )
            filler_penalty = sum(filler_result.values()) * 5
            speech_score = max(0, sentiment_score - filler_penalty)

            content_score = min(100, len(keywords) * 10)

            final_score = (emotion_score * 0.3) + (speech_score * 0.3) + (content_score * 0.4)

            # Clean up temp files
            try:
                os.remove(video_path)
                os.remove(audio_path)
            except Exception:
                pass

            status.success("✅ Analysis completed!")

        # ---------- DISPLAY RESULTS ----------
        st.success(f"### Overall Interview Score: {final_score:.1f}/100")

        c1, c2, c3 = st.columns(3)
        c1.metric("Emotion", f"{emotion_score:.1f}/100")
        c2.metric("Speech Quality", f"{speech_score:.1f}/100")
        c3.metric("Content", f"{content_score:.1f}/100")

        st.subheader("Transcript")
        st.write(transcript)

        st.subheader("Details")
        st.write(f"**Sentiment:** {sentiment[0]['label']} ({sentiment[0]['score']*100:.1f}%)")
        st.write(f"**Speaking Speed:** {wpm:.1f} WPM")
        st.write(f"**Filler Words:** {filler_result if filler_result else 'None detected'}")
        st.write(f"**Keywords Detected:** {keywords}")

        st.subheader("Emotion Breakdown")
        for emo, cnt in emotion_counts.items():
            pct = (cnt / total_frames) * 100
            st.write(f"- {emo.capitalize()}: {pct:.1f}%")

        st.subheader("Recommendations")
        if filler_penalty <= 10:
            st.write("✅ Good filler word control")
        else:
            st.write("⚠️ Try reducing filler words")

        if 110 <= wpm <= 160:
            st.write("✅ Good speaking pace")
        else:
            st.write("⚠️ Adjust speaking pace (aim for 110-160 WPM)")

        if emotion_score >= 60:
            st.write("✅ Positive/calm facial expression overall")
        else:
            st.write("⚠️ Try to appear more relaxed and confident on camera")

        # --- PDF Download ---
        st.divider()
        pdf_bytes = generate_pdf(
            final_score, emotion_score, speech_score, content_score,
            transcript, sentiment, wpm, filler_result, keywords, emotion_counts, total_frames
        )
        st.download_button(
            label="📄 Download Full Report (PDF)",
            data=pdf_bytes,
            file_name="interview_assessment_report.pdf",
            mime="application/pdf"
        )
