import streamlit as st
import cv2
import whisper
from deepface import DeepFace
from transformers import pipeline
import spacy
import librosa
from collections import Counter
import tempfile
import subprocess
import os
@st.cache_resource
def load_whisper():
    return whisper.load_model("tiny")

@st.cache_resource
def load_sentiment():
    return pipeline("sentiment-analysis")

@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")
st.set_page_config(page_title="Interview Assessment System", page_icon="🎤", layout="centered")

st.title("🎤 Multimodal Interview Assessment System")
st.caption("Upload an interview video to get an AI-powered scorecard: facial emotion, speech quality, content, and pacing.")

uploaded_video = st.file_uploader("Upload your interview video", type=["mp4"])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_video.read())
    video_path = tfile.name
    st.video(uploaded_video)

    if st.button("Analyze Interview"):
        with st.spinner("Analyzing... this can take a minute or two on first run"):

            # --- Extract audio from video ---
            audio_path = video_path.replace(".mp4", ".mp3")
            subprocess.run(
                ["ffmpeg", "-y", "-i", video_path, "-q:a", "0", "-map", "a", audio_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

            # --- Module 2: Speech-to-text ---
            whisper_model = load_whisper()
            transcript = whisper_model.transcribe(audio_path)["text"].strip()

            # --- Module 3: Text analysis ---
            classifier = load_sentiment()
            sentiment = classifier(transcript)

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

            # --- Module 1: Facial emotion ---
            cap = cv2.VideoCapture(video_path)
            frames, i = [], 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if i % 10 == 0:
                    frames.append(frame)
                i += 1
            cap.release()

            emotions_list = []
            for frame in frames:
                try:
                    r = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
                    emotions_list.append(r[0]["dominant_emotion"])
                except Exception:
                    pass

            emotion_counts = Counter(emotions_list)
            total_frames = len(emotions_list) if emotions_list else 1

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
