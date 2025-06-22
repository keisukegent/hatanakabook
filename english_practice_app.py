import streamlit as st
import json
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import base64
import io
import tempfile
import speech_recognition as sr
from gtts import gTTS
import time
import random
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
from pydub import AudioSegment
import numpy as np
from difflib import SequenceMatcher

# アプリの設定
st.set_page_config(
    page_title="🗣️ English Practice App",
    page_icon="🗣️",
    layout="wide"
)

# データファイルのパス
DATA_FILE = "english_practice_data.json"

# フレーズデータベース
PHRASE_DATABASE = {
    "Shopping": [
        {"english": "How much does this cost?", "japanese": "これはいくらですか？", "difficulty": 1},
        {"english": "Can I get a discount?", "japanese": "割引はできますか？", "difficulty": 2},
        {"english": "Do you have this in a different size?", "japanese": "これの違うサイズはありますか？", "difficulty": 2},
        {"english": "Where is the fitting room?", "japanese": "試着室はどこですか？", "difficulty": 1},
        {"english": "I'll take this one, please.", "japanese": "これをください。", "difficulty": 1},
        {"english": "Could you wrap this as a gift?", "japanese": "これをギフト包装してもらえますか？", "difficulty": 3},
    ],
    "Restaurant": [
        {"english": "Could I see the menu, please?", "japanese": "メニューを見せてもらえますか？", "difficulty": 1},
        {"english": "I'd like to make a reservation.", "japanese": "予約をしたいのですが。", "difficulty": 2},
        {"english": "What do you recommend?", "japanese": "おすすめは何ですか？", "difficulty": 1},
        {"english": "I'm allergic to nuts.", "japanese": "ナッツアレルギーです。", "difficulty": 2},
        {"english": "Could we have the check, please?", "japanese": "お会計をお願いします。", "difficulty": 1},
        {"english": "The food was delicious!", "japanese": "料理がとても美味しかったです！", "difficulty": 2},
    ],
    "Hotel": [
        {"english": "I have a reservation under the name...", "japanese": "...の名前で予約しています。", "difficulty": 2},
        {"english": "What time is check-out?", "japanese": "チェックアウトは何時ですか？", "difficulty": 1},
        {"english": "Could you call a taxi for me?", "japanese": "タクシーを呼んでもらえますか？", "difficulty": 2},
        {"english": "Is breakfast included?", "japanese": "朝食は含まれていますか？", "difficulty": 1},
        {"english": "The air conditioning isn't working.", "japanese": "エアコンが動きません。", "difficulty": 2},
        {"english": "Could I get extra towels?", "japanese": "タオルを追加でもらえますか？", "difficulty": 2},
    ],
    "Transportation": [
        {"english": "Where is the nearest subway station?", "japanese": "最寄りの地下鉄駅はどこですか？", "difficulty": 1},
        {"english": "How much is a ticket to...?", "japanese": "...までのチケットはいくらですか？", "difficulty": 1},
        {"english": "What time does the last train leave?", "japanese": "最終電車は何時ですか？", "difficulty": 2},
        {"english": "I missed my flight.", "japanese": "飛行機に乗り遅れました。", "difficulty": 2},
        {"english": "Is this seat taken?", "japanese": "この席は空いていますか？", "difficulty": 1},
        {"english": "Could you tell me when we reach...?", "japanese": "...に着いたら教えてもらえますか？", "difficulty": 3},
    ]
}

# セッション状態の初期化
def initialize_session_state():
    if 'user_data' not in st.session_state:
        st.session_state.user_data = load_user_data()
    if 'current_phrase' not in st.session_state:
        st.session_state.current_phrase = None
    if 'current_scene' not in st.session_state:
        st.session_state.current_scene = "Shopping"
    if 'study_mode' not in st.session_state:
        st.session_state.study_mode = "Browse"

# ユーザーデータの読み込み
def load_user_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "progress": {},
        "study_history": [],
        "srs_schedule": {}
    }

# ユーザーデータの保存
def save_user_data():
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(st.session_state.user_data, f, ensure_ascii=False, indent=2)

# TTS音声生成
def generate_tts_audio(text, lang='en'):
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        
        # 一時ファイルに保存
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tts.save(tmp_file.name)
            
            # Base64エンコードしてHTML埋め込み用に変換
            with open(tmp_file.name, 'rb') as audio_file:
                audio_bytes = audio_file.read()
                audio_base64 = base64.b64encode(audio_bytes).decode()
            
            # 一時ファイルを削除
            os.unlink(tmp_file.name)
            
            return audio_base64
    except Exception as e:
        st.error(f"音声生成エラー: {e}")
        return None

# 音声再生HTML
def create_audio_html(audio_base64):
    return f"""
    <audio controls>
        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        Your browser does not support the audio element.
    </audio>
    """

# SRS（間隔反復学習）システム
def calculate_next_review(difficulty, success):
    base_intervals = [1, 3, 7, 14, 30, 90]  # 日数
    
    if success:
        if difficulty < len(base_intervals) - 1:
            difficulty += 1
    else:
        difficulty = max(0, difficulty - 1)
    
    next_review = datetime.now() + timedelta(days=base_intervals[min(difficulty, len(base_intervals) - 1)])
    return next_review.isoformat(), difficulty

# 復習が必要なフレーズを取得
def get_due_phrases():
    current_time = datetime.now().isoformat()
    due_phrases = []
    
    for phrase_id, schedule_info in st.session_state.user_data["srs_schedule"].items():
        if schedule_info["next_review"] <= current_time:
            due_phrases.append(phrase_id)
    
    return due_phrases

# 学習記録の追加
def add_study_record(phrase, scene, success, study_type):
    record = {
        "timestamp": datetime.now().isoformat(),
        "phrase": phrase["english"],
        "scene": scene,
        "success": success,
        "study_type": study_type
    }
    st.session_state.user_data["study_history"].append(record)
    
    # SRSスケジュール更新
    phrase_id = f"{scene}_{phrase['english']}"
    if phrase_id in st.session_state.user_data["srs_schedule"]:
        current_difficulty = st.session_state.user_data["srs_schedule"][phrase_id]["difficulty"]
    else:
        current_difficulty = 0
    
    next_review, new_difficulty = calculate_next_review(current_difficulty, success)
    st.session_state.user_data["srs_schedule"][phrase_id] = {
        "next_review": next_review,
        "difficulty": new_difficulty,
        "total_reviews": st.session_state.user_data["srs_schedule"].get(phrase_id, {}).get("total_reviews", 0) + 1
    }
    
    save_user_data()

# 進捗グラフの作成
def create_progress_chart():
    if not st.session_state.user_data["study_history"]:
        st.info("学習記録がありません。フレーズを練習してみてください！")
        return
    
    # データの準備
    history = st.session_state.user_data["study_history"]
    df = pd.DataFrame(history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    
    # 日別学習回数
    daily_counts = df.groupby('date').size().reset_index(name='count')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 日別学習回数グラフ
    ax1.plot(daily_counts['date'], daily_counts['count'], marker='o')
    ax1.set_title('Daily Study Count')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Number of Phrases Studied')
    ax1.tick_params(axis='x', rotation=45)
    
    # シーン別成功率
    scene_success = df.groupby('scene')['success'].mean().reset_index()
    ax2.bar(scene_success['scene'], scene_success['success'])
    ax2.set_title('Success Rate by Scene')
    ax2.set_xlabel('Scene')
    ax2.set_ylabel('Success Rate')
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    st.pyplot(fig)

# 音声認識用の設定
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# 音声処理クラス
class AudioProcessor:
    def __init__(self):
        self.audio_frames = []
        self.is_recording = False
        
    def recv(self, frame):
        if self.is_recording:
            sound = frame.to_ndarray()
            self.audio_frames.append(sound)
        return frame

# 音声認識と発音評価
def recognize_speech_and_evaluate(audio_data, target_phrase):
    try:
        # 音声認識
        recognizer = sr.Recognizer()
        
        # AudioSegmentからwav形式に変換
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_data))
        wav_data = io.BytesIO()
        audio_segment.export(wav_data, format="wav")
        wav_data.seek(0)
        
        # 音声認識実行
        with sr.AudioFile(wav_data) as source:
            audio = recognizer.record(source)
        
        recognized_text = recognizer.recognize_google(audio, language='en-US')
        
        # 発音評価（類似度計算）
        similarity = calculate_similarity(recognized_text.lower(), target_phrase.lower())
        
        return {
            "recognized_text": recognized_text,
            "similarity": similarity,
            "success": similarity > 0.7
        }
        
    except sr.UnknownValueError:
        return {
            "recognized_text": "音声を認識できませんでした",
            "similarity": 0.0,
            "success": False
        }
    except sr.RequestError as e:
        return {
            "recognized_text": f"音声認識サービスエラー: {e}",
            "similarity": 0.0,
            "success": False
        }
    except Exception as e:
        return {
            "recognized_text": f"エラーが発生しました: {e}",
            "similarity": 0.0,
            "success": False
        }

# テキスト類似度計算
def calculate_similarity(text1, text2):
    return SequenceMatcher(None, text1, text2).ratio()

# 音声録音とWebRTC
def show_voice_recording_ui(target_phrase):
    st.subheader("🎤 Voice Recording Practice")
    
    st.write(f"**Target phrase:** {target_phrase}")
    st.write("Click 'Start Recording' and speak the phrase clearly.")
    
    # WebRTCストリーマーの設定
    webrtc_ctx = webrtc_streamer(
        key="speech-recognition",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=256,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": False, "audio": True},
    )
    
    if webrtc_ctx.audio_receiver:
        st.write("🎤 Recording in progress...")
        
        # 録音データの処理
        if st.button("🛑 Stop Recording and Analyze"):
            audio_frames = []
            
            # WebRTCから音声フレームを取得
            while True:
                try:
                    audio_frames.append(webrtc_ctx.audio_receiver.get_frame(timeout=1))
                except:
                    break
            
            if audio_frames:
                # 音声データを結合
                sound_data = []
                for frame in audio_frames:
                    sound = frame.to_ndarray()
                    sound_data.append(sound)
                
                if sound_data:
                    # 音声データをバイト配列に変換
                    combined_audio = np.concatenate(sound_data)
                    
                    # AudioSegmentで処理
                    audio_segment = AudioSegment(
                        combined_audio.tobytes(),
                        frame_rate=frame.sample_rate,
                        sample_width=combined_audio.dtype.itemsize,
                        channels=1
                    )
                    
                    # 音声認識と評価
                    audio_buffer = io.BytesIO()
                    audio_segment.export(audio_buffer, format="wav")
                    audio_buffer.seek(0)
                    
                    result = recognize_speech_and_evaluate(audio_buffer.read(), target_phrase)
                    
                    # 結果表示
                    st.write("### 🎯 Recognition Results")
                    st.write(f"**Recognized:** {result['recognized_text']}")
                    st.write(f"**Similarity:** {result['similarity']:.2%}")
                    
                    if result['success']:
                        st.success("🎉 Great pronunciation! Well done!")
                        st.balloons()
                    else:
                        st.warning("🔄 Keep practicing! Try speaking more clearly.")
                    
                    return result
            else:
                st.warning("No audio data received. Please try recording again.")
    
    return None

# 簡単な録音機能（ファイルアップロード版）
def show_simple_recording_ui(target_phrase):
    st.subheader("🎤 Voice Practice (Upload Audio)")
    
    st.write(f"**Target phrase:** {target_phrase}")
    st.info("Record yourself saying the phrase and upload the audio file (WAV, MP3, M4A supported)")
    
    uploaded_file = st.file_uploader(
        "Upload your recording",
        type=['wav', 'mp3', 'm4a', 'ogg'],
        help="Record the phrase and upload your audio file"
    )
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        if st.button("🎯 Analyze Pronunciation"):
            with st.spinner("Analyzing your pronunciation..."):
                # 音声ファイルを読み込み
                audio_data = uploaded_file.read()
                
                # 音声認識と評価
                result = recognize_speech_and_evaluate(audio_data, target_phrase)
                
                # 結果表示
                st.write("### 🎯 Recognition Results")
                st.write(f"**Target:** {target_phrase}")
                st.write(f"**Recognized:** {result['recognized_text']}")
                st.write(f"**Similarity:** {result['similarity']:.2%}")
                
                # 発音スコア表示
                if result['similarity'] >= 0.9:
                    st.success("🏆 Excellent pronunciation! Perfect!")
                    st.balloons()
                    score = "Excellent"
                elif result['similarity'] >= 0.7:
                    st.success("🎉 Good pronunciation! Well done!")
                    score = "Good"
                elif result['similarity'] >= 0.5:
                    st.warning("🔄 Needs improvement. Keep practicing!")
                    score = "Needs Practice"
                else:
                    st.error("🔁 Try again. Speak more clearly.")
                    score = "Poor"
                
                # 発音練習記録
                return {
                    "recognized_text": result['recognized_text'],
                    "similarity": result['similarity'],
                    "success": result['success'],
                    "score": score
                }
    
    return None

# メイン画面
def main():
    st.title("🗣️ English Conversation Practice App")
    
    # サイドバー
    st.sidebar.title("📚 Navigation")
    
    # セッション状態でページを管理
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "🏠 Home"
    
    page = st.sidebar.selectbox("Choose a page:", 
                               ["🏠 Home", "📖 Study Phrases", "🔄 SRS Review", "📊 Progress", "⚙️ Settings"],
                               index=["🏠 Home", "📖 Study Phrases", "🔄 SRS Review", "📊 Progress", "⚙️ Settings"].index(st.session_state.current_page))
    
    # ページが変更された場合は更新
    if page != st.session_state.current_page:
        st.session_state.current_page = page
        st.rerun()
    
    if st.session_state.current_page == "🏠 Home":
        show_home()
    elif st.session_state.current_page == "📖 Study Phrases":
        show_study_phrases()
    elif st.session_state.current_page == "🔄 SRS Review":
        show_srs_review()
    elif st.session_state.current_page == "📊 Progress":
        show_progress()
    elif st.session_state.current_page == "⚙️ Settings":
        show_settings()

def show_home():
    st.header("Welcome to English Conversation Practice! 🎉")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Your Stats")
        total_studied = len(st.session_state.user_data["study_history"])
        due_count = len(get_due_phrases())
        st.metric("Total Phrases Studied", total_studied)
        st.metric("Due for Review", due_count)
    
    with col2:
        st.subheader("🎯 Quick Actions")
        
        # Start Studying ボタン
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            start_study = st.button("📖 Study", type="primary", use_container_width=True)
        with col_btn2:
            if due_count > 0:
                start_review = st.button(f"🔄 Review ({due_count})", type="secondary", use_container_width=True)
            else:
                start_review = False
        
        # ページ遷移処理
        if start_study:
            st.session_state.current_page = "📖 Study Phrases"
            st.rerun()
        
        if start_review:
            st.session_state.current_page = "🔄 SRS Review"
            st.rerun()
    
    st.subheader("📝 Available Scenes")
    scene_cols = st.columns(len(PHRASE_DATABASE))
    
    for i, (scene, phrases) in enumerate(PHRASE_DATABASE.items()):
        with scene_cols[i]:
            st.write(f"**{scene}**")
            st.write(f"{len(phrases)} phrases")

def show_study_phrases():
    st.header("📖 Study Phrases")
    
    # シーン選択
    scene = st.selectbox("Choose a scene:", list(PHRASE_DATABASE.keys()))
    phrases = PHRASE_DATABASE[scene]
    
    # フレーズ選択
    phrase_options = [f"{i+1}. {phrase['english']}" for i, phrase in enumerate(phrases)]
    selected_idx = st.selectbox("Choose a phrase:", range(len(phrase_options)), 
                               format_func=lambda x: phrase_options[x])
    
    current_phrase = phrases[selected_idx]
    
    # フレーズ表示
    st.subheader("📝 Current Phrase")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**English:**")
        st.write(f"*{current_phrase['english']}*")
        
        # 音声再生ボタン
        if st.button("🔊 Play Audio"):
            with st.spinner("Generating audio..."):
                audio_base64 = generate_tts_audio(current_phrase['english'])
                if audio_base64:
                    st.markdown(create_audio_html(audio_base64), unsafe_allow_html=True)
    
    with col2:
        show_japanese = st.checkbox("Show Japanese Translation")
        if show_japanese:
            st.write("**Japanese:**")
            st.write(current_phrase['japanese'])
    
    # 練習セクション
    st.subheader("🎯 Practice")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✅ I know this phrase", type="primary"):
            add_study_record(current_phrase, scene, True, "self_assessment")
            st.success("Great! Phrase marked as known.")
            st.balloons()
    
    with col2:
        if st.button("❌ I need more practice", type="secondary"):
            add_study_record(current_phrase, scene, False, "self_assessment")
            st.info("No worries! Keep practicing.")
    
    # 録音セクション
    st.subheader("🎤 Voice Practice")
    
    recording_method = st.radio(
        "Choose recording method:",
        ["Upload Audio File", "Real-time Recording (WebRTC)"],
        help="Upload: Record with your device and upload. WebRTC: Record directly in browser."
    )
    
    if recording_method == "Upload Audio File":
        recording_result = show_simple_recording_ui(current_phrase['english'])
    else:
        recording_result = show_voice_recording_ui(current_phrase['english'])
    
    # 録音結果を学習記録に追加
    if recording_result:
        add_study_record(current_phrase, scene, recording_result['success'], "pronunciation_practice")
        
        # 発音スコアの詳細表示
        st.subheader("📊 Pronunciation Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Similarity Score", f"{recording_result['similarity']:.1%}")
            st.metric("Pronunciation Grade", recording_result.get('score', 'N/A'))
        
        with col2:
            if 'recognized_text' in recording_result:
                st.write("**What you said:**")
                st.write(f"_{recording_result['recognized_text']}_")
                st.write("**Target phrase:**")
                st.write(f"_{current_phrase['english']}_")

def show_srs_review():
    st.header("🔄 SRS Review")
    
    due_phrases = get_due_phrases()
    
    if not due_phrases:
        st.success("🎉 No phrases due for review! Great job!")
        return
    
    st.write(f"**{len(due_phrases)} phrases** are due for review.")
    
    # ランダムに1つ選択
    if 'review_phrase_id' not in st.session_state:
        st.session_state.review_phrase_id = random.choice(due_phrases)
    
    phrase_id = st.session_state.review_phrase_id
    scene, phrase_text = phrase_id.split('_', 1)
    
    # フレーズを検索
    current_phrase = None
    for phrase in PHRASE_DATABASE[scene]:
        if phrase['english'] == phrase_text:
            current_phrase = phrase
            break
    
    if current_phrase:
        st.subheader("📝 Review This Phrase")
        
        # 英語を隠して日本語から練習
        st.write("**Japanese:**")
        st.write(f"*{current_phrase['japanese']}*")
        
        if st.button("🔍 Show English"):
            st.write("**English:**")
            st.write(f"*{current_phrase['english']}*")
            
            # 音声再生
            with st.spinner("Generating audio..."):
                audio_base64 = generate_tts_audio(current_phrase['english'])
                if audio_base64:
                    st.markdown(create_audio_html(audio_base64), unsafe_allow_html=True)
        
        # 評価ボタン
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Easy", type="primary"):
                add_study_record(current_phrase, scene, True, "srs_review")
                st.success("Excellent! Next review scheduled.")
                # 次のフレーズへ
                remaining_due = [p for p in due_phrases if p != phrase_id]
                if remaining_due:
                    st.session_state.review_phrase_id = random.choice(remaining_due)
                    st.rerun()
                else:
                    st.session_state.review_phrase_id = None
                    st.rerun()
        
        with col2:
            if st.button("❌ Hard", type="secondary"):
                add_study_record(current_phrase, scene, False, "srs_review")
                st.info("Review scheduled sooner. Keep practicing!")
                # 次のフレーズへ
                remaining_due = [p for p in due_phrases if p != phrase_id]
                if remaining_due:
                    st.session_state.review_phrase_id = random.choice(remaining_due)
                    st.rerun()
                else:
                    st.session_state.review_phrase_id = None
                    st.rerun()

def show_progress():
    st.header("📊 Your Progress")
    
    # 統計情報
    col1, col2, col3 = st.columns(3)
    
    total_studied = len(st.session_state.user_data["study_history"])
    if total_studied > 0:
        success_rate = sum(1 for record in st.session_state.user_data["study_history"] if record["success"]) / total_studied
        
        with col1:
            st.metric("Total Phrases Studied", total_studied)
        
        with col2:
            st.metric("Success Rate", f"{success_rate:.1%}")
        
        with col3:
            st.metric("Days Studied", len(set(record["timestamp"][:10] for record in st.session_state.user_data["study_history"])))
    
    # グラフ表示
    st.subheader("📈 Learning Charts")
    create_progress_chart()
    
    # 最近の学習記録
    st.subheader("📝 Recent Study History")
    if st.session_state.user_data["study_history"]:
        recent_records = st.session_state.user_data["study_history"][-10:]  # 最新10件
        for record in reversed(recent_records):
            status = "✅" if record["success"] else "❌"
            st.write(f"{status} **{record['scene']}**: {record['phrase']} _{record['timestamp'][:16]}_")
    else:
        st.info("No study records yet. Start practicing to see your progress!")

def show_settings():
    st.header("⚙️ Settings")
    
    st.subheader("📊 Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Export Data"):
            data_str = json.dumps(st.session_state.user_data, ensure_ascii=False, indent=2)
            st.download_button(
                label="Download Data",
                data=data_str,
                file_name="english_practice_data.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("🗑️ Clear All Data", type="secondary"):
            if st.checkbox("I confirm I want to delete all data"):
                st.session_state.user_data = {
                    "progress": {},
                    "study_history": [],
                    "srs_schedule": {}
                }
                save_user_data()
                st.success("All data cleared!")
                st.rerun()
    
    st.subheader("ℹ️ About")
    st.write("""
    **English Conversation Practice App**
    
    Features:
    - 📖 Scene-based phrase collections
    - 🔊 Native pronunciation with TTS
    - 🔄 Spaced Repetition System (SRS)
    - 📊 Progress tracking and charts
    - 🎤 Voice recognition support
    
    Created with Streamlit and Python.
    """)

# アプリケーション実行
if __name__ == "__main__":
    initialize_session_state()
    main()