# ─────────────────────────────────────────────────────────────────────────────
# modules/speech_handler.py
#
# PURPOSE:
#   Handle everything audio-related:
#     • Record microphone input          (PyAudio + SpeechRecognition)
#     • Convert speech → text (STT)     (SpeechRecognition / Google API)
#     • Convert text → speech (TTS)     (pyttsx3 offline OR gTTS online)
#     • Analyse audio quality            (librosa)
#
# KEY CONCEPTS COVERED:
#   • Context managers (with statement)
#   • Threading for non-blocking audio
#   • File I/O
#   • try/except/finally for resource cleanup
# ─────────────────────────────────────────────────────────────────────────────

import io                    # Built-in: in-memory file-like objects
import os
import time
import tempfile              # Built-in: create temporary files safely
import threading             # Built-in: run tasks concurrently
import wave                  # Built-in: read/write .wav files
from typing import Optional, Callable

import numpy as np           # Numerical operations on audio arrays
import pyaudio               # Low-level audio I/O (C-extension)
import speech_recognition as sr  # High-level STT wrapper
import pyttsx3               # Offline TTS engine
from gtts import gTTS        # Google Text-to-Speech (online)
import librosa               # Audio analysis (mel spectrograms, feature extraction)
import soundfile as sf       # Read/write audio files (better format support than wave)

import config


# ─────────────────────────────────────────────────────────────────────────────
# CLASS: SpeechHandler
# ─────────────────────────────────────────────────────────────────────────────
class SpeechHandler:
    """
    Manages all speech I/O operations for the interview bot.

    Attributes are set in __init__; methods are organised as:
      record_audio()      → start/stop microphone recording
      transcribe_audio()  → convert audio file → text
      speak_text()        → convert text → spoken audio
      analyse_audio()     → librosa-based quality metrics
    """

    def __init__(self) -> None:
        # ── PyAudio setup ─────────────────────────────────────────────────────
        # PyAudio is a Python binding for PortAudio (cross-platform audio I/O).
        # We create ONE PyAudio instance and reuse it for all recordings.
        self._pa = pyaudio.PyAudio()

        # ── SpeechRecognition setup ───────────────────────────────────────────
        # sr.Recognizer() is the main class.  It can use many backends
        # (Google Web API, Sphinx offline, Whisper, etc.).
        self._recognizer = sr.Recognizer()
        self._recognizer.energy_threshold = 300   # Microphone sensitivity
        self._recognizer.dynamic_energy_threshold = True
        # dynamic_energy_threshold=True lets the recognizer auto-adjust
        # to ambient noise levels over time.

        # ── pyttsx3 setup (offline TTS) ───────────────────────────────────────
        # pyttsx3 uses the OS's built-in TTS engine:
        #   Windows → SAPI5,  macOS → NSSpeechSynthesizer,  Linux → espeak
        try:
            self._tts_engine = pyttsx3.init()
            self._tts_engine.setProperty("rate",   170)   # Words per minute
            self._tts_engine.setProperty("volume", 0.9)   # 0.0 – 1.0
            self._tts_ready = True
        except Exception as e:
            print(f"[SpeechHandler] pyttsx3 init failed: {e}")
            self._tts_ready = False

        # ── Recording state ───────────────────────────────────────────────────
        self._recording: bool = False   # True while actively capturing audio
        self._audio_frames: list[bytes] = []  # Raw PCM frames collected during recording
        self._record_thread: Optional[threading.Thread] = None

    # ─────────────────────────────────────────────────────────────────────────
    # RECORDING
    # ─────────────────────────────────────────────────────────────────────────
    def start_recording(self) -> None:
        """
        Begin capturing audio from the default microphone in a background thread.

        Threading:
          Audio recording is blocking (it waits for each chunk of audio).
          If we ran it on the main thread, the Streamlit UI would freeze.
          threading.Thread lets us run _record_loop() concurrently.
          daemon=True means the thread will die automatically when the
          main program exits.
        """
        if self._recording:
            return  # Guard: don't start twice

        self._recording = True
        self._audio_frames = []   # Clear previous recording

        self._record_thread = threading.Thread(
            target=self._record_loop,
            daemon=True,
        )
        self._record_thread.start()

    def stop_recording(self) -> Optional[str]:
        """
        Stop recording and save the captured audio to a temporary .wav file.

        Returns the file path (str) so the caller can pass it to transcribe_audio().
        Returns None if no audio was captured.
        """
        self._recording = False

        # Wait for the recording thread to finish (join = wait)
        if self._record_thread and self._record_thread.is_alive():
            self._record_thread.join(timeout=3.0)
            # timeout=3.0 means we wait at most 3 seconds before giving up.

        if not self._audio_frames:
            return None

        # Save to a temporary file using the wave module
        return self._save_wav(self._audio_frames)

    def _record_loop(self) -> None:
        """
        Internal loop that reads audio chunks from the microphone.
        Runs in the background thread started by start_recording().
        """
        # pa.open() returns a Stream object – using `as` to name it `stream`
        stream = self._pa.open(
            format=pyaudio.paInt16,            # 16-bit PCM audio
            channels=config.AUDIO_CHANNELS,    # 1 = mono
            rate=config.SAMPLE_RATE,           # 16000 Hz
            input=True,                        # We're recording (input device)
            frames_per_buffer=config.CHUNK_SIZE,
        )

        try:
            while self._recording:
                # stream.read() blocks until CHUNK_SIZE frames are ready.
                # exception_on_overflow=False prevents crashes when the buffer
                # fills faster than we can read (common on slow machines).
                data: bytes = stream.read(config.CHUNK_SIZE, exception_on_overflow=False)
                self._audio_frames.append(data)

                # Auto-stop on silence detection
                if self._is_silence(data):
                    self._silence_frames = getattr(self, "_silence_frames", 0) + 1
                    silence_chunks_needed = int(
                        config.SILENCE_DURATION * config.SAMPLE_RATE / config.CHUNK_SIZE
                    )
                    if self._silence_frames >= silence_chunks_needed:
                        self._recording = False
                else:
                    self._silence_frames = 0
        finally:
            # The `finally` block ALWAYS runs – even if an exception occurred.
            # This guarantees we always close the audio stream (avoids resource leak).
            stream.stop_stream()
            stream.close()

    def _is_silence(self, data: bytes) -> bool:
        """
        Detect silence by computing RMS (Root Mean Square) amplitude.

        np.frombuffer() interprets raw bytes as a numpy array of int16 values.
        np.mean(arr**2) computes mean of squares.
        np.sqrt() of that is the RMS – a measure of signal power.
        """
        audio_array = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        if len(audio_array) == 0:
            return True
        rms = np.sqrt(np.mean(audio_array ** 2))
        # Normalise from int16 range (0–32767) to 0–1 range
        normalised_rms = rms / 32767.0
        return normalised_rms < config.SILENCE_THRESHOLD

    def _save_wav(self, frames: list[bytes]) -> str:
        """
        Write PCM frames to a .wav file using the `wave` module.

        tempfile.mktemp() generates a unique filename in the system's temp dir.
        We use our own dir (config.TEMP_AUDIO_DIR) for easier cleanup.
        """
        filename = os.path.join(
            config.TEMP_AUDIO_DIR,
            f"recording_{int(time.time())}.wav"
        )

        # `with` statement (context manager):
        # Automatically calls wf.close() when the block exits, even on error.
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(config.AUDIO_CHANNELS)
            wf.setsampwidth(self._pa.get_sample_size(pyaudio.paInt16))
            # get_sample_size(paInt16) returns 2 (bytes per 16-bit sample)
            wf.setframerate(config.SAMPLE_RATE)
            wf.writeframes(b"".join(frames))
            # b"".join(frames) concatenates a list of bytes objects into one.

        return filename

    # ─────────────────────────────────────────────────────────────────────────
    # SPEECH-TO-TEXT  (STT)
    # ─────────────────────────────────────────────────────────────────────────
    def transcribe_audio(self, audio_path: str) -> tuple[str, float]:
        """
        Convert a .wav file to text using SpeechRecognition.

        Uses Google's free Web Speech API (requires internet).
        Falls back to CMU Sphinx (offline) if Google fails.

        Returns
        -------
        (transcript_text, confidence_0_to_1)
        """
        try:
            with sr.AudioFile(audio_path) as source:
                # sr.AudioFile is a context manager that opens the .wav file
                # record() reads the entire audio file into an AudioData object
                audio_data: sr.AudioData = self._recognizer.record(source)

            # Try Google Web Speech API first (free, no key needed, internet required)
            try:
                result = self._recognizer.recognize_google(
                    audio_data,
                    language="en-US",
                    show_all=True,  # Return all alternatives, not just the top one
                )
                if result and "alternative" in result:
                    best = result["alternative"][0]
                    text = best.get("transcript", "")
                    confidence = best.get("confidence", 0.8)
                    return (text, confidence)
                return ("", 0.0)

            except sr.RequestError:
                # RequestError: network issue or API unavailable
                # Fall back to Sphinx (offline, less accurate)
                try:
                    text = self._recognizer.recognize_sphinx(audio_data)
                    return (text, 0.5)   # Lower confidence for offline engine
                except Exception:
                    return ("", 0.0)

        except Exception as e:
            print(f"[SpeechHandler] Transcription error: {e}")
            return ("", 0.0)

    # ─────────────────────────────────────────────────────────────────────────
    # TEXT-TO-SPEECH  (TTS)
    # ─────────────────────────────────────────────────────────────────────────
    def speak_text(self, text: str, use_gtts: bool = False) -> Optional[bytes]:
        """
        Convert text to spoken audio.

        Parameters
        ----------
        text      : The text to speak
        use_gtts  : If True, use gTTS (Google, online, more natural voice).
                    If False, use pyttsx3 (offline, robotic but always works).

        Returns
        -------
        bytes of MP3 audio data (for Streamlit's st.audio()) or None.
        """
        if use_gtts:
            return self._speak_gtts(text)
        else:
            return self._speak_pyttsx3(text)

    def _speak_gtts(self, text: str) -> Optional[bytes]:
        """
        Use gTTS to convert text to MP3 bytes.

        gTTS sends the text to Google's TTS servers and returns audio.
        io.BytesIO() is an in-memory file – no disk I/O needed.
        """
        try:
            tts = gTTS(text=text, lang="en", slow=False)
            # lang="en" → English, slow=False → normal speed

            audio_buffer = io.BytesIO()
            # BytesIO() creates a file-like object in RAM.
            # tts.write_to_fp() writes MP3 data into it.
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            # seek(0) rewinds to the start so the caller can read from position 0.
            return audio_buffer.read()
        except Exception as e:
            print(f"[SpeechHandler] gTTS error: {e}")
            return None

    def _speak_pyttsx3(self, text: str) -> Optional[bytes]:
        """
        Use pyttsx3 to synthesise speech, saving to a temp .wav file.
        Returns the file bytes so Streamlit can play them.
        """
        if not self._tts_ready:
            return None
        try:
            tmp_path = os.path.join(config.TEMP_AUDIO_DIR, "tts_output.wav")
            self._tts_engine.save_to_file(text, tmp_path)
            self._tts_engine.runAndWait()
            # runAndWait() blocks until all queued TTS tasks complete.

            with open(tmp_path, "rb") as f:
                return f.read()   # "rb" = read binary mode
        except Exception as e:
            print(f"[SpeechHandler] pyttsx3 error: {e}")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # AUDIO ANALYSIS  (librosa)
    # ─────────────────────────────────────────────────────────────────────────
    def analyse_audio_quality(self, audio_path: str) -> dict:
        """
        Use librosa to extract audio features that indicate speech quality.

        librosa is a Python library for music and audio analysis.
        It provides functions for feature extraction (MFCCs, pitch, energy etc.)

        Returns a dict with:
          speaking_rate   : estimated words per minute
          avg_pitch_hz    : average fundamental frequency (pitch)
          energy_variance : how much the volume varies (0=monotone)
          pause_ratio     : fraction of audio that is silence
        """
        try:
            # librosa.load() reads the audio file and returns:
            #   y  : numpy array of audio samples (float32, range [-1, 1])
            #   sr : sample rate (we force it to config.SAMPLE_RATE)
            y, sr_loaded = librosa.load(audio_path, sr=config.SAMPLE_RATE)

            # ── Duration ──────────────────────────────────────────────────────
            duration_sec = librosa.get_duration(y=y, sr=sr_loaded)

            # ── Pitch (F0) estimation using librosa's pyin algorithm ──────────
            # pyin returns:
            #   f0          : array of pitch values (Hz)  – NaN where unvoiced
            #   voiced_flag : bool array  – True where voiced
            #   voiced_prob : probability of voicing
            f0, voiced_flag, _ = librosa.pyin(
                y,
                fmin=librosa.note_to_hz("C2"),  # Minimum pitch (≈65 Hz)
                fmax=librosa.note_to_hz("C7"),  # Maximum pitch (≈2093 Hz)
            )
            voiced_f0 = f0[voiced_flag]  # Keep only voiced (non-NaN) values
            avg_pitch = float(np.nanmean(voiced_f0)) if len(voiced_f0) > 0 else 0.0

            # ── RMS Energy ────────────────────────────────────────────────────
            # RMS energy per frame – shows how loud each moment is
            rms = librosa.feature.rms(y=y)[0]   # Shape: (n_frames,)
            energy_variance = float(np.std(rms)) # Std deviation = how much it varies

            # ── Pause detection ───────────────────────────────────────────────
            # Frames below a threshold are "silent" (pauses)
            silence_mask = rms < config.SILENCE_THRESHOLD
            pause_ratio  = float(np.mean(silence_mask))

            # ── Approximate speaking rate ──────────────────────────────────────
            # This is a rough estimate: voiced regions / (avg word length in seconds)
            voiced_duration = duration_sec * (1 - pause_ratio)
            speaking_rate   = (voiced_duration / 0.3) / (duration_sec / 60)
            # 0.3 sec ≈ average duration of one syllable

            return {
                "duration_sec":     round(duration_sec, 1),
                "avg_pitch_hz":     round(avg_pitch, 1),
                "energy_variance":  round(energy_variance, 4),
                "pause_ratio":      round(pause_ratio, 3),
                "speaking_rate_wpm": round(speaking_rate, 0),
            }
        except Exception as e:
            print(f"[SpeechHandler] Audio analysis error: {e}")
            return {}

    # ─────────────────────────────────────────────────────────────────────────
    # CLEANUP
    # ─────────────────────────────────────────────────────────────────────────
    def cleanup(self) -> None:
        """Release PyAudio resources. Call this when the app shuts down."""
        try:
            self._pa.terminate()
        except Exception:
            pass
