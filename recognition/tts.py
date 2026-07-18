import asyncio
import edge_tts
from playsound import playsound
import tempfile
import os
import threading

VOICE = "vi-VN-HoaiMyNeural"
# Muốn giọng nam thì dùng:
# VOICE = "vi-VN-NamMinhNeural"


class TextToSpeech:
    def __init__(self):
        self.lock = threading.Lock()

    async def _generate(self, text, filename):
        communicate = edge_tts.Communicate(text=text, voice=VOICE)
        await communicate.save(filename)

    def speak(self, text):
        with self.lock:
            filename = os.path.join(tempfile.gettempdir(), "speech.mp3")

            asyncio.run(self._generate(text, filename))

            playsound(filename)

            try:
                os.remove(filename)
            except PermissionError:
                pass

    def speak_async(self, text):
        threading.Thread(
            target=self.speak,
            args=(text,),
            daemon=True
        ).start()


tts = TextToSpeech()