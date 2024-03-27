import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import pipeline

transcriber = pipeline(
  "automatic-speech-recognition",
  model="BELLE-2/Belle-whisper-large-v2-zh"
)

transcriber.model.config.forced_decoder_ids = (
  transcriber.tokenizer.get_decoder_prompt_ids(
    language="zh",
    task="transcribe"
  )
)

transcription = transcriber("split.mp3")