# Models

All **102** BuzzASR models are released on Hugging Face under the
[**BuzzASR** organization](https://huggingface.co/BuzzASR) — one model per language,
MIT-licensed. Each is a fine-tune of
[openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3), shipped as fp16
safetensors, so it drops into any standard Whisper pipeline.

## Where to find them

- **Organization:** <https://huggingface.co/BuzzASR>
- Each language has its own repo, named by language, e.g.
  [`BuzzASR/finnish`](https://huggingface.co/BuzzASR/finnish),
  [`BuzzASR/mongolian`](https://huggingface.co/BuzzASR/mongolian).
- We release the **best recipe per language** (Simple/SFT or Full/FFT), chosen on validation —
  never on the test set. The table below lists which recipe ships for each language.

## Using a model

```python
import torch, torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor

name  = "BuzzASR/finnish"                        # any language repo under the org
model = WhisperForConditionalGeneration.from_pretrained(name, torch_dtype=torch.float16).to("cuda").eval()
proc  = WhisperProcessor.from_pretrained(name)

wav, sr = torchaudio.load("audio.wav")           # 16 kHz mono
feats = proc(wav[0], sampling_rate=16000, return_tensors="pt").input_features.to("cuda").half()
ids = model.generate(feats, num_beams=1, no_repeat_ngram_size=3, repetition_penalty=1.2)
print(proc.batch_decode(ids, skip_special_tokens=True)[0])
```

The language/task prompt is baked into each model's generation config, so no `language=` argument
is needed.

## Available models

🏆 marks the 27 languages where the model reaches the lowest CER of every open system we compare
against (Whisper-large-v3, Omnilingual 1B/7B, MMS, Qwen3-ASR, Cohere) on the combined
FLEURS + Common Voice test set.

| Language | Recipe | Combined CER (%) | Model |
|---|---|---|---|
| Afrikaans | Simple (SFT) | 5.35 | [BuzzASR/afrikaans](https://huggingface.co/BuzzASR/afrikaans) |
| Amharic | Full (FFT) | 9.07 | [BuzzASR/amharic](https://huggingface.co/BuzzASR/amharic) |
| Arabic | Simple (SFT) | 11.74 | [BuzzASR/arabic](https://huggingface.co/BuzzASR/arabic) |
| Armenian 🏆 | Full (FFT) | 2.28 | [BuzzASR/armenian](https://huggingface.co/BuzzASR/armenian) |
| Assamese | Full (FFT) | 11.83 | [BuzzASR/assamese](https://huggingface.co/BuzzASR/assamese) |
| Asturian 🏆 | Simple (SFT) | 4.89 | [BuzzASR/asturian](https://huggingface.co/BuzzASR/asturian) |
| Azerbaijani | Simple (SFT) | 5.22 | [BuzzASR/azerbaijani](https://huggingface.co/BuzzASR/azerbaijani) |
| Belarusian | Simple (SFT) | 2.75 | [BuzzASR/belarusian](https://huggingface.co/BuzzASR/belarusian) |
| Bengali | Full (FFT) | 7.79 | [BuzzASR/bengali](https://huggingface.co/BuzzASR/bengali) |
| Bosnian | Simple (SFT) | 3.59 | [BuzzASR/bosnian](https://huggingface.co/BuzzASR/bosnian) |
| Bulgarian 🏆 | Simple (SFT) | 1.21 | [BuzzASR/bulgarian](https://huggingface.co/BuzzASR/bulgarian) |
| Burmese | Full (FFT) | 24.01 | [BuzzASR/burmese](https://huggingface.co/BuzzASR/burmese) |
| Cantonese 🏆 | Simple (SFT) | 13.01 | [BuzzASR/cantonese](https://huggingface.co/BuzzASR/cantonese) |
| Catalan | Full (FFT) | 4.21 | [BuzzASR/catalan](https://huggingface.co/BuzzASR/catalan) |
| Cebuano | Simple (SFT) | 5.91 | [BuzzASR/cebuano](https://huggingface.co/BuzzASR/cebuano) |
| Croatian 🏆 | Simple (SFT) | 3.09 | [BuzzASR/croatian](https://huggingface.co/BuzzASR/croatian) |
| Czech | Simple (SFT) | 3.41 | [BuzzASR/czech](https://huggingface.co/BuzzASR/czech) |
| Danish | Full (FFT) | 14.90 | [BuzzASR/danish](https://huggingface.co/BuzzASR/danish) |
| Dutch 🏆 | Simple (SFT) | 1.49 | [BuzzASR/dutch](https://huggingface.co/BuzzASR/dutch) |
| English | Full (FFT) | 4.39 | [BuzzASR/english](https://huggingface.co/BuzzASR/english) |
| Estonian 🏆 | Full (FFT) | 1.40 | [BuzzASR/estonian](https://huggingface.co/BuzzASR/estonian) |
| Filipino | Simple (SFT) | 5.04 | [BuzzASR/filipino](https://huggingface.co/BuzzASR/filipino) |
| Finnish | Full (FFT) | 28.96 | [BuzzASR/finnish](https://huggingface.co/BuzzASR/finnish) |
| French | Full (FFT) | 6.68 | [BuzzASR/french](https://huggingface.co/BuzzASR/french) |
| Fulah | Full (FFT) | 16.04 | [BuzzASR/fulah](https://huggingface.co/BuzzASR/fulah) |
| Galician 🏆 | Simple (SFT) | 1.48 | [BuzzASR/galician](https://huggingface.co/BuzzASR/galician) |
| Georgian | Full (FFT) | 12.54 | [BuzzASR/georgian](https://huggingface.co/BuzzASR/georgian) |
| German | Simple (SFT) | 3.20 | [BuzzASR/german](https://huggingface.co/BuzzASR/german) |
| Greek | Simple (SFT) | 5.05 | [BuzzASR/greek](https://huggingface.co/BuzzASR/greek) |
| Gujarati | Full (FFT) | 8.39 | [BuzzASR/gujarati](https://huggingface.co/BuzzASR/gujarati) |
| Hausa 🏆 | Simple (SFT) | 5.50 | [BuzzASR/hausa](https://huggingface.co/BuzzASR/hausa) |
| Hebrew 🏆 | Simple (SFT) | 10.89 | [BuzzASR/hebrew](https://huggingface.co/BuzzASR/hebrew) |
| Hindi | Full (FFT) | 14.32 | [BuzzASR/hindi](https://huggingface.co/BuzzASR/hindi) |
| Hungarian | Simple (SFT) | 2.68 | [BuzzASR/hungarian](https://huggingface.co/BuzzASR/hungarian) |
| Icelandic | Full (FFT) | 7.45 | [BuzzASR/icelandic](https://huggingface.co/BuzzASR/icelandic) |
| Igbo | Full (FFT) | 13.97 | [BuzzASR/igbo](https://huggingface.co/BuzzASR/igbo) |
| Indonesian | Simple (SFT) | 2.63 | [BuzzASR/indonesian](https://huggingface.co/BuzzASR/indonesian) |
| Irish 🏆 | Simple (SFT) | 15.70 | [BuzzASR/irish](https://huggingface.co/BuzzASR/irish) |
| Italian | Simple (SFT) | 2.86 | [BuzzASR/italian](https://huggingface.co/BuzzASR/italian) |
| Japanese 🏆 | Full (FFT) | 21.08 | [BuzzASR/japanese](https://huggingface.co/BuzzASR/japanese) |
| Javanese | Simple (SFT) | 5.15 | [BuzzASR/javanese](https://huggingface.co/BuzzASR/javanese) |
| Kabuverdianu | Simple (SFT) | 5.03 | [BuzzASR/kabuverdianu](https://huggingface.co/BuzzASR/kabuverdianu) |
| Kamba | Simple (SFT) | 15.92 | [BuzzASR/kamba](https://huggingface.co/BuzzASR/kamba) |
| Kannada | Full (FFT) | 10.29 | [BuzzASR/kannada](https://huggingface.co/BuzzASR/kannada) |
| Kazakh | Full (FFT) | 7.47 | [BuzzASR/kazakh](https://huggingface.co/BuzzASR/kazakh) |
| Khmer | Full (FFT) | 25.99 | [BuzzASR/khmer](https://huggingface.co/BuzzASR/khmer) |
| Korean 🏆 | Simple (SFT) | 4.61 | [BuzzASR/korean](https://huggingface.co/BuzzASR/korean) |
| Kyrgyz | Simple (SFT) | 11.16 | [BuzzASR/kyrgyz](https://huggingface.co/BuzzASR/kyrgyz) |
| Lao | Full (FFT) | 20.95 | [BuzzASR/lao](https://huggingface.co/BuzzASR/lao) |
| Latvian | Simple (SFT) | 3.87 | [BuzzASR/latvian](https://huggingface.co/BuzzASR/latvian) |
| Lingala | Simple (SFT) | 7.42 | [BuzzASR/lingala](https://huggingface.co/BuzzASR/lingala) |
| Lithuanian 🏆 | Simple (SFT) | 2.11 | [BuzzASR/lithuanian](https://huggingface.co/BuzzASR/lithuanian) |
| Luganda | Simple (SFT) | 24.97 | [BuzzASR/luganda](https://huggingface.co/BuzzASR/luganda) |
| Luo | Simple (SFT) | 76.92 | [BuzzASR/luo](https://huggingface.co/BuzzASR/luo) |
| Luxembourgish | Full (FFT) | 8.55 | [BuzzASR/luxembourgish](https://huggingface.co/BuzzASR/luxembourgish) |
| Macedonian 🏆 | Simple (SFT) | 1.31 | [BuzzASR/macedonian](https://huggingface.co/BuzzASR/macedonian) |
| Malay | Simple (SFT) | 2.64 | [BuzzASR/malay](https://huggingface.co/BuzzASR/malay) |
| Malayalam | Full (FFT) | 15.38 | [BuzzASR/malayalam](https://huggingface.co/BuzzASR/malayalam) |
| Maltese 🏆 | Simple (SFT) | 2.66 | [BuzzASR/maltese](https://huggingface.co/BuzzASR/maltese) |
| Mandarin | Full (FFT) | 19.17 | [BuzzASR/mandarin](https://huggingface.co/BuzzASR/mandarin) |
| Maori | Full (FFT) | 10.06 | [BuzzASR/maori](https://huggingface.co/BuzzASR/maori) |
| Marathi | Simple (SFT) | 49.19 | [BuzzASR/marathi](https://huggingface.co/BuzzASR/marathi) |
| Mongolian 🏆 | Full (FFT) | 5.21 | [BuzzASR/mongolian](https://huggingface.co/BuzzASR/mongolian) |
| Nepali | Full (FFT) | 12.23 | [BuzzASR/nepali](https://huggingface.co/BuzzASR/nepali) |
| Northern Sotho | Full (FFT) | 11.79 | [BuzzASR/northern-sotho](https://huggingface.co/BuzzASR/northern-sotho) |
| Norwegian 🏆 | Simple (SFT) | 2.78 | [BuzzASR/norwegian](https://huggingface.co/BuzzASR/norwegian) |
| Nyanja | Full (FFT) | 10.99 | [BuzzASR/nyanja](https://huggingface.co/BuzzASR/nyanja) |
| Occitan 🏆 | Full (FFT) | 10.24 | [BuzzASR/occitan](https://huggingface.co/BuzzASR/occitan) |
| Oriya | Full (FFT) | 16.33 | [BuzzASR/oriya](https://huggingface.co/BuzzASR/oriya) |
| Oromo | Full (FFT) | 18.59 | [BuzzASR/oromo](https://huggingface.co/BuzzASR/oromo) |
| Pashto | Simple (SFT) | 11.10 | [BuzzASR/pashto](https://huggingface.co/BuzzASR/pashto) |
| Persian 🏆 | Full (FFT) | 4.71 | [BuzzASR/persian](https://huggingface.co/BuzzASR/persian) |
| Polish | Full (FFT) | 7.75 | [BuzzASR/polish](https://huggingface.co/BuzzASR/polish) |
| Portuguese | Simple (SFT) | 3.65 | [BuzzASR/portuguese](https://huggingface.co/BuzzASR/portuguese) |
| Punjabi 🏆 | Full (FFT) | 8.84 | [BuzzASR/punjabi](https://huggingface.co/BuzzASR/punjabi) |
| Romanian | Simple (SFT) | 3.12 | [BuzzASR/romanian](https://huggingface.co/BuzzASR/romanian) |
| Russian | Simple (SFT) | 2.97 | [BuzzASR/russian](https://huggingface.co/BuzzASR/russian) |
| Serbian | Full (FFT) | 60.85 | [BuzzASR/serbian](https://huggingface.co/BuzzASR/serbian) |
| Shona | Simple (SFT) | 8.42 | [BuzzASR/shona](https://huggingface.co/BuzzASR/shona) |
| Sindhi | Full (FFT) | 11.59 | [BuzzASR/sindhi](https://huggingface.co/BuzzASR/sindhi) |
| Slovak 🏆 | Full (FFT) | 2.19 | [BuzzASR/slovak](https://huggingface.co/BuzzASR/slovak) |
| Slovenian 🏆 | Simple (SFT) | 2.31 | [BuzzASR/slovenian](https://huggingface.co/BuzzASR/slovenian) |
| Somali | Full (FFT) | 18.24 | [BuzzASR/somali](https://huggingface.co/BuzzASR/somali) |
| Sorani Kurdish | Full (FFT) | 7.43 | [BuzzASR/sorani-kurdish](https://huggingface.co/BuzzASR/sorani-kurdish) |
| Spanish | Simple (SFT) | 1.95 | [BuzzASR/spanish](https://huggingface.co/BuzzASR/spanish) |
| Swahili | Full (FFT) | 12.69 | [BuzzASR/swahili](https://huggingface.co/BuzzASR/swahili) |
| Swedish 🏆 | Full (FFT) | 2.20 | [BuzzASR/swedish](https://huggingface.co/BuzzASR/swedish) |
| Tajik | Simple (SFT) | 5.61 | [BuzzASR/tajik](https://huggingface.co/BuzzASR/tajik) |
| Tamil | Simple (SFT) | 67.64 | [BuzzASR/tamil](https://huggingface.co/BuzzASR/tamil) |
| Telugu | Full (FFT) | 13.03 | [BuzzASR/telugu](https://huggingface.co/BuzzASR/telugu) |
| Thai | Simple (SFT) | 28.38 | [BuzzASR/thai](https://huggingface.co/BuzzASR/thai) |
| Turkish 🏆 | Simple (SFT) | 4.00 | [BuzzASR/turkish](https://huggingface.co/BuzzASR/turkish) |
| Ukrainian | Simple (SFT) | 3.44 | [BuzzASR/ukrainian](https://huggingface.co/BuzzASR/ukrainian) |
| Umbundu | Simple (SFT) | 19.91 | [BuzzASR/umbundu](https://huggingface.co/BuzzASR/umbundu) |
| Urdu | Simple (SFT) | 9.24 | [BuzzASR/urdu](https://huggingface.co/BuzzASR/urdu) |
| Uzbek 🏆 | Simple (SFT) | 3.86 | [BuzzASR/uzbek](https://huggingface.co/BuzzASR/uzbek) |
| Vietnamese | Simple (SFT) | 5.38 | [BuzzASR/vietnamese](https://huggingface.co/BuzzASR/vietnamese) |
| Welsh 🏆 | Full (FFT) | 5.24 | [BuzzASR/welsh](https://huggingface.co/BuzzASR/welsh) |
| Wolof | Full (FFT) | 15.67 | [BuzzASR/wolof](https://huggingface.co/BuzzASR/wolof) |
| Xhosa | Simple (SFT) | 11.48 | [BuzzASR/xhosa](https://huggingface.co/BuzzASR/xhosa) |
| Yoruba | Full (FFT) | 23.67 | [BuzzASR/yoruba](https://huggingface.co/BuzzASR/yoruba) |
| Zulu | Simple (SFT) | 11.86 | [BuzzASR/zulu](https://huggingface.co/BuzzASR/zulu) |
