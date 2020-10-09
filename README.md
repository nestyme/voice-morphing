# voice-morphing
voice morphing **currently** based on gender detector/age regressor &amp; spectral features modification [SFFT, Griphin-Lim argoritmn]\
According to https://www.researchgate.net/publication/269562531_The_Aging_Voice (and amount of another medical research papers), male and female voice changes with age in different ways.

| Articulation feature (AF)  | Male | Female |
| ------------- | ------------- | ------------- |
| pitch  | higher  | lower  |
| volume  | lower  | lower  |
| tremor | higher  | higher  |

1. Baseline
Train female/male voice classifier on TIMIT dataset and then manually change voice parametres, e.g. pitch/volume/tremor.
To analyze how to change pitch/volume/tremor, we need a speech factorizer. Age regressor and Gender Detector can help to annotate large amounts of data for future training. As a baseline, firstly I trained `gender classifier` and `age regressor` to choose voice morphing strategies. I tuned hyperpams for pitch shifter (SFFT+Griphin-Lim, in future can be replaced with vocoder e.g. WaveGlow) taking into consideration predicted age from `age_regressor`. \
2. Future experiments:\
  a. Replace Griphin-Lim with vocoder \
  b. Mine pseudolabeled data via trained detector and regressor on TIMIT (WSJ, LibriVox, Youtube) \
  b. http://www.apsipa.org/proceedings/2019/pdfs/68.pdf [VAE with speaker individuality block]\
  c. https://www.isca-speech.org/archive/Interspeech_2019/pdfs/1778.pdf [accent-changer PPG approach] \
  d. https://github.com/SerialLain3170/VoiceConversion [Cycle-GAN] \
## Current results
First results can be found in **test** folder in this repo. It contains female voice (recorded from macbook Pro microphone) and male voice (recorded as Telegram audiomessage + applied noise reduction filter). Then morphed files passed through age_regressor pipeline trying to guess age of fake-aged speaker.


| Articulation feature (AF)  | Male | Female |
| ------------- | ------------- | ------------- |
| original | 22 |  24 |  
| old | 46  | 51  |
| young | 18  | 17  |
