# voice-morphing
voice morphing **currently** based on gender detector &amp; spectral features modification
According to https://www.researchgate.net/publication/269562531_The_Aging_Voice, male and female voice changes with age in different ways.

| Articulation feature (AF)  | Male | Female |
| ------------- | ------------- | ------------- |
| pitch  | higher  | lower  |
| volume  | lower  | lower  |
| tremor | higher  | higher  |

1. Baseline
Train female/male voice classifier on TIMIT dataset and then manually change voice parametres, e.g. pitch/volume/tremor.
To analyze how to change pitch/volume/tremor, we need a speech factorizer. Age regressor and Gender Detector can help to annotate large amounts of data for future training.
2. Future experiments:\
  a. http://www.apsipa.org/proceedings/2019/pdfs/68.pdf [VAE with speaker individuality block]\
  b. https://www.isca-speech.org/archive/Interspeech_2019/pdfs/1778.pdf [accent-changer PPG approach] \
  c. https://github.com/SerialLain3170/VoiceConversion [Cycle-GAN] \
## Current results
As a baseline, firstly I trained `gender classifier` and `age regressor` to choose voice morphing strategies. I tuned hyperpams for pitch shifter (SFFT+Griphin-Lim, in future can be replaced with vocoder e.g. WaveGlow) taking into consideration predicted age from `age_regressor`. \
First results can be found in **test** folder in this repo. It contains female voice (recorded from macbook Pro microphone) and male voice (recorded as Telegram audiomessage + applied noise reduction filter). Then morphed files passed through age_regressor pipeline trying to guess age of fake-aged speaker.


| Articulation feature (AF)  | Male | Female |
| ------------- | ------------- | ------------- |
| original | 22 |  24 |  
| old | 46  | 51  |
| young | 18  | 17  |
