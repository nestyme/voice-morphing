# voice-matureness
voice matureness changer based on gender detector &amp; spectral features modification
According to https://www.researchgate.net/publication/269562531_The_Aging_Voice, male and female voice changes with age in different ways.

| Articulation feature (AF)  | Male | Female |
| ------------- | ------------- | ------------- |
| pitch  | higher  | lower  |
| volume  | lower  | lower  |
| tremor | higher  | higher  |

1. Baseline
Train female/male voice classifier on TIMIT dataset and then manually change voice parametres, e.g. pitch/volume/tremor.
To analyze how to change pitch/volume/tremor, we need a speech factorizer.
2. Future experiments:\
  a. http://www.apsipa.org/proceedings/2019/pdfs/68.pdf [VAE with speaker individuality block]\
  b. https://www.isca-speech.org/archive/Interspeech_2019/pdfs/1778.pdf [accent-changer PPG approach] \
  c. https://github.com/SerialLain3170/VoiceConversion [Cycle-GAN] 
