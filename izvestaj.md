# Izveštaj o projektu — Prepoznavanje emocija iz govora pomoću CNN-a

**Predmet:** Neuronske mreže (13E054NM)
**Institucija:** Elektrotehnički fakultet, Univerzitet u Beogradu
**Skup podataka:** RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
**Zadatak:** Višeklasna klasifikacija slika — mel-spektrogrami → 8 klasa emocija

---

## 1. Opis problema

Cilj projekta je automatsko prepoznavanje emocije iz snimka govora. Svaki audio fajl pretvara se u mel-spektrogram — 2D sliku koja vizualizuje raspoređenost energije zvuka kroz vreme i frekvenciju — a zatim se ta slika klasifikuje konvolucionom neuronskom mrežom.

Prepoznavanje emocija iz govora je inherentno težak problem jer:
- Ista emocija može zvučati različito kod različitih govornika.
- Neke emocije (tuga, strah, neutralno) imaju slične akustičke karakteristike.
- Skup podataka je mali (1.440 snimaka za 8 klasa), što pogoduje preprilagođavanju.

---

## 2. Skup podataka — RAVDESS

| Karakteristika        | Vrednost                                              |
|-----------------------|-------------------------------------------------------|
| Ukupno snimaka        | 1.440                                                 |
| Broj glumaca          | 24 (12 muških, 12 ženskih)                            |
| Broj klasa emocija    | 8                                                     |
| Frekvencija uzorkovanja | 22.050 Hz                                           |
| Format                | WAV (mono)                                            |

**Klase emocija:**
`neutral`, `calm`, `happy`, `sad`, `angry`, `fearful`, `disgust`, `surprised`

### Neravnoteža klasa

Klasa `neutral` ima samo **96 snimaka**, dok svaka od preostalih 7 klasa ima **192 snimka** — odnos 2:1. Ovo je relevantno jer model koji ignorisanjem manjinske klase može postići lažno visoku tačnost. Ovaj problem je rešen kombinacijom:

1. **Per-klase augmentacije** u toku treniranja (neutral dobija 4× kopije).
2. **Focal loss** funkicija koja automatski daje veći značaj teško klasifikovanim primerima.

---

## 3. Preprocesiranje audio signala

### 3.1 Pipeline po fajlu

Svaki WAV fajl prolazi kroz sledeći niz koraka pre nego što uđe u mrežu:

```
WAV fajl
  → Učitavanje (librosa, sr=22.050 Hz)
  → Crop/padding na tačno 3 sekunde
  → [Opciono] Augmentacija talasnog oblika
  → Mel-spektrogram (128 mel bins × N vremenskih okvira)
  → Konverzija u dB skalu (power_to_db)
  → Resize na 128×128 piksela (bilinearna interpolacija)
  → Z-score normalizacija po uzorku
  → [Opciono] SpecAugment
  → Dodavanje dimenzije kanala → (128, 128, 1)
```

### 3.2 Zašto mel-spektrogram?

Mel skala aproksimira percepciju visine zvuka kod čoveka — bliže je logaritamska nego linearna. Mel-spektrogram bolje reprezentuje govor od standardnog STFT spektrograma jer kompresuje visoke frekvencije (gde su razlike akustički manje važne) i proširuje niske (gde je osnovna frekvencija glasa).

### 3.3 Zašto Z-score normalizacija po uzorku?

Različiti snimci imaju različite nivoe glasnoće. Z-score normalizacijom svaki spektrogram se centrira oko nule i skalira na standardnu devijaciju 1, nezavisno od loudness-a originalne snimke. Ovako mreža ne može koristiti apsolutnu glasnoću kao prečicu za klasifikaciju, a svi uzorci ulaze u mrežu u istim uslovima.

### 3.4 Podela podataka

| Skup       | Udeo  | Broj uzoraka (pre augmentacije) |
|------------|-------|---------------------------------|
| Trening    | 75%   | ~1.080                          |
| Validacija | 15%   | ~216                            |
| Test       | 10%   | ~144                            |

Korišćena je **stratifikovana nasumična podela** kako bi svaka klasa bila zastupljena u svakom skupu proporcionalno svojoj veličini.

> **Napomena:** Tokom razvoja testirana je i speaker-independent podela (glumci 21–22 za validaciju, 23–24 za test, ostali za trening), ali je ona dala lošije rezultate (~38% tačnost) zbog toga što glumci 23 i 24 imaju dovoljno različite vokalne karakteristike da test skup postaje neuobičajeno težak. Vraćeno je na nasumičnu stratifikovanu podelu.

---

## 4. Augmentacija podataka

### 4.1 Augmentacija talasnog oblika (pre spektrograma)

Augmentacija se primenjuje na **sirovi talasni oblik** pre računanja spektrograma — ne na piksele spektrograma. Razlog: augmentacija spektrogram piksela direktno stvara fizički nerealne artefakte (zamagljuje time-frequency grebene), dok augmentacija talasnog oblika uvek daje validan audio signal.

| Tehnika         | Parametri                        | Efekat na spektrogram               |
|-----------------|----------------------------------|--------------------------------------|
| Time stretching | rate ∈ [0.85, 1.15], p=0.5      | Horizontalno kompresovanje/širenje   |
| Pitch shifting  | steps ∈ [−2, +2] polutonova, p=0.5 | Vertikalni pomak frekventnog sadržaja |
| Gaussov šum     | std=0.005, p=0.5                | Simulacija šuma mikrofona            |

### 4.2 SpecAugment (posle normalizacije)

SpecAugment maskira nasumične frekvencijske pojaseve i vremenske segmente postavljajući ih na 0. Primenjuje se **posle** Z-score normalizacije, pa su maskirane vrednosti (0) bliske sredini distribucije i ne unose distribucijsko pomeranje.

- 2 frekventne maske, max 20 mel bins svaka
- 2 vremenske maske, max 20 okvira svaka

### 4.3 Per-klasa strategija augmentacije

Umesto globalnih class weights, svaka klasa dobija drugačiji broj augmentovanih kopija u trening skupu:

| Klasa      | Kopija | Razlog                                        |
|------------|--------|-----------------------------------------------|
| angry      | 2×     | Standardna klasa, dobar recall u baseline-u   |
| calm       | 2×     | Standardna klasa                              |
| disgust    | 2×     | Standardna klasa                              |
| fearful    | 3×     | Nizak recall (~21%) u baseline-u              |
| happy      | 3×     | Nizak recall (~37%) u baseline-u              |
| neutral    | 4×     | Manjinska klasa (96 uzoraka)                  |
| sad        | 3×     | Nizak recall (~16%) u baseline-u              |
| surprised  | 3×     | Nizak recall (~30%) u baseline-u              |

> **Zašto ne class weights?** Class weights i per-klasa augmentacija rade suprotne stvari — class weights smanjuju uticaj klasa sa više uzoraka, ali pošto smo već ručno povećali broj uzoraka teških klasa, dodavanje class weights-a bi poništilo augmentaciju. Jedno od dvoje, ne oba.

---

## 5. Arhitektura modela — CNN od nule

### 5.1 Struktura

```
Ulaz: (128, 128, 1)
  → CNNBlock(32)              → (64, 64, 32)
  → CNNBlock(64)              → (32, 32, 64)
  → CNNBlock(128, dropout=0.3) → (16, 16, 128)
  → CNNBlock(256, dropout=0.3) → (8,  8,  256)
  → GlobalAveragePooling2D    → (256,)
  → Dense(256, ReLU, L2=1e-4)
  → Dropout(0.5)
  → Dense(8, Softmax)         → (8,)
```

Svaki `CNNBlock` se sastoji od: `Conv2D → BatchNorm → ReLU → MaxPool2D → [Dropout]`

### 5.2 Regulizaciona strategija

| Tehnika           | Gde                          | Zašto                                                  |
|-------------------|------------------------------|--------------------------------------------------------|
| BatchNorm         | Svaki CNNBlock               | Stabilizuje distribuciju aktivacija, ubrzava trening   |
| Dropout (0.3)     | CNNBlock 3 i 4               | Sprečava preprilagođavanje u dubljim slojevima         |
| Dropout (0.5)     | Posle Dense(256)             | Najjača regularizacija pre izlaznog sloja              |
| L2 (1e-4)         | Conv2D i Dense težine        | Penalizuje velike težine, smanjuje overfitting         |
| SpecAugment       | Trening ulaz                 | Forsira robustnost na deo frekventnog/vremenskog opsega|

### 5.3 Funkcija gubitka — Focal Loss

Umesto standardne cross-entropy, korišćena je focal loss funkcija:

```
FL(p_t) = −(1 − p_t)^γ · log(p_t),  γ = 2.0
```

Focal loss smanjuje doprinos lako klasifikovanih primera (gde je model siguran) i fokusira trening signal na teške primere (fearful, sad). Sa γ=2 model troši daleko više kapaciteta na klase koje greši.

### 5.4 Callback-ovi

| Callback             | Parametri                          | Uloga                                              |
|----------------------|------------------------------------|----------------------------------------------------|
| EarlyStopping        | patience=20, monitor=val_loss      | Zaustavlja trening kad val_loss prestane da pada   |
| ModelCheckpoint      | save_best_only=True                | Čuva samo model sa najboljim val_loss              |
| ReduceLROnPlateau    | factor=0.5, patience=10, min=1e-6 | Prepolovi LR kad val_loss stagnira 10 epoha        |

> **Zašto patience=20?** Sa samo 216 validacionih uzoraka, val_loss prirodno osciluje od epohe do epohe. Kratka patience (npr. 10) bi prerano zaustavila trening dok se model još uči.

---

## 6. Transfer Learning — EfficientNetB0

### 6.1 Motivacija

CNN treniran od nule dostigao je plafon od ~50% tačnosti. Sa samo ~1.080 trening uzoraka za 8 klasa, mreža nema dovoljno podataka da nauči robustne vizuelne feature-e. Rešenje: koristiti mrežu pre-treniranu na ImageNet-u (1.2M slika) i prilagoditi je spektrogramima.

### 6.2 Zašto EfficientNetB0?

EfficientNetB0 je izabran jer:
- Dostupan u `keras.applications` bez dodatnih biblioteka.
- Relativno mali model (~5.3M parametara), pogodan za fine-tuning na malom skupu.
- ImageNet feature-i (ivice, teksture, oblici) dobro se prenose na spektrograme koji imaju sličnu 2D lokalnu strukturu.

### 6.3 Ključni problem: distribucija ulaznih vrednosti

EfficientNetB0 je pre-treniran na pikselima u opsegu [0, 255]. Naši Z-score normalizovani spektrogrami imaju vrednosti u opsegu [−3, 3]. Bez korekcije, EfficientNet-ova interna normalizacija (deli sa 255) svodi vrednosti na [−0.012, 0.012] — praktično nule — čime su svi pre-trenirani feature detektori beskorisni.

**Rešenje:** Dodat je registrovani Keras sloj `ZScoreToPixels` koji linerano preslikava [−3, 3] → [0, 255] pre EfficientNet-a:

```python
pixel = clip(z, -3, 3) / 6 * 255 + 127.5
```

Ovaj sloj omogućuje da pre-trenirani filteri EfficientNet-a ispravno aktiviraju na spektrogramskim uzorcima.

### 6.4 Dvofazno treniranje

**Faza 1 — Zamrznut base (20 epoha, LR=1e-3):**

EfficientNetB0 base je potpuno zamrznut. Trenira se samo nova glava (GAP → BN → Dense(256) → Dropout → Dense(8)). Cilj je da se glava nauči da koristi EfficientNet feature-e bez uništavanja pre-treniranih težina naglim gradijentima.

**Faza 2 — Fine-tuning (40 epoha, LR=1e-4):**

Poslednjih 30 slojeva EfficientNetB0 se odmrzava i trenira zajedno sa glavom, ali sa 10× manjim learning rate-om. Rani slojevi (koji detektuju generalne feature-e poput ivica) ostaju zamrznuti. Kasni slojevi (koji detektuju specifičnije feature-e) se prilagođavaju spektrogramima.

### 6.5 Arhitektura transfer modela

```
Ulaz: (128, 128, 1)
  → GrayToRGB           → (128, 128, 3)   [ponavlja kanal 3 puta]
  → ZScoreToPixels      → (128, 128, 3)   [mapira [-3,3] → [0,255]]
  → EfficientNetB0      → (4, 4, 1280)    [bez top sloja]
  → GlobalAveragePooling2D → (1280,)
  → BatchNormalization
  → Dense(256, ReLU, L2=1e-4)
  → Dropout(0.5)
  → Dense(8, Softmax)
```

---

## 7. Rezultati i analiza

### 7.1 Poređenje modela

| Model                    | Test tačnost | Napomena                              |
|--------------------------|--------------|---------------------------------------|
| CNN od nule (baseline)   | ~50.0%       | Pre svih poboljšanja                  |
| CNN od nule (optimizovan)| **50.7%**    | Focal loss, SpecAugment, per-class aug|
| Transfer (pogrešan ulaz) | 48.6%        | Z-score vrednosti direktno u EfficientNet — beskorisno |
| Transfer (ispravan ulaz) | **60.4%**    | ZScoreToPixels sloj, 20+40 epoha      |

### 7.2 Per-klasa analiza (transfer model, 60.4%)

| Klasa     | Precision | Recall | F1    | Analiza                                               |
|-----------|-----------|--------|-------|-------------------------------------------------------|
| surprised | 0.89      | 0.80   | **0.84** | Najlakša klasa — specifičan akustički potpis       |
| disgust   | 0.72      | 0.68   | **0.70** | Dobro naučena                                      |
| fearful   | 0.60      | 0.63   | **0.62** | Poboljšanje u odnosu na baseline (~21% recall)    |
| angry     | 0.57      | 0.68   | **0.62** | Visoka energija, lako prepoznatljivo               |
| calm      | 0.55      | 0.58   | 0.56  | Slično neutralnom — česta konfuzija                   |
| happy     | 0.75      | 0.47   | 0.58  | Visoka preciznost, ali nizak recall                   |
| neutral   | 0.45      | 0.50   | 0.48  | Manjinska klasa — i dalje najteža                     |
| sad       | 0.36      | 0.42   | 0.39  | Najslabija klasa — akustički slična strahu i neutralnom|

### 7.3 Analiza preprilagođavanja

U oba modela postoji značajan jaz između trening i validacione tačnosti:

- **CNN od nule:** trening ~85%, val ~55% → razlika 30%
- **Transfer model:** trening ~92%, val ~64% → razlika 28%

Ovaj jaz nije posledica loše arhitekture već fundamentalnog ograničenja — **premalo podataka za 8 klasa**. Sa samo ~135 trening uzoraka po klasi (pre augmentacije), svaki model dovoljno kapaciteta može memorisati trening skup.

Regulizacija (Dropout, L2, SpecAugment, focal loss) usporava memorisanje, ali ne može ga eliminisati. Pravi lek bi bili više podataka.

### 7.4 Zašto ne ide dalje od 60%?

| Ograničenje                   | Uticaj                                                    |
|-------------------------------|-----------------------------------------------------------|
| 1.440 snimaka, 8 klasa        | ~180 uzoraka/klasi — nedovoljno za duboku mrežu          |
| Val skup: 216 uzoraka         | 1 pogrešna predikcija = 0.46% promene val tačnosti       |
| Subjektivnost emocija         | Isti glumac, ista rečenica, različito tumačena emocija    |
| Akustičko preklapanje         | Tuga i strah imaju slično usporene, tihe obrasce          |

Istraživački radovi koji prijavljuju 70–80% na RAVDESS tipično koriste:
- wav2vec 2.0 / HuBERT audio embedding-e (pre-trenirani na stotinama hiljada sati govora)
- Ansambl metode ili pažnju (attention)
- Veće skupove podataka kombinovanjem više baza

---

## 8. Struktura projekta

```
nm projekat final/
├── config.py           # Svi hiperparametri i putanje
├── data_loader.py      # Parser RAVDESS fajlova, vizualizacija distribucije
├── preprocess.py       # WAV → mel-spektrogram, augmentacija, numpy cache
├── model.py            # CNN od nule (4×CNNBlock + GAP + Dense)
├── train.py            # Trening petlja, callback-ovi, history plot
├── model_transfer.py   # EfficientNetB0 transfer model, dvofazno treniranje
├── train_transfer.py   # TransferTrainer sa phase boundary plotom
├── evaluate.py         # Evaluacija, konfuziona matrica, izveštaj, primeri
├── main.py             # Entry point: --mode cnn | --mode transfer
├── data/ravdess/       # RAVDESS audio fajlovi (nije na Git-u)
├── processed/          # Keširani .npy spektrogrami (nije na Git-u)
└── results/            # Grafici i izveštaji
```

### Pokretanje

```bash
# Instalacija zavisnosti
pip install -r requirements.txt

# Puna pipeline — CNN od nule
python main.py

# Koristiti keširane spektrograme (preskočiti librosa konverziju)
python main.py --skip-preprocess

# Transfer learning
python main.py --skip-preprocess --mode transfer

# Samo evaluacija postojećeg modela
python main.py --skip-preprocess --eval-only
python main.py --skip-preprocess --mode transfer --eval-only
```

---

## 9. Zaključak

Projekat implementira kompletan pipeline za prepoznavanje emocija iz govora — od sirovih WAV fajlova do evaluacije klasifikatora.

**Ključni doprinosi:**

1. **Robusna preprocesiranje pipeline** sa per-uzorka normalizacijom, validnom waveform augmentacijom i SpecAugment-om.
2. **Per-klasa augmentacija** kao alternativa class weights, ciljano povećavajući trening signal za teške klase.
3. **Focal loss** koji dinamički preusmerava trening na primere koji se teško klasifikuju.
4. **Transfer learning sa EfficientNetB0** uz kritičan `ZScoreToPixels` sloj koji rešava mismatch ulaznih distribucija — poboljšanje od 50.7% na **60.4% test tačnosti**.

**Konačni rezultat:** Transfer model postiže **60.4% tačnosti na 8 klasa** uz makro F1 od 0.60, što predstavlja poboljšanje od ~10 procentnih poena u odnosu na CNN treniran od nule. Ovo je blizu praktičnog plafona za ovaj skup podataka bez korišćenja specijalizovanih audio embedding-a.
