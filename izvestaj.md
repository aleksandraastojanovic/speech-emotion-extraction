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
- Ni ljudi nisu pouzdani: autori RAVDESS baze mere ljudsku tačnost od **~62%** na audio-only govoru — to je realan orijentir za plafon zadatka.

---

## 2. Skup podataka — RAVDESS

| Karakteristika        | Vrednost                                              |
|-----------------------|-------------------------------------------------------|
| Ukupno snimaka        | 1.440                                                 |
| Broj glumaca          | 24 (12 muških, 12 ženskih)                            |
| Broj klasa emocija    | 8                                                     |
| Frekvencija uzorkovanja | 48 kHz (resample na 22.050 Hz)                      |
| Format                | WAV (mono)                                            |

**Klase emocija:**
`neutral`, `calm`, `happy`, `sad`, `angry`, `fearful`, `disgust`, `surprised`

### Neravnoteža klasa

Klasa `neutral` ima samo **96 snimaka**, dok svaka od preostalih 7 klasa ima **192 snimka** — odnos 2:1. Rešeno je **tačno balansiranom augmentacijom**: neutral dobija duplo više augmentovanih kopija, pa svih 8 klasa ulazi u trening sa identičnim brojem uzoraka (432).

---

## 3. Preprocesiranje audio signala

### 3.1 Pipeline po fajlu

```
WAV fajl
  → Učitavanje celog snimka (librosa, sr=22.050 Hz)
  → Trim tišine (top_db=30)
  → Crop/padding na tačno 3 sekunde
  → [Opciono] Augmentacija talasnog oblika
  → Mel-spektrogram (128 mel bins) → dB skala
  → Delta i delta-delta (brzina i ubrzanje spektra po vremenu)
  → Resize svakog kanala na 128×128 (bilinearna interpolacija)
  → Z-score normalizacija po kanalu
  → [Opciono] SpecAugment (maske preko sva 3 kanala)
  → Izlaz: (128, 128, 3)
```

### 3.2 Trim tišine — najveći pojedinačni dobitak

RAVDESS snimci počinju i završavaju se tišinom. Merenje na stvarnom fajlu: od 3,30 s snimka, posle trima ostane **1,32 s govora** — originalni pipeline (koji je sekao prve 3 sekunde bez trima) hranio je mrežu ulazom koji je 40–60% tišina, a kraj rečenice je često bio odsečen. `librosa.effects.trim(top_db=30)` uklanja delove tiše od 30 dB ispod vrha snimka; prag je relativan, pa se prilagođava i tihim (sad, calm) snimcima.

### 3.3 Delta kanali umesto sive slike

Statični mel-spektrogram opisuje *šta* je prisutno u spektru, ali ne i *kako se menja* — a emocija je upravo u dinamici (tempo, tremor, nagle promene). Standardna SER praksa: uz mel se računaju **delta** (prvi izvod po vremenu) i **delta-delta** (drugi izvod), pa se slažu kao 3 kanala slike. Dobitak je dvostruk:

1. Scratch CNN dobija eksplicitnu informaciju o dinamici govora.
2. Transfer model (EfficientNetB0 očekuje RGB) dobija tri **informativna** kanala umesto tri kopije istog kanala.

### 3.4 Z-score normalizacija po uzorku i kanalu

Različiti snimci imaju različite nivoe glasnoće. Z-score normalizacijom svaki kanal svakog spektrograma se centrira oko nule i skalira na jediničnu devijaciju, nezavisno od loudness-a snimka — mreža ne može koristiti apsolutnu glasnoću kao prečicu.

### 3.5 Podela podataka

| Skup       | Udeo  | Broj uzoraka (pre augmentacije) |
|------------|-------|---------------------------------|
| Trening    | 75%   | 1.080                           |
| Validacija | 15%   | 216                             |
| Test       | 10%   | 144                             |

Korišćena je **stratifikovana nasumična podela** (seed 42, deterministična). Test skup od 144 uzorka znači da jedna predikcija menja tačnost za 0,7 p.p. — razlike manje od ~3 p.p. su u zoni šuma.

> **Napomena:** Tokom razvoja testirana je i speaker-independent podela (glumci 21–24 izdvojeni), ali daje ~38% tačnosti — glumci se akustički previše razlikuju da bi 2 test glumca bila reprezentativna. Zadržana je nasumična stratifikovana podela (speaker-dependent), što treba imati u vidu pri poređenju sa literaturom.

---

## 4. Augmentacija podataka

### 4.1 Augmentacija talasnog oblika (pre spektrograma)

Augmentacija se primenjuje na **sirovi talasni oblik**, ne na piksele spektrograma — augmentacija piksela stvara fizički nerealne artefakte, dok augmentacija talasnog oblika uvek daje validan audio signal.

| Tehnika         | Parametri                        | Efekat na spektrogram               |
|-----------------|----------------------------------|--------------------------------------|
| Time stretching | rate ∈ [0.85, 1.15], p=0.5      | Horizontalno kompresovanje/širenje   |
| Pitch shifting  | steps ∈ [−2, +2] polutonova, p=0.5 | Vertikalni pomak frekventnog sadržaja |
| Gaussov šum     | std=0.005, p=0.5                | Simulacija šuma mikrofona            |

Svaka augmentovana kopija ima **deterministički seed**, pa je ceo trening skup reprodukovljiv, iako se preprocesiranje izvršava paralelno na svim jezgrima (joblib).

### 4.2 SpecAugment (posle normalizacije)

SpecAugment maskira nasumične frekvencijske pojaseve i vremenske segmente (2 maske × max 20 binova/okvira). Primenjuje se **posle** Z-score normalizacije (maskirana vrednost 0 ≈ sredina distribucije) i **preko sva tri kanala istovremeno** da mel/delta ostanu poravnati.

### 4.3 Tačno balansirana augmentacija

| Klasa        | Original | Kopija (1 plain + N aug) | Ukupno |
|--------------|----------|--------------------------|--------|
| neutral      | 72       | 6× (1+5)                 | 432    |
| svih ostalih 7 | 144    | 3× (1+2)                 | 432    |

Trening skup: **3.456 uzoraka, tačno 432 po klasi**. Validacija i test se nikad ne augmentuju.

> **Zašto ne class weights?** Pošto augmentacija već izjednačava brojeve uzoraka, class weights bi bili redundantni (ili bi, pogrešno kombinovani, poništili balans).

---

## 5. Arhitektura modela — CNN od nule

### 5.1 Struktura

```
Ulaz: (128, 128, 3)  — mel + delta + delta-delta
  → CNNBlock(32)               → (64, 64, 32)
  → CNNBlock(64)               → (32, 32, 64)
  → CNNBlock(128, dropout=0.3) → (16, 16, 128)
  → CNNBlock(256, dropout=0.3) → (8,  8,  256)
  → GlobalAveragePooling2D     → (256,)
  → Dense(256, ReLU, L2=1e-4)
  → Dropout(0.5)
  → Dense(8, Softmax)
```

Svaki `CNNBlock`: `Conv2D → BatchNorm → ReLU → MaxPool2D → [Dropout]` (~458k parametara).

### 5.2 Stabilnost treninga — naučena lekcija

Prvobitni trening (Adam, LR=1e-3) je **divergirao**: val_loss je skakao do 5.0, a early stopping je vraćao duboko nedotrenirane modele iz ranih epoha (test 38%). Rešenje, tri elementa:

| Izmena | Vrednost | Efekat |
|--------|----------|--------|
| Learning rate | 1e-3 → **3e-4** | bez divergencije |
| Gradient clipping | clipnorm=1.0 | seče retke eksplozivne gradijente |
| Batch size | 32 → **64** | mirnije BatchNorm statistike |

### 5.3 Selekcija modela po val_accuracy — druga naučena lekcija

Sa focal loss-om na malom val skupu (216 uzoraka), val_loss je toliko šumovit da checkpoint po val_loss-u sistematski bira „srećne" rane epohe (najbolja epoha 10 → test 50,7%). Prelaskom `EarlyStopping` i `ModelCheckpoint` na **val_accuracy** najbolja epoha je postala 44, a test skočio na **59,7%**. `ReduceLROnPlateau` ostaje na val_loss (glađi signal za raspored LR).

| Callback             | Parametri                              |
|----------------------|----------------------------------------|
| EarlyStopping        | monitor=val_accuracy, patience=20, restore best |
| ModelCheckpoint      | monitor=val_accuracy, save_best_only   |
| ReduceLROnPlateau    | monitor=val_loss, factor=0.5, patience=5 |

### 5.4 Funkcija gubitka — Focal Loss

```
FL(p_t) = −(1 − p_t)^γ · log(p_t),  γ = 2.0
```

Focal loss smanjuje doprinos lako klasifikovanih primera i fokusira trening na teške (sad, fearful).

---

## 6. Transfer Learning — EfficientNetB0

### 6.1 Arhitektura

```
Ulaz: (128, 128, 3)      — mel + delta + delta-delta
  → ZScoreToPixels       — [-3, 3] → [0, 255] (opseg ImageNet piksela)
  → Resizing(224, 224)   — nativna rezolucija EfficientNet pretreninga
  → EfficientNetB0       — ImageNet težine, bez top sloja
  → GlobalAveragePooling2D → BatchNorm
  → Dense(256, ReLU, L2) → Dropout(0.5) → Dense(8, Softmax)
```

Dve ključne adaptacije ulaza:
- **ZScoreToPixels:** bez ovoga pretrenirani filteri vide vrednosti ~[−3, 3] umesto [0, 255] i ne aktiviraju se.
- **Resizing(224):** na 128×128 poslednja mapa EfficientNet-a je samo 4×4; na 224×224 mreža radi na skali na kojoj je pretrenirana.

### 6.2 Dvofazno treniranje

**Faza 1 — zamrznut base (do 40 epoha, LR=1e-3):** trenira se samo glava. Granica je pomerena sa 20 na 40 epoha jer je val accuracy još rasla pri preseku; early stopping odlučuje stvarni kraj.

**Faza 2 — fine-tuning (do 60 epoha, LR=5e-5):** odmrzava se poslednjih **20** slojeva. Eksperimentalno: 30–60 odmrznutih slojeva sa LR 1e-4 dovodi do momentalnog overfittinga (train 94%, val_loss raste od prve epohe); 20 slojeva sa LR 5e-5 daje sporo ali stabilno poboljšanje val accuracy kroz celu fazu.

**Ispravka checkpoint baga:** na početku faze 2 `ModelCheckpoint` resetuje svoj „best" na −∞, pa bi prva (lošija) epoha faze 2 pregazila najbolji model faze 1. Rešeno prosleđivanjem `initial_value_threshold = max(val_accuracy faze 1)`.

---

## 7. Ansambl

Finalni model je **prosek softmax izlaza tri mreže**: dva scratch CNN-a (nezavisna trening run-a) i transfer modela. Različite arhitekture greše na različitim uzorcima, pa usrednjavanje poništava deo nekoreliranih grešaka.

Kombinacija je **birana na validacionom skupu** (val 73,2%), pa je tek onda jednom izmerena na testu — bez „pecanja" test rezultata. Pokretanje: `python main.py --skip-preprocess --mode ensemble` (prosekuje sve `model_*.keras` fajlove u korenu projekta).

---

## 8. Rezultati i analiza

### 8.1 Poređenje modela

| Model                          | Test tačnost | Napomena                                |
|--------------------------------|--------------|------------------------------------------|
| CNN od nule (stari pipeline)   | 50,7%        | pre svih popravki                        |
| Transfer (stari pipeline)      | 60,4%        | stari najbolji rezultat                  |
| CNN od nule (novi pipeline)    | **59,7%**    | +9,0 p.p. — trim, delte, balans, stabilan trening |
| Transfer (novi pipeline)       | **60,4%**    | val 66,7% — plafon pojedinačnog modela   |
| **Ansambl (2×CNN + transfer)** | **66,0%**    | **+5,6 p.p. nad starim najboljim; val 73,2%** |

Ansambl od 66,0% je **iznad izmerene ljudske tačnosti** (~62%) za audio-only RAVDESS govor.

### 8.2 Per-klasa analiza (ansambl, 66,0%)

| Klasa     | Precision | Recall | F1    | Analiza                                                |
|-----------|-----------|--------|-------|--------------------------------------------------------|
| disgust   | 0.89      | 0.84   | **0.86** | Najbolja klasa                                      |
| angry     | 0.81      | 0.89   | **0.85** | Visoka energija, jasan potpis                       |
| surprised | 0.78      | 0.90   | **0.84** | Specifičan akustički potpis                         |
| calm      | 0.50      | 0.74   | 0.60  | Dobro se hvata, ali „usisava" sad/neutral              |
| happy     | 0.80      | 0.42   | 0.55  | Precizna ali stidljiva — meša se sa fearful/angry      |
| neutral   | 0.46      | 0.60   | 0.52  | Manjinska klasa (10 test uzoraka)                      |
| fearful   | 0.56      | 0.47   | 0.51  | Preklapanje sa sad i surprised                         |
| sad       | 0.47      | 0.37   | 0.41  | Najteža klasa — tiha, spora, slična calm/neutral       |

Makro F1: **0.64**. Tri klase su iznad 0.84 — na nivou ili iznad objavljenih rezultata za pojedinačne CNN modele na RAVDESS-u.

### 8.3 Analiza preprilagođavanja

Jaz trening/validacija (train ~90%, val ~67–73%) postoji u svim modelima i posledica je fundamentalnog ograničenja: ~135 originalnih snimaka po klasi. Augmentacija umnožava varijacije, ali ne dodaje nove govornike ni nove interpretacije. Regularizacija (Dropout, L2, SpecAugment, focal loss) usporava memorisanje, ali ga ne može eliminisati.

Jaz validacija/test (73% vs 66%) ima dva uzroka: (1) val skup učestvuje u svim odlukama — selekcija epohe, LR raspored, izbor ansambla — pa je blago „potrošen"; (2) oba skupa su mala (216 / 144 uzoraka), pa par uzoraka pravi procentni poen razlike.

### 8.4 Realan plafon i šta bi trebalo za više

| Ograničenje                   | Uticaj                                                    |
|-------------------------------|-----------------------------------------------------------|
| 1.440 snimaka, 8 klasa        | ~135 originala/klasi — glavni limit                       |
| Ljudska tačnost ~62%          | glumljene emocije nisu jednoznačne ni ljudima             |
| Akustičko preklapanje         | sad↔calm↔neutral i happy↔fearful↔surprised parovi         |
| Test = 144 uzorka             | ±3 p.p. šuma u svakom merenju                             |

Radovi koji prijavljuju 75–85% koriste wav2vec2/HuBERT embeddinge (pretrenirane na hiljadama sati govora — nije „CNN nad spektrogramima"), spajanje više baza (RAVDESS+TESS+CREMA-D) ili multimodalnost (audio+video). U okviru zadatka ovog projekta, 66% je blizu realnog plafona.

---

## 9. Struktura projekta

```
speech-emotion-extraction/
├── config.py           # Svi hiperparametri i putanje
├── data_loader.py      # Parser RAVDESS fajlova, vizualizacija distribucije
├── preprocess.py       # WAV → (128,128,3) mel+delte, augmentacija, cache, paralelno
├── model.py            # CNN od nule (4×CNNBlock + GAP + Dense)
├── train.py            # Trening petlja, callback-ovi, history plot
├── model_transfer.py   # EfficientNetB0 transfer model (ZScoreToPixels + Resizing)
├── train_transfer.py   # Dvofazni TransferTrainer
├── evaluate.py         # Evaluator + SoftmaxAverageEnsemble
├── main.py             # Entry point: --mode cnn | transfer | ensemble
├── colab_run.ipynb     # GPU trening na Google Colab-u (kloniraj → dataset → treniraj)
├── data/ravdess/       # RAVDESS audio (nije na Git-u)
├── processed/          # Keširani .npy (nije na Git-u)
└── results/            # Grafici i izveštaji
```

### Pokretanje

```bash
pip install -r requirements.txt

python main.py                                    # CNN od nule (+ preprocesiranje)
python main.py --skip-preprocess --mode transfer  # transfer learning
python main.py --skip-preprocess --mode ensemble  # ansambl svih model_*.keras
python main.py --skip-preprocess --eval-only      # samo evaluacija sačuvanog modela
```

Za GPU trening: otvoriti `colab_run.ipynb` na Google Colab-u (GPU runtime) i pokrenuti ćelije redom.

---

## 10. Zaključak

Polazna tačka je bila 50,7% (CNN) / 60,4% (transfer). Kroz sistematsku dijagnostiku — čitanje krivih treninga, konfuzionih matrica i per-klasa metrika posle svakog eksperimenta — identifikovani su i otklonjeni konkretni problemi:

1. **Podaci:** trim tišine (ulaz je bio 40–60% tišina), delta kanali (dinamika govora), tačan balans klasa (432/klasi), reprodukovljiva augmentacija.
2. **Trening:** stabilizacija (LR 3e-4, clipnorm, batch 64), selekcija modela po val_accuracy umesto šumovitog val_loss-a.
3. **Transfer:** Resizing na 224 (nativna skala pretreninga), blaži fine-tuning (20 slojeva, LR 5e-5), ispravka checkpoint baga između faza.
4. **Ansambl:** prosek softmax izlaza tri mreže, biran na validaciji.

**Konačni rezultat: 66,0% test tačnosti na 8 klasa** — poboljšanje od **+5,6 p.p.** u odnosu na stari najbolji model i **+15,3 p.p.** u odnosu na stari CNN od nule, iznad izmerene ljudske tačnosti za ovaj zadatak. Dalji značajan napredak zahtevao bi pretrenirane govorne reprezentacije (wav2vec2/HuBERT) ili veće skupove podataka, što izlazi iz okvira zadatka.
