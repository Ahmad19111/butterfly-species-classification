# Butterfly Species Classification

Projekt im Rahmen der Lehrveranstaltung **Deep Learning for Computer
Vision** an der Hochschule Bielefeld (HSBI).

Ziel des Projekts ist die Klassifikation von Bildern fünf ausgewählter
Schmetterlingsarten sowie der Vergleich verschiedener
Transfer-Learning-Verfahren.

**GitHub-Repository:**
https://github.com/Ahmad19111/butterfly-species-classification

> **Projektstatus:** Alle Experimente sind abgeschlossen und auf dem
> Test-Set ausgewertet (siehe [Ergebnisse](#ergebnisse)).

## Projektziele

Im Projekt wurden vortrainierte Bildverarbeitungsmodelle für die
Schmetterlingsklassifikation untersucht. Die drei Versuchsansätze sind
abgeschlossen:

1.  **CNN -- Transfer Learning:** Anpassung eines vortrainierten
    Convolutional Neural Network an die ausgewählten
    Schmetterlingsklassen.
2.  **ViT -- Linear Probing:** Der vortrainierte Vision Transformer
    bleibt eingefroren; trainiert wird ausschließlich ein
    Klassifikationskopf.
3.  **ViT -- LoRA:** Anpassung eines vortrainierten Vision Transformers
    mithilfe von Low-Rank Adaptation (LoRA).

Das CNN wird entsprechend der Vorgabe der Lehrveranstaltung mit
vortrainierten Gewichten verwendet und nicht von Grund auf neu
trainiert.

## Datensatz

Der verwendete Datensatz umfasst Bilder von fünf Schmetterlingsarten.
Motten sind ausgeschlossen.

  Klasse                    Anzahl der Bilder
  ----------------------- -------------------
  Apatura iris                          1.182
  Brenthis daphne                         946
  Hamearis lucina                       1.211
  Melitaea diamina                      1.105
  Nymphalis polychloros                 1.040
  **Gesamt**                        **5.484**

Die Bilder wurden manuell überprüft. Entfernt wurden Collagen, unscharfe
oder stark verpixelte Bilder, exakte Duplikate sowie Aufnahmen, auf
denen der Schmetterling zu klein oder zu weit entfernt war.

### Quelle des Datensatzes

Barkmann et al. (2025), *Scientific Data*, 12, Artikel 1369.\
https://doi.org/10.1038/s41597-025-05708-z

Figshare-Datensatz: https://doi.org/10.25452/figshare.plus.29135618

Die ursprüngliche Sammlung umfasst zahlreiche Schmetterlings- und
Mottenarten. Für dieses Projekt werden ausschließlich die fünf oben
aufgeführten Schmetterlingsarten verwendet.

## Datenaufteilung (Group Split)

Die Daten werden stratifiziert im Verhältnis 70/15/15 aufgeteilt
(Seed 42): 3.836 Trainings-, 824 Validierungs- und 824 Testbilder.

Eine rein zufällige Aufteilung ist hier ungeeignet: Viele Bilder stammen
aus derselben Beobachtung (gleiches Individuum, gleiche Aufnahmeserie)
und sind sich sehr ähnlich. Landen solche Bilder in Training und Test,
überschätzt die Testgenauigkeit die echte Generalisierung (Data Leakage).
Bei der ursprünglichen Zufallsaufteilung lagen 451 Bildpaare mit einem
Abstand der Beobachtungs-IDs von höchstens 5 über Split-Grenzen hinweg,
davon 154 mit direkt benachbarter ID (Abstand 1).

Deshalb werden Bilder zu **Gruppen** zusammengefasst, die immer
vollständig in einem Split landen. Zwei Bilder derselben Klasse gehören
zur selben Gruppe, wenn

-   ihre Beobachtungs-IDs höchstens 5 auseinanderliegen, oder
-   sie nahezu identisch sind (Perceptual Hash, Hamming-Abstand ≤ 4).

Die Gruppen sind die Zusammenhangskomponenten dieses Graphen (4.510
Gruppen, größte mit 11 Bildern). Danach liegen 0 Gruppen und 0
Beinahe-Duplikate über mehreren Splits. Schwellwerte stehen in
`configs/data.yaml`; die alte Zufallsaufteilung liegt zum Vergleich in
`data/splits/random_split/`.

## Ablauf der Skripte

Alle Skripte werden im Projektordner als Modul aufgerufen; alle
Hyperparameter stehen in `configs/`.

| Schritt | Befehl | Ergebnis |
|---|---|---|
| 1. Datenanalyse | `python -m src.data.analyze_data` | `data/metadata/data_stats.csv` |
| 2. Vorverarbeitung | `python -m src.data.preprocess` | `data/processed/` (EXIF-Korrektur, kürzere Seite 384 px) |
| 3. Aufteilung | `python -m src.data.make_splits` | `data/splits/{train,val,test}.csv` |
| 4. CNN Fine-Tuning | `python -m src.training.train_cnn` | ResNet50, Phase A Kopf, Phase B alles |
| 5. ViT Linear Probing | `python -m src.training.train_vit_lp` | Features gecacht, nur Kopf trainiert |
| 6. ViT LoRA | `python -m src.training.train_vit_lora --rank 4` (ebenso 8, 16) | LoRA auf `qkv`, α = 2r |
| 7. Test-Auswertung | `python -m src.evaluation.evaluate` | Metriken und Vorhersagen unter `experiments/results/` |
| 8. Signifikanztest | `python -m src.evaluation.mcnemar` | McNemar-Ergebnisse in `reports/mcnemar.csv` |

Optionale Kontrollen: `python -m src.data.check_duplicates`
(Beinahe-Duplikate über Splits) und `python -m src.data.show_augmentations`
(Beispielbilder der Augmentierung). Die Trainingsskripte unterstützen
`--debug` für einen kurzen Testlauf. Schritte 4 und 6 lassen sich mit
`run_all.ps1` nacheinander ausführen.

## Ergebnisse

Test-Set des Group Split (824 Bilder), Eingabe 224×224, Seed 42,
NVIDIA GTX 1650 (4 GB). In Klammern: 95-%-Bootstrap-Konfidenzintervall
(1.000 Resamples). Zeit = reine Trainingszeit (bei Linear Probing inkl.
Feature-Extraktion des Trainingssplits).

| Modell | Genauigkeit [%] | Makro-F1 [%] | Trainierb. Param. | Zeit [min] | GPU-Speicher [MB] |
|---|---|---|---:|---:|---:|
| ResNet50 Fine-Tuning | 98,91 [98,1; 99,5] | 98,92 [98,1; 99,5] | 23,5 Mio. | 34,1 | 1.745 |
| ViT-B/16 Linear Probing | 98,42 [97,5; 99,3] | 98,48 [97,5; 99,3] | 3,8 Tsd. | 7,4 | 697 |
| ViT-B/16 LoRA r = 4 | 98,79 [98,1; 99,5] | 98,81 [98,0; 99,5] | 151,3 Tsd. | 111,3 | 2.087 |
| ViT-B/16 LoRA r = 8 | **99,15** [98,5; 99,8] | **99,18** [98,6; 99,8] | 298,8 Tsd. | 114,9 | 2.089 |
| ViT-B/16 LoRA r = 16 | 98,79 [97,9; 99,5] | 98,80 [98,0; 99,5] | 593,7 Tsd. | 114,9 | 2.094 |

Die vollständigen Metriken, Trainingsverläufe und Vorhersagen liegen in
`experiments/results/`. Die zusammengefassten Testmetriken stehen in
`experiments/results/test_results.csv`; die Einzelwerte sind zusätzlich
in den jeweiligen JSON-Dateien gespeichert.

Kein paarweiser Unterschied ist statistisch signifikant (exakter
McNemar-Test mit Holm-Korrektur, alle korrigierten p = 1,0; kleinstes
unkorrigiertes p = 0,146 für Linear Probing vs. LoRA r = 8). Zum
Vergleich erreicht ResNet50 auf der alten Zufallsaufteilung 99,15 %.

## Projektstruktur

``` text
butterfly-species-classification/
├── configs/                  # Konfigurationsdateien für Experimente
├── data/
│   ├── metadata/             # Datenstatistik und Duplikatprüfung
│   └── splits/               # Group- und Random-Splits
├── experiments/
│   ├── logs/                 # Ausgewählte Trainingsprotokolle
│   └── results/              # Metriken, Verläufe und Vorhersagen
├── reports/
│   └── mcnemar.csv           # Ergebnis des paarweisen Signifikanztests
├── src/
│   ├── data/                 # Laden und Vorverarbeitung der Daten
│   ├── evaluation/           # Evaluation und Metriken
│   ├── models/               # Modelldefinitionen
│   ├── training/             # Trainingsabläufe
│   └── utils/                # Gemeinsame Hilfsfunktionen
├── .gitignore
├── README.md
├── requirements.txt
└── run_all.ps1
```

Rohbilder, verarbeitete Bilder, Feature-Caches, Checkpoints und die
virtuelle Umgebung bleiben lokal und werden nicht in Git versioniert.
Lege die Bilder unter `data/raw/<Klassenname>/` ab, bevor du die
Datenanalyse und Vorverarbeitung startest. Die CSV-Dateien in
`data/splits/` und die kompakten Versuchsergebnisse werden versioniert.

## Einrichtung der Entwicklungsumgebung

### Voraussetzungen

-   Python 3.12
-   Git
-   Für lokales GPU-Training wird eine NVIDIA-GPU mit kompatiblem
    Treiber empfohlen.
-   Verwendete Kernbibliotheken: PyTorch, TorchVision, Transformers,
    PEFT und timm.

### 1. Repository klonen

``` bash
git clone https://github.com/Ahmad19111/butterfly-species-classification.git
cd butterfly-species-classification
```

### 2. Virtuelle Umgebung erstellen und aktivieren

**Windows PowerShell:**

``` powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 3. PyTorch und TorchVision installieren

PyTorch und TorchVision müssen passend zum Betriebssystem, zur GPU und
zum installierten Treiber ausgewählt werden.

Für die aktuelle lokale Windows-Entwicklungsumgebung wurde folgender
Befehl verwendet:

``` powershell
python -m pip install torch==2.15.0+cu132 torchvision==0.29.2+cu132 --index-url https://download.pytorch.org/whl/cu132
```

Falls diese Variante für ein anderes System nicht geeignet ist, sollte
der passende Installationsbefehl über den offiziellen PyTorch-Installer
ausgewählt werden:

https://pytorch.org/get-started/locally/

### 4. Weitere Abhängigkeiten installieren

``` powershell
python -m pip install -r requirements.txt
```

### 5. GPU-Unterstützung überprüfen (optional)

``` powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA verfügbar:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Nicht verfügbar')"
```

Der Wert `True` bei `torch.cuda.is_available()` bedeutet, dass PyTorch
auf CUDA zugreifen kann.

## Datenverwaltung und Zusammenarbeit

-   Der Bilddatensatz, die virtuelle Umgebung und große
    Modell-Checkpoints werden nicht in GitHub hochgeladen.
-   Die ausgewählten Artenordner bleiben lokal unter `data/raw/` oder
    werden über den gemeinsam genutzten Speicherort des Teams
    bereitgestellt.
-   Alle Teammitglieder verwenden dieselben Dateien zur Aufteilung in
    Trainings-, Validierungs- und Testdaten. Das ist wichtig für faire
    und vergleichbare Experimente.
-   Für jeden Versuch werden mindestens Random Seed, Modell und
    vortrainierte Gewichte, Eingabegröße, Batch Size, Learning Rate,
    Anzahl der Epochen und Softwareversionen dokumentiert.
-   Quellcode, Konfigurationsdateien und ausgewählte Ergebnisse werden
    über GitHub geteilt. Der Abschlussbericht wird separat abgegeben und
    ist nicht Teil des Code-Repositories.

## Reproduzierbarkeit

Für einen fairen Vergleich der Verfahren müssen dieselbe Datenaufteilung
und dokumentierte Versuchseinstellungen verwendet werden. Zufalls-Seeds
sollten festgelegt und protokolliert werden. Dennoch können geringe
Ergebnisunterschiede durch abweichende Hardware, Softwareversionen oder
GPU-Ausführungen entstehen.

## Literatur und Datenquellen

Barkmann et al. (2025). Veröffentlichung des Datensatzes in *Scientific
Data*, 12, Artikel 1369. https://doi.org/10.1038/s41597-025-05708-z

Datensatz auf Figshare: https://doi.org/10.25452/figshare.plus.29135618
