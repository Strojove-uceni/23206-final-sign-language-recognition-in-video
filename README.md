# Finální projekt Sign Language Recognition in Video
**Autoři:** Jiří Nábělek a David Rendl

Vysvětlení metod a postupů je k dispozici v souboru SU2_Final_Notebook

**Upozornění: Důrazně doporučujeme stáhnout repozitář a spouštět kódy lokálně, Google Colab může způsobovat problémy, a především inferenci v něm vůbec nespustíte.**

## Spuštění
1. Postup pro lokální supštění generování datasetu.
- Ve skriptu `Secondary_buffer.py` přesat proměnnou `path`, aby naváděla do kmenové složky datasetu (nebo stáhnout připravená data z Google Disku).
- Pro vygenerování celého datasetu vyprazdnit list `sings`. Pro vygenerování konkrétních slov je potřeba tyto slova vypsat do listu.
- Data se vygenerují do složky Matrix v kmenové složce.

2. Postup pro lokální spuštění trénování:
- Ve skriptu `Model_training.py` nastavit hyperparametry ve slovníku `params` (především přepsat cestu ke složce Matrix, přepsat klíč k Weight & Biases).
- Inicializovat model ze seznamu modelů.

## Seznam modelů:
- AslMatrixModel
- AslMatrixModel2
- ImageClassifier (předtrénovaný ResNet)
- ResNet50
- AslCnnRnnModel

Jmenované modely jsou nejúspěšnějšími pokusy (pro více informaci viz `Final_SU2_notebook.ipynb`).
