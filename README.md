# Heuristic-Based Saliency Prediction for Abstract Art

This repository contains the source code for the master's thesis
**"Stebėtojo dėmesio prognozė abstrakčioje tapyboje"**
(*Observer’s Attention Prediction in Abstract Art*), completed at Vilnius Tech in 2025.

The system implements a cognitively motivated saliency prediction model based on visual attention principles from art theory. It uses a modular heuristic approach, a genetic algorithm for rule optimization, and a postprocessing pipeline to enhance fixation prediction.

---

## Projekto struktūra

### `src/`

Pagrindinis kodo katalogas, kuriame saugomos visos sistemos dedamosios. Kiekvienas modulis atitinka konkretų modelio komponentą:

- **`heuristics/`**
  Implementuoja meno teorijos euristikas:
  - **kontrasto** (contrast),
  - **izoliacijos** (isolation),
  - **grupavimo** (grouping),
  - **simetrijos** (symmetry) dėsnius.
  Kiekviena euristika sukuria saliency žemėlapį pagal atitinkamą taisyklę.

- **`modulation/`**
  Kontekstinės moduliacijos logika.
  Čia aprašyti paveikslo bruožų (features) išskyrimo metodai ir moduliacijos formulės, kurios leidžia modelio svorius pritaikyti konkretaus paveikslo savybėms.

- **`postprocessing/`**
  Dėmesio židinių išskyrimo ir peak-shift stiprinimo algoritmai.
  Šiame etape saliency žemėlapis transformuojamas į aiškesnes, fiksacijas atitinkančias struktūras.

- **`ga/`**
  Klasikinio genetinio algoritmo realizacija.
  Atliekamas euristikų svorių optimizavimas, siekiant geriausio atitikimo su žmogaus dėmesio duomenimis.

- **`ga_mod/`**
  Moduliuoto genetinio algoritmo versija.
  Papildytas paveikslo bruožų vektorių integracija, leidžianti skaičiuoti dinamiškai adaptuotus svorius pagal turinį.

- **`utils/`**
  Pagalbinės funkcijos: duomenų įkėlimas, žemėlapių normalizavimas, failų valdymas.

---

## Kiti katalogai

- **`data/`**
  Įvesties duomenys ir tarpinių rezultatų failai:
  - `input_images/`: Paveikslų įvestys
  - `heuristic_maps/`: Sugeneruoti euristiniai žemėlapiai
  - `ground_truth_saliency/`: Žmogaus fiksacijų žemėlapiai
  - `feature_vectors/`: Iš anksto apskaičiuoti paveikslo bruožai moduliacijai

- **`results/`**
  Čia įrašomi galutiniai rezultatai:
  - `saliency_output/`: Kombinuoti ir postprocesuoti žemėlapiai
  - `evaluation_scores/`: SIM, CC, KL įverčiai
  - `plots/`: Fitneso kreivės ir analizės grafikai

---

## Pagrindiniai failai

- **`heuristics.py`**
  Euristikų funkcijų sąrašas.

- **`run_ga.py`**
  Paleidžia klasikinį genetinį algoritmą. Naudoja fiksuotus euristikų svorius.

- **`run_ga_mod.py`**
  Paleidžia moduliuotą GA versiją. Kiekvienas paveikslas gauna individualiai adaptuotus svorius.

- **`generate_final_maps.py`**
  Naudojant `best_individual.json` sukonstruoja galutinius saliency žemėlapius konkrečiam euristikų rinkiniui (moduliuotam arba ne).

- **`evaluate_model.py`**
  Atlieka palyginimą su žmogaus dėmesio duomenimis. Skaičiuoja SIM, CC, KL-divergenciją.

- **`requirements.txt`**
  Visų naudotų Python bibliotekų sąrašas.

---

## Kodo logika ir vykdymo eiga

1. **Euristikų saliency žemėlapių generavimas**
   -> kiekvienas paveikslas apdorojamas visomis euristikomis.

2. **Optimizavimas su genetiniu algoritmu**
   -> euristikų svoriai koreguojami pagal atitikimą su žmogaus dėmesio žemėlapiais.

3. **Moduliacija (nebūtina)**
   -> euristikų svoriai adaptuojami pagal paveikslo savybes.

4. **Postprocesavimas**
   -> išskiriami dėmesio centrai naudojant peak-shift principą.

5. **Vertinimas**
   -> modelio rezultatai lyginami su žmogaus duomenimis.

---

## Pastabos skaitytojui

Šis projektas orientuotas į modelio **skaidrumą ir aiškumą**, todėl kiekviena taisyklė realizuota atskirai, o visa sistema sudaryta iš **interpretuojamų komponentų**, kuriuos galima keisti, pridėti ar optimizuoti.

Šaltinio kodas pateikiamas kaip nuoroda šio darbo prieduose.

---

