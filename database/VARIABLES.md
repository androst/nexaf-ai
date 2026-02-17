# NEXAF Dataset Variables

This document describes the variables available in each SPSS data file used in the NEXAF-AI analysis.

---

## 1. Episode Data (`df_ep5_fil_sep25_EFL.sav`)

**170,181 episodes × 25 variables**

Individual AF episode records from ILR devices.

| Variable | Description |
|----------|-------------|
| `ID` | Patient ID |
| `serialno` | ILR serial number |
| `date_ilr_implant` | Date of ILR implantation |
| `date_randomization` | Date of randomization (= CPET date) |
| `time_start_ep` | Episode start datetime |
| `time_stop_ep` | Episode end datetime |
| `af_episode_minutes` | Episode duration (minutes) |
| `ep_number` | Episode sequence number |
| `episode_start_during_day` | Daytime episode flag (6:00-22:00) |
| `AF_MEAN_RR_INTERVAL_msec` | Mean RR interval (milliseconds) |
| `AF_MEAN_RR_RATE_bpm` | Mean ventricular rate (bpm) |
| `hour` | Hour of episode start (0-23) |
| `weekday_num_ep` | Day of week (numeric) |
| `weekday_string_ep` | Day of week (string) |

---

## 2. Baseline Data (`NEXAF_baselinefil_171125_EFL.sav`)

**295 patients × 714 variables**

Comprehensive baseline characteristics at study enrollment.

### Demographics

| Variable | Description |
|----------|-------------|
| `ID` | Patient ID |
| `Kjønn` | Sex (1=Male, 2=Female) |
| `F_år` | Birth year |
| `Alder` | Age (years) |
| `BL_høyde` | Height (cm) |
| `BL_vekt` | Weight (kg) |
| `BL_BMI` | Body mass index |
| `BL_midjemål` | Waist circumference (cm) |
| `BL_fettprosent` | Body fat percentage |

### AF History

| Variable | Description |
|----------|-------------|
| `år_AF_diagnose` | Year of first AF diagnosis |
| `BL_AF_type` | AF type at baseline |
| `BL_ablasjon_nei_ja` | Prior ablation (0=No, 1=Yes) |
| `BL_ablasjon_antall` | Number of prior ablations |
| `BL_konvertering_nei_ja` | Prior cardioversion (0=No, 1=Yes) |
| `BL_konvertering_antall` | Number of prior cardioversions |
| `BL_mEHRA` | Modified EHRA symptom score |

### Comorbidities

| Variable | Description |
|----------|-------------|
| `BL_HTN` | Hypertension |
| `BL_DM1` | Diabetes Type I |
| `BL_DM2` | Diabetes Type II |
| `BL_NSTEMI` | Prior NSTEMI |
| `BL_STEMI` | Prior STEMI |
| `BL_angina` | Angina pectoris |
| `BL_CABG` | Prior CABG |
| `BL_HFrEF` | Heart failure with reduced EF |
| `BL_HFpEF` | Heart failure with preserved EF |
| `BL_hjerneinfarkt` | Prior stroke |
| `BL_TIA` | Prior TIA |
| `BL_karsykdom` | Vascular disease (PAD, aorta) |
| `BL_COPD` | COPD (GOLD A/B) |
| `BL_OSA` | Obstructive sleep apnea |
| `CHA2DS2VA` | CHA₂DS₂-VASc score |

### Combined Comorbidity Variables

| Variable | Description |
|----------|-------------|
| `BL_komb.HF` | Any heart failure (HFrEF/HFpEF/other) |
| `BL_komb.TIAinfarkt` | Any cerebrovascular event (TIA/stroke/bleed) |
| `BL_komb.DM` | Any diabetes (Type I or II) |
| `BL_komb.karsykdom` | Any vascular disease |

### Medications

| Variable | Description |
|----------|-------------|
| `BL_betabl` | Beta-blocker use |
| `BL_antiarrytm` | Antiarrhythmic use |
| `BL_antikoag` | Anticoagulation use |
| `BL_platehemmende` | Antiplatelet use |
| `BL_blodtrykkssenkende` | Antihypertensive use |
| `BL_kolesterolsenkende` | Lipid-lowering use |
| `BL_betablokker_idag` | Beta-blocker taken today |
| `BL_kalsiumantagonist` | Calcium channel blocker |

### Laboratory Values

| Variable | Description |
|----------|-------------|
| `BL_kolesterol_total` | Total cholesterol |
| `BL_HDL` | HDL cholesterol |
| `BL_LDL` | LDL cholesterol |
| `BL_triglyserider` | Triglycerides |
| `BL_kreatinin` | Creatinine |
| `BL_eGFR` | Estimated GFR |
| `BL_glukose` | Fasting glucose |
| `BL_hsCRP_pre` | hs-CRP |
| `BL_NTproBNP_pre` | NT-proBNP |
| `BL_TnT_pre` | Troponin T |

### CPET (Cardiopulmonary Exercise Test)

| Variable | Description |
|----------|-------------|
| `BL_CPET_dato` | CPET date |
| `BL_CPET_max_VO2_mlkgmin` | Peak VO₂ (ml/kg/min) |
| `BL_CPET_max_VO2_lmin` | Peak VO₂ (L/min) |
| `BL_CPET_Hfmax` | Maximum heart rate |
| `BL_CPET_max_hf` | Peak heart rate |
| `BL_CPET_RERmax` | Maximum RER |
| `BL_CPET_max_borg` | Maximum Borg score |
| `BL_CPET_max_kmt` | Maximum speed (km/h) |
| `BL_CPET_max_stigning` | Maximum incline (%) |
| `BL_CPET_HF_1min` | HR at 1 min recovery |
| `BL_CPET_steg1_*` | Step 1 submaximal values |

### Echocardiography (BL_ prefix)

#### LV Function
| Variable | Description |
|----------|-------------|
| `BL_lvef_bip_ai` | LV ejection fraction biplane (AI) |
| `BL_ef_biplane_03_percent` | LV EF biplane (%) |
| `BL_lvedv_bip_ai` | LV end-diastolic volume (ml) |
| `BL_lvesv_bip_ai` | LV end-systolic volume (ml) |
| `BL_lvot_sv_ml` | Stroke volume (ml) |
| `BL_lvot_co_l_min` | Cardiac output (L/min) |

#### LA Assessment
| Variable | Description |
|----------|-------------|
| `BL_la_cm` | Left atrial diameter (cm) |
| `BL_la_as_a4c_cm2` | LA area 4-chamber (cm²) |
| `BL_laesv_bip_ai` | LA end-systolic volume biplane (ml) |
| `BL_laesv_mod_bp_ml` | LA ESV modified biplane (ml) |
| `BL_auto_laq_vmax_ml` | LA volume max (ml) |
| `BL_auto_laq_vmin_ml` | LA volume min (ml) |
| `BL_auto_laq_gls_r_percent` | LA reservoir strain (%) |
| `BL_auto_laq_gls_cd_percent` | LA conduit strain (%) |
| `BL_auto_laq_gls_ct_percent` | LA contractile strain (%) |

#### RV Function & Diastolic
| Variable | Description |
|----------|-------------|
| `BL_mm_tapse_cm` | TAPSE (cm) |
| `BL_rv_sprime_velocity_m_s` | RV S' velocity (m/s) |
| `BL_mv_e_velocity_m_s` | Mitral E velocity (m/s) |
| `BL_mv_a_velocity_m_s` | Mitral A velocity (m/s) |
| `BL_mv_e_a_ratio_1` | E/A ratio |
| `BL_mv_e_eprime_average_ratio_calc_1` | E/e' ratio (average) |

### Quality of Life & Symptoms

| Variable | Description |
|----------|-------------|
| `BL_AFEQT_total_score` | AFEQT total score |
| `BL_AFEQT_subscore_symptoms` | AFEQT symptoms subscore |
| `BL_AFEQT_subscore_activities` | AFEQT activities subscore |
| `BL_AFEQT_subscore_treatment` | AFEQT treatment subscore |
| `BL_RAND_PCS12` | SF-12 Physical Component Score |
| `BL_RAND_MCS12` | SF-12 Mental Component Score |
| `BL_EQVAS` | EQ-VAS health rating |
| `BL_Afssc_total_frequency` | AF symptom checklist (frequency) |
| `BL_Afssc_total_severity` | AF symptom checklist (severity) |
| `BL_Afssc1a-16a` | Individual symptom frequency |
| `BL_Afssc1b-16b` | Individual symptom severity |

### Lifestyle

| Variable | Description |
|----------|-------------|
| `BL_ExeF` | Exercise frequency |
| `BL_ExeInt` | Exercise intensity |
| `BL_ExeDu` | Exercise duration |
| `BL_ExeBorg` | Exercise Borg scale |
| `BL_SmoStat` | Smoking status |
| `BL_AlcFLY` | Alcohol frequency (last 12 months) |
| `BL_Smoking_status` | Smoking status (Never/Former/Current) |

---

## 3. Burden & Outcomes (`Hovedfil_analyser_hovedartikkel021225_oppdatertAFburden.sav`)

**294 patients × 1,155 variables**

Main analysis file with outcomes and longitudinal data.

### Randomization

| Variable | Description |
|----------|-------------|
| `Rand_arm` | Randomization arm (1=Training, 2=Control) |
| `Rand_tidspunkt` | Randomization datetime |
| `date_cpet` | CPET/randomization date |
| `Strat_senter` | Stratification center |
| `Strat_AF_type` | Stratification AF type |

### AF Burden (Study Period)

| Variable | Description |
|----------|-------------|
| `af_burden_4w_prerand` | AF burden 4 weeks pre-randomization |
| `af_burden_percent_studyperiod` | AF burden % during study |
| `af_burden_percent_studyperiod_censabl` | AF burden % (censored at ablation) |
| `n_af_episodes_studyperiod` | Number of episodes during study |
| `n_af_episodes_over1h_studyperiod` | Episodes >1 hour |
| `n_af_episodes_over24h_studyperiod` | Episodes >24 hours |
| `mean_af_ep_dur_mins_studyperiod` | Mean episode duration (min) |
| `median_af_ep_dur_mins_studyperiod` | Median episode duration (min) |
| `mean_af_ep_heartrate_studyperiod` | Mean HR during AF |
| `time_under_obs_days` | Days under observation |

### Post-Intervention Procedures

| Variable | Description |
|----------|-------------|
| `Post_ablasjon_nei_ja` | Post ablation (0=No, 1=Yes) |
| `Post_ablasjon_antall` | Number of post ablations |
| `Post_konvertering_nei_ja` | Post cardioversion (0=No, 1=Yes) |
| `Post_konvertering_antall` | Number of post cardioversions |

### Post-Intervention Medications

| Variable | Description |
|----------|-------------|
| `Post_betabl` | Post beta-blocker use |
| `Post_antiarrytm` | Post antiarrhythmic use |
| `Post_antikoag` | Post anticoagulation use |

### Exercise Data

| Variable | Description |
|----------|-------------|
| `BL_ExeMin` | Baseline exercise minutes/week |
| `Six_months_ExeMin` | 6-month exercise minutes/week |
| `Post_ExeMin` | Post exercise minutes/week |

### Quality of Life - 6 Months

| Variable | Description |
|----------|-------------|
| `Six_months_Afeqt_symptoms_score` | AFEQT symptoms at 6 months |
| `Six_months_Afeqt_activities_score` | AFEQT activities at 6 months |
| `Six_months_Afeqt_treatment_score` | AFEQT treatment at 6 months |
| `Six_months_Afeqt_total_score` | AFEQT total at 6 months |

### Quality of Life - Post

| Variable | Description |
|----------|-------------|
| `Post_Afeqt_symptoms_score` | AFEQT symptoms post |
| `Post_Afeqt_activities_score` | AFEQT activities post |
| `Post_Afeqt_treatment_score` | AFEQT treatment post |
| `Post_Afeqt_total_score` | AFEQT total post |
| `AFEQT_change` | Change in AFEQT |

### Hospitalization

| Variable | Description |
|----------|-------------|
| `Post_AF_hosp` | AF-related hospitalization (count) |

### Adverse Events

| Variable | Description |
|----------|-------------|
| `AE1_type` - `AE3_type` | Adverse event type |
| `AE1_grad` - `AE3_grad` | Adverse event severity |
| `AE1_trening` - `AE3_trening` | Training-related flag |

### Post Echocardiography (POST_ prefix - uppercase)

Same variables as baseline with `POST_` prefix instead of `BL_`.

| Baseline | Post | Description |
|----------|------|-------------|
| `BL_lvef_bip_ai` | `POST_lvef_bip_ai` | LV EF Biplane (%) |
| `BL_lvedv_bip_ai` | `POST_lvedv_bip_ai` | LV EDV (ml) |
| `BL_lvesv_bip_ai` | `POST_lvesv_bip_ai` | LV ESV (ml) |
| `BL_la_cm` | `POST_la_cm` | LA Diameter (cm) |
| `BL_laesv_bip_ai` | `POST_laesv_bip_ai` | LA ESV Biplane (ml) |
| `BL_auto_laq_vmax_ml` | `POST_auto_laq_vmax_ml` | LA Volume Max (ml) |
| `BL_auto_laq_gls_r_percent` | `POST_auto_laq_gls_r_percent` | LA Reservoir Strain (%) |
| `BL_lvot_sv_ml` | `POST_lvot_sv_ml` | Stroke Volume (ml) |
| `BL_lvot_co_l_min` | `POST_lvot_co_l_min` | Cardiac Output (L/min) |

### Post CPET (Post_CPET_ prefix - mixed case)

| Baseline | Post | Description |
|----------|------|-------------|
| `BL_CPET_max_VO2_mlkgmin` | `Post_CPET_max_VO2_mlkgmin` | VO₂peak (ml/kg/min) |
| `BL_CPET_max_VO2_lmin` | `Post_CPET_max_VO2_lmin` | VO₂peak (L/min) |
| `BL_CPET_Hfmax` | `Post_CPET_Hfmax` | HR Max (bpm) |
| `BL_CPET_max_hf` | `Post_CPET_max_hf` | Peak HR (bpm) |
| `BL_CPET_RERmax` | `Post_CPET_RERmax` | RER Max |
| `BL_CPET_max_kmt` | `Post_CPET_max_kmt` | Max Speed (km/h) |
| `BL_CPET_max_stigning` | `Post_CPET_max_stigning` | Max Incline (%) |
| `BL_CPET_max_borg` | `Post_CPET_max_borg` | Peak Borg Score |
| `BL_CPET_HF_1min` | `Post_CPET_HF_1min` | HR at 1 min Recovery (bpm) |

---

## 4. Validation Data (`valideringsdatasett300126.sav`)

**2,518 episodes × 4 variables**

Manual validation of AF episodes.

| Variable | Description |
|----------|-------------|
| `ID` | Patient ID |
| `time_start_ep` | Episode start time |
| `af_episode_minutes` | Episode duration |
| `afib_validert` | Validated as true AF (0/1) |

---

## Variable Naming Conventions

| Prefix | Meaning |
|--------|---------|
| `BL_` | Baseline measurement |
| `Six_months_` | 6-month follow-up |
| `Post_` / `POST_` | Post-intervention measurement |
| `Strat_` | Stratification variable |
| `Rand_` | Randomization variable |
| `AE#_` | Adverse event (numbered) |

---

## Value Codings & Counts

### Key Categorical Variables

| Variable | Value | Label | N |
|----------|-------|-------|---|
| `Kjønn` (Sex) | 1 | Male | 203 |
| | 2 | Female | 90 |
| `BL_AF_type` | 1 | Paroxysmal | 223 |
| | 2 | Persistent | 72 |
| `Strat_AF_type` | 6 | Paroksysmal | 222 |
| | 7 | Persisterende | 72 |
| `Rand_arm` | 1 | Training | 146 |
| | 2 | Control | 148 |

### mEHRA Score (`BL_mEHRA`)

| Value | N | Description |
|-------|---|-------------|
| 1 | 13 | No symptoms |
| 2a | 55 | Mild symptoms (2A/2a combined: 63) |
| 2b | 73 | Moderate symptoms (2B/2b combined: 74) |
| 3 | 90 | Severe symptoms |
| 4 | 33 | Disabling symptoms |
| 2, 2-3 | 19 | Ambiguous coding |
| . | 3 | Missing |

### Binary Variables (0=No, 1=Yes)

| Variable | No (n) | Yes (n) |
|----------|--------|---------|
| `BL_HTN` | 172 | 123 |
| `BL_ablasjon_nei_ja` | 238 | 57 |
| `BL_konvertering_nei_ja` | 166 | 129 |
| `Post_ablasjon_nei_ja` | — | 20 |
| `Post_konvertering_nei_ja` | — | 44 |

**Note**: Post variables use missing (NaN) for "No" rather than 0.

## Notes

- **Norwegian variable labels**: Many variables have Norwegian labels (e.g., "Kjønn" = Sex, "Alder" = Age)
- **Sex variable (`Kjønn`)**: Coded as 1=Male, 2=Female (NOT 0/1)
- **Binary variables**: Generally coded as 0=No, 1=Yes (except Sex)
- **Missing data**: Represented as system missing in SPSS
- **Date formats**: Various date/time formats; see `pyreadstat` for parsing
- **POST prefix conventions**:
  - Echo variables: `POST_` (uppercase) - e.g., `POST_lvef_bip_ai`
  - CPET variables: `Post_CPET_` (mixed case) - e.g., `Post_CPET_max_VO2_mlkgmin`
  - Clinical variables: `Post_` (title case) - e.g., `Post_ablasjon_nei_ja`
- **LA volumes**: Only LAESV (end-systolic) available, no LAEDV (end-diastolic)

## Data File Contents

| File | Variables |
|------|-----------|
| `NEXAF_baselinefil_171125_EFL.sav` | `BL_*` baseline variables (echo, CPET, labs, etc.) |
| `Hovedfil_analyser_hovedartikkel021225_oppdatertAFburden.sav` | `POST_*` echo, `Post_CPET_*` CPET, `Post_*` clinical, outcomes |
| `df_ep5_fil_sep25_EFL.sav` | Episode-level AF data |
| `valideringsdatasett300126.sav` | Validation data |

**Important**: Post-intervention data (`POST_*`, `Post_CPET_*`) is in the **Hovedfil** (burden file), NOT the baseline file.
