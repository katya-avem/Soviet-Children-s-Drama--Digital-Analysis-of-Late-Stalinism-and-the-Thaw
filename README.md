# Soviet Children’s Drama: Digital Analysis of Late Stalinism and the Thaw

This repository contains analytical code and metadata for a Digital Humanities project devoted to the study of Soviet children’s drama from late Stalinism to the Khrushchev Thaw. The project applies quantitative text analysis methods to explore how models of family, childhood, and moral authority were constructed and transformed in postwar Soviet culture.

The project is part of the PhD dissertation  
**_The Path of the Soviet Family from Late Stalinism to the Thaw_**,  
by **Ekaterina Kolevatova**,  
conducted at **LMU Munich**.

---

## Project Scope

The research investigates ideological representations of the family in Soviet children’s theatre during two key periods:

- **Late Stalinism (1945–1953)**
- **Khrushchev Thaw (1954–1964)**

Children’s theatre serves as a particularly revealing object of analysis. As one of the most intensively regulated cultural domains, it played a central role in shaping normative models of behavior, morality, and emotional life for young audiences. Its tendency toward clear moral polarization (right/wrong, good/bad, collective/individual) makes it especially suitable for the study of Soviet ideological discourse.

Because the process of de-Stalinization was uneven and gradual, shifts in family and childhood representations are often difficult to capture through close reading alone. This project therefore adopts a hybrid methodological approach, combining computational analysis with traditional literary and cultural interpretation.

---

## Data Availability and Copyright

The full texts of the plays are **not included** in this repository. All plays remain under copyright  
(Russian copyright law: author’s lifetime + 70 years).

### What is publicly available here:

- structured metadata for all plays included in the study,
- analytical code for **TF-IDF**, **Z-score analysis**, and **topic modeling**,
- documentation of preprocessing and methodological decisions.

Aggregated data and further materials may be shared upon request for academic purposes.

---

## Corpus Description (Metadata)

The underlying corpus used in the dissertation consists of **54 Soviet children’s plays**, divided into two subcorpora:

- **23 late Stalinist plays** (141,264 tokens after lemmatization)
- **31 Thaw-era plays** (161,900 tokens after lemmatization)

All token counts and vocabulary sizes were calculated **after lemmatization** to ensure cross-period comparability.

The plays were drawn primarily from two major Soviet theatrical sources:

- **_Pionerskii teatr_**, a collection published under the auspices of the Union of Soviet Writers in cooperation with the Central Committee of the Komsomol and the *Molodaia Gvardiia* publishing house. The series aimed to support children’s amateur theatre and school drama circles by providing ideologically approved and easily stageable plays.

- **_Teatr_**, the central Soviet theatre journal and official organ of the Committee for Arts. The journal documented major premieres, theatrical debates, and normative expectations for contemporary drama across the Soviet Union.

This repository includes tables with metadata for both subcorpora, listing original titles, English translations, authors, and publication sources.

---

## Methods Implemented in This Repository

The analytical component of the project relies on three complementary computational techniques:

### 1. TF-IDF (Term Frequency–Inverse Document Frequency)

TF-IDF was used to identify lexemes that are especially characteristic of individual plays and subcorpora. This method highlights contrasts between family-oriented and collective vocabulary, as well as between emotional and ideological registers.

### 2. Z-Score Analysis

Z-score analysis was applied to compare lexical distributions across the two historical periods. While TF-IDF captures internal distinctiveness, Z-analysis identifies statistically significant differences between the late Stalinist and Thaw corpora, making it possible to trace ideological and semantic shifts over time.

### 3. Topic Modeling (LDA and STM)

Latent Dirichlet Allocation (LDA) was used to extract stable thematic clusters across the corpus. Structural Topic Modeling (STM) incorporates temporal metadata (year of publication), allowing topic prevalence to be traced diachronically. This makes it possible to visualize how thematic emphases change in relation to historical and cultural transformations.

All topic interpretation is supported by close reading, concordance analysis (AntConc), and engagement with scholarship on Soviet cultural and literary ideology.

---

## Repository Contents

- `corpus/` — tables with plays and metadata for both subcorpora  
- `src/` — scripts for TF-IDF, Z-analysis, and topic modeling  
- `figures/` — outputs for visualizations  
- `README.md` — project description and documentation