# AGENTS.md — AI Technical Handover & Architecture Contract (`partial-functions`)

Dette dokumentet er skrevet for AI-agenter (Antigravity, Claude, Cursor, m.fl.) som jobber i dette repoet eller bruker `partial-functions` som matematisk og algoritmisk grunnmur i tilknyttede prosjekter (som `doc-to-vec`, `metadata-roaring` og `sqlite-backend`).

---

## 1. Kjernefilosofi: «Fortran-modellen» for partielle funksjoner

I stedet for å modellere tekstvektorer, kollokasjoner og tidsserier som fikserte $N$-tupler i $\mathbb{R}^N$ (fulle av nuller) eller som trege Python-`dict`/`set`, representeres alt som **partielle funksjoner med endelig støtte**:

$$f : D \rightharpoonup \mathbb{R}$$

### Datastrukturen (`PartialFunc` i [partial_func.py](partial_func.py))
Hver partiell funksjon består av to stramt alignede komponenter (Structure of Arrays / «Fortran-modellen»). Navnet viser til Fortran-stilen med parallelle, flate arrays i stedet for en array av objekter, ikke til kolonne-major-lagring av matriser:
1. **`dom : pyroaring.BitMap`** — En 32-bits Roaring Bitmap (`uint32`) som representerer definisjonsområdet $\text{dom}(f) \subseteq \mathbb{N}$ (f.eks. globale ord-ID-er $1 \dots M$, dokument-ID-er, chunk-ID-er eller årstall).
2. **`val : np.ndarray[float64]`** — Et kontinuerlig flyttalls-array av lengde $|\text{dom}(f)|$, sortert i **eksakt samme rekkefølge** som elementene i `dom`. Det $i$-te satte bitet i `dom` har sin verdi i `val[i]`.

**Eksplisitte nuller:** `PartialFunc` filtrerer ikke bort `val[i] == 0`. $w \in \text{dom}(f)$ med $f(w) = 0$ er derfor noe annet enn $w \notin \text{dom}(f)$ for snitt-operasjoner (produkt, kvotient, restriksjon og distansene i 2B). Fjern nuller med en restriksjon før du sammenligner domener, hvis det er det du mener.

### Den tredelte arbeidsflyten: Koding $\to$ Funksjons-algebra $\to$ Dekoding
1. **Koding (Inn):** Ord (strenger), metadata-uttrykk og tidsrom kodes **én gang** ved inngangen til heltalls-ID-er, `BitMap` og `PartialFunc` (se [`AlgebraCodec`](codec_and_traceability.py)).
2. **Funksjons-algebra (Midt):** Alle beregninger skjer lukket inne i algebraen over `(BitMap, ndarray)` — helt uten strenger, uten SQL-joins og uten oppslag i enkeltbøker.
3. **Ettersporbar dekoding (Ut):** Kun sluttresultatet (f.eks. `pf.top_k(10)` eller konkordans-pekere `doc_id = c >> 11, chunk = c & 2047`) dekodes tilbake til tekst.

### Grenser for `chunk_id`
$\text{chunk\_id} = (\text{doc\_id} \ll 11) \mid \text{local\_chunk}$ skal få plass i en 32-bits `BitMap`. Det gir to harde grenser:
* **`doc_id` < $2^{21}$** (≈ 2,1 millioner dokumenter per indeks). Større samlinger, for eksempel aviser, må deles i shards eller bruke et annet skift.
* **`local_chunk` ≤ 2047** (2 048 chunks per dokument). `collocations_and_trends.py` klipper stille til 2047, så chunks utover dette havner i samme slot. Bøker med flere chunks gir feil konkordans-pekere uten noen feilmelding.

Endrer du `K_SHIFT` i `collocations_and_trends.py`, må de samme konstantene (`>> 11`, `& 2047`) oppdateres overalt de brukes.

---

## 2. Matematiske invarianter som agenter SKAL bruke

### A. Vektorrom uten fiksert omgivende dimensjon
* **Nullvektor (`PartialFunc.zero()`):** Den tomme funksjonen ($\text{dom}(\mathbf{0}) = \emptyset$).
* **Lineærkombinasjon ($\alpha f + \beta g$):** Definert på **unionen** $\text{dom}(f) \cup \text{dom}(g)$, der $w \notin \text{dom}(f)$ tolkes som $0$.
* **Punktvis produkt ($f * g$) og kvotient ($f / g$):** Definert på **snittet** $\text{dom}(f) \cap \text{dom}(g)$. `f / g` sjekker ikke for $g(w) = 0$ og gir da `inf`/`nan` (jf. eksplisitte nuller over). Restrikter $g$ til ikke-null-verdier først, for eksempel ved relative trendlinjer der $\tau_{\text{total}}(t)$ kan være 0.
* **Restriksjon ($f.\text{restrict}(S)$):** Begrenser $f$ til delmengden $\text{dom}(f) \cap S$ der $S$ er en `BitMap`.

### B. Informasjonsteoretiske distanser KUN over snittet $\text{dom}(P) \cap \text{dom}(Q)$
Når $P$ og $Q$ er sannsynlighetsfordelinger (`p = f.to_prob()`, slik at $\sum_{w \in \text{dom}(P)} P(w) = 1$), skal du **aldri** konstruere unionen $\text{dom}(P) \cup \text{dom}(Q)$ for å beregne divergens eller distanse. Bruk de eksakte snitt-identitetene over $I = \text{dom}(P) \cap \text{dom}(Q)$:

1. **Jensen-Shannon Divergens ($\text{JSD} \in [0, 1]$ bits):**
   $$\text{JSD}(P \,\|\, Q) = 1 + \frac{1}{2} \sum_{w \in I} \left[ P(w) \log_2 \frac{P(w)}{P(w)+Q(w)} + Q(w) \log_2 \frac{Q(w)}{P(w)+Q(w)} \right]$$
2. **Total Variation ($\text{TV} = \frac{1}{2}\|P - Q\|_1 \in [0, 1]$):**
   $$\text{TV}(P, Q) = 1 - \sum_{w \in I} \min\!\big(P(w), Q(w)\big)$$
3. **Hellinger-distanse ($H \in [0, 1]$):**
   $$H(P, Q) = \sqrt{1 - \sum_{w \in I} \sqrt{P(w)\,Q(w)}}$$

### C. 100 % Ettersporbarhet (Dekomponering av skalarer til `PartialFunc`)
Enhver skalar likhet eller divergens i denne modellen er summen av en eksplisitt partiell funksjon:
* **`u.explain_dot(v)`:** Returnerer den partielle funksjonen $h = u * v$ på $\text{dom}(u) \cap \text{dom}(v)$ der $\sum h = \langle u, v \rangle$. Bruk `h.top_k(k)` for å vise nøyaktig hvilke ord som bærer likheten.
* **`p.explain_jsd_divergence(q)`:** Returnerer den partielle funksjonen der hvert ord $w \in \text{dom}(P) \cup \text{dom}(Q)$ har sitt eksakte additive bidrag i bits til $\text{JSD}(P \,\|\, Q)$. Dette er det **eneste tillatte unntaket** fra regelen i 2B: Unionen bygges her fordi forklaringen skal vise bidraget fra hvert ord, også ord som bare finnes i én av fordelingene (de bidrar med $\tfrac{1}{2}P(w)$ eller $\tfrac{1}{2}Q(w)$). Selve skalaren $\text{JSD}$ beregnes fortsatt over snittet.

---

## 3. Ytelsesregler og «Gotchas» med `pyroaring` + `numpy`

> Tidene i denne seksjonen og i 4B er indikative og målt lokalt på én maskin med Shard 0 / DDK 641. De viser hvilken metode som er raskest *relativt*, ikke absolutte garantier. Kjør benchmark-skriptene på nytt før du siterer tall.

1. **Null-kopi uthenting fra `BitMap` til `numpy`:**
   `bm.to_array()` i `pyroaring` returnerer en `array.array('I')`, **ikke** en `np.ndarray` (som gir `TypeError` ved boolsk maskering `arr[mask]`). Pakk den alltid inn null-kopi med:
   ```python
   arr = np.frombuffer(bm.to_array(), dtype=np.uint32)
   ```
   (Dette gjøres automatisk og caches i `pf.dom_array`).
2. **Frekvensnummerering ($1 \dots M$) og Roaring Containere:**
   Nummerer alltid det globale ordforrådet etter **fallende frekvens** (fra `totals_500k.parquet`). Da havner ~88 % av ordene i en bok i Roaring **Container 0** ($\text{ID} < 65\,536$, lagret som en 8 KB SIMD-bitmaske), mens sjeldne ord og OCR-feil ($\ge 65\,536$) havner i kompakte 16-bits array-containere.
3. **Valg av metode for verdi-oppslag ved snitt $I = \text{dom}(f) \cap \text{dom}(g)$:**
   * **Ved 1-mot-mange eller alle-mot-alle sammenligninger:** Bruk [`SparseWorkspace`](partial_func.py) (Gustavsons algoritme). `ws.scatter(f)` legger verdiene til $f$ ut i et flatt arbeidsarray (`work[dom_f] = val_f`), slik at alle påfølgende sammenligninger med $g_1, g_2, \dots$ skjer ved direkte indeksering i `~5.5 µs` per bokpar uten søk!
   * **Ved enkeltstående parvise snitt:** Bruk `f.align_intersection(g, method="gallop")` (`~12 µs`) eller 64-bits `BitBlock + POPCNT` ([`_bitblock_intersect_dot_and_jsd`](partial_func.py), `~7.3 µs`) som finner eksakt indeks i `val` med maskinvare-popcount `offset + popcnt(mask & (lowbit - 1))`.
   * **Unngå** å kalle `[f.dom.rank(w) - 1 for w in inter_bm]` i en Python-løkke når $|I|$ er stor.
4. **Ikke pakk `(word_id << 32) | val` inn i `BitMap64`:**
   Det ødelegger domenesnitt (fordi to bøker med ulike verdier for samme ord får ulike 64-bits tall) og gir 1-elements containere i `BitMap64` (~4× større minnebruk). Hold deg til 32-bit `BitMap` + alignet `ndarray` («Fortran-modellen»), eventuelt [`BSIPartialFunc`](bsi_roaring.py) (Bit-Sliced Roaring) kun når du skal gjøre lynraske `sum_on(S)` (`1.5 µs`) eller terskelsøk på ferdig kvantiserte heltall.

---

## 4. Kobling mot søster-repoene (`doc-to-vec` og `metadata-roaring` / `sqlite-backend`)

### A. Kollokasjoner og trendlinjer uten bokoppslag ([collocations_and_trends.py](collocations_and_trends.py))
I en Roaring-korpusindeks der hvert ord $w$ har $C_w = \text{RM}(\text{chunk\_id})$ (med $\text{chunk\_id} = (\text{doc\_id} \ll 11) \mid \text{local\_chunk}$) eller $D_w = \text{RM}(\text{doc\_id})$, og hvert tidssteg $t$ har en Run-container bitmap $K_t$:
* **Kollokasjonsprofil for en kontekst-bitmap $A$ (f.eks. $C_x \cap K_t$):**
  $$f_A(w) = \texttt{A.intersection\_cardinality(C\_w)} \quad\longrightarrow\quad \texttt{PartialFunc}\text{ i } \mathcal{F}(\text{Words}, \mathbb{R})$$
* **Trendlinje for $A$ over tid:**
  $$\tau_A(t) = \texttt{A.intersection\_cardinality(K\_t)} \quad\longrightarrow\quad \texttt{PartialFunc}\text{ i } \mathcal{F}(\text{Time}, \mathbb{R})$$
* **Relativ trendlinje:** $\tau_{\text{rel}} = \tau_A / \tau_{\text{total}}$ (punktvis kvotient av `PartialFunc`).
* **Diakron PMI-trendlinje:** $\log_2\!\big(\tau_{x \wedge y} / ((\tau_x * \tau_y) / \tau_{\text{total}})\big)$.

### B. Rank-indeksert `word_unigrams` i SQLite ([convert_shard_to_rank.py](convert_shard_to_rank.py))
I stedet for å lagre `unigrams` som et 2D B-tre `PRIMARY KEY (book_id, cf_id) WITHOUT ROWID` (~77,3 millioner rader per shard), brukes «Fortran-modellen» også på posisjonsindeksen:
```sql
CREATE TABLE word_unigrams (
    cf_id     INTEGER PRIMARY KEY,  -- Selve rowid i SQLite B-treet (kun 1 rad per cf_id!)
    offsets   BLOB,                 -- NULL når df==1, ellers uint32[k + 1] byte-offsets
    pos_arena BLOB NOT NULL         -- Alle k posisjons-bitmaps limt kant-i-kant i book_id-rekkefølge
);
```
* **Oppslag for bok `b`:** Finn rangeringen $r = \texttt{docpost.rank(b) - 1}$ og klipp ut `pos_arena[offsets[r] : offsets[r + 1]]` (enten via én samlet `SELECT` eller selektivt med `con.blobopen("word_unigrams", "pos_arena", cf_id, readonly=True)`).
* **Effekt på Shard 0 (8 906 bøker):** Kutter antall B-tre-rader med **84,7 %** (`77,3M` $\to$ `11,8M`), reduserer filstørrelsen fra **6,66 GB til 4,90 GB** (-26,4 %, og sparer i tillegg 1,90 GB chunk-sidecar), og gjør `NEAR_5` **8×–37× raskere** enn det gamle B-treet.

### C. Filer i dette repoet
* [partial_func.py](partial_func.py) — Kjerneklassen `PartialFunc`, Numba-kjerner (`gallop`, `twopointer`, `bitblock_popcnt`), `SparseWorkspace`, PMI, JSD/TV/Hellinger og ettersporbar dekomponering.
* [test_mat_and_totals.py](test_mat_and_totals.py) — Innlasting av `totals_500k.parquet` og `mat_bundle.json`, verifikasjon av vektorromsaksiomer og benchmark av snitt-metodene.
* [collocations_and_trends.py](collocations_and_trends.py) — `RoaringCorpusIndex` for ren Roaring-aggregering av kollokasjoner, diakron drift og trendlinjer (testet på både DDK 641 og Shard 0 med 8 906 bøker).
* [codec_and_traceability.py](codec_and_traceability.py) — `AlgebraCodec` med full pipeline fra inn-koding via funksjons-algebra til ettersporbar dekoding av topp-ord og konkordans-pekere.
* [bsi_roaring.py](bsi_roaring.py) — Referanseimplementasjon og benchmark av Bit-Sliced Roaring (`BSIPartialFunc`).
* [bench_rank_unigrams.py](bench_rank_unigrams.py) — Benchmark av klassisk `(book_id, cf_id)` B-tre mot Rank-indeksert `word_unigrams`.
* [convert_shard_to_rank.py](convert_shard_to_rank.py) — 2-pass bøtte-konverterer som bygger `imag_roaring_rank_0.db` fra en eksisterende Roaring-shard på ~4,5 minutter.

---

## 5. Data, lokale filer og kommandoer

### Data utenfor repoet
* `totals_500k.parquet` (global ordfrekvens, grunnlaget for ID-nummereringen i 3.2) og `mat_bundle.json` (DDK 641) ligger i søsterrepoet `doc-to-vec`. Stien er hardkodet som `DOC_TO_VEC_DIR` i `test_mat_and_totals.py`. Endre den der hvis repoet ligger et annet sted. Ikke kopier filene hit.
* `*.db`, `*.parquet`, `*.sqlite` og `*.tsv` er gitignored. `imag_roaring_rank_0.db` (≈ 5 GB) er en lokal byggefil fra `convert_shard_to_rank.py`. Den skal aldri committes og bør ikke bygges på nytt uten grunn (≈ 4,5 min).

### Oppsett
Avhengighetene står i `pyproject.toml` (numba, numpy, pandas, pyarrow, pyroaring, scipy, Python ≥ 3.11). Repoet har flat modulstruktur og ingen `[build-system]`, så installer avhengighetene direkte i et virtuelt miljø:
```bash
python -m venv .venv && source .venv/bin/activate
pip install "numba>=0.60" "numpy>=2.0" "pandas>=2.0" "pyarrow>=15.0" "pyroaring>=1.0" "scipy>=1.13"
```

### Kjøring
Det finnes ingen testrammeverk. Hver modul kjøres direkte og skriver ut verifikasjon og benchmark:
* `python test_mat_and_totals.py`: vektorromsaksiomer og snitt-metoder. Krever dataene fra `doc-to-vec`.
* `python codec_and_traceability.py` og `python bsi_roaring.py`: demoer og benchmarks for de enkelte delene.
* `python collocations_and_trends.py` og `python bench_rank_unigrams.py`: krever Shard 0 og metadata fra søsterrepoet `metadata-roaring` (`shards/imag_roaring_main_0.db`, `metadata_core_books.sqlite`). Stiene er hardkodet øverst i skriptene.
* `python convert_shard_to_rank.py --help`: viser argumentene for konvertering av en eksisterende Roaring-shard.
