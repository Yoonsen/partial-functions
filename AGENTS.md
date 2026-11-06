# AGENTS.md — AI Technical Handover & Architecture Contract (`partial-functions`)

Dette dokumentet er skrevet for AI-agenter (Antigravity, Claude, Cursor, m.fl.) som jobber i dette repoet eller bruker `partial-functions` som matematisk og algoritmisk grunnmur i tilknyttede prosjekter (som `doc-to-vec`, `metadata-roaring` og `sqlite-backend`).

---

## 1. Kjernefilosofi: «Fortran-modellen» for partielle funksjoner

I stedet for å modellere tekstvektorer, kollokasjoner og tidsserier som fikserte $N$-tupler i $\mathbb{R}^N$ (fulle av nuller) eller som trege Python-`dict`/`set`, representeres alt som **partielle funksjoner med endelig støtte**:

$$f : D \rightharpoonup \mathbb{R}$$

### Datastrukturen (`PartialFunc` i [partial_func.py](file:///Users/larsj/Github/partial-functions/partial_func.py))
Hver partiell funksjon består av to stramt alignede komponenter (Structure of Arrays / «Fortran-modellen»):
1. **`dom : pyroaring.BitMap`** — En 32-bits Roaring Bitmap (`uint32`) som representerer definisjonsområdet $\text{dom}(f) \subseteq \mathbb{N}$ (f.eks. globale ord-ID-er $1 \dots M$, dokument-ID-er, chunk-ID-er eller årstall).
2. **`val : np.ndarray[float64]`** — Et kontinuerlig flyttalls-array av lengde $|\text{dom}(f)|$, sortert i **eksakt samme rekkefølge** som elementene i `dom`. Det $i$-te satte bitet i `dom` har sin verdi i `val[i]`.

### Den tredelte arbeidsflyten: Koding $\to$ Funksjons-algebra $\to$ Dekoding
1. **Koding (Inn):** Ord (strenger), metadata-uttrykk og tidsrom kodes **én gang** ved inngangen til heltalls-ID-er, `BitMap` og `PartialFunc` (se [`AlgebraCodec`](file:///Users/larsj/Github/partial-functions/codec_and_traceability.py#L34-L115)).
2. **Funksjons-algebra (Midt):** Alle beregninger skjer lukket inne i algebraen over `(BitMap, ndarray)` — helt uten strenger, uten SQL-joins og uten oppslag i enkeltbøker.
3. **Ettersporbar dekoding (Ut):** Kun sluttresultatet (f.eks. `pf.top_k(10)` eller konkordans-pekere `doc_id = c >> 11, chunk = c & 2047`) dekodes tilbake til tekst.

---

## 2. Matematiske invarianter som agenter SKAL bruke

### A. Vektorrom uten fiksert omgivende dimensjon
* **Nullvektor (`PartialFunc.zero()`):** Den tomme funksjonen ($\text{dom}(\mathbf{0}) = \emptyset$).
* **Lineærkombinasjon ($\alpha f + \beta g$):** Definert på **unionen** $\text{dom}(f) \cup \text{dom}(g)$, der $w \notin \text{dom}(f)$ tolkes som $0$.
* **Punktvis produkt ($f * g$) og kvotient ($f / g$):** Definert på **snittet** $\text{dom}(f) \cap \text{dom}(g)$.
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
* **`p.explain_jsd_divergence(q)`:** Returnerer den partielle funksjonen der hvert ord $w \in \text{dom}(P) \cup \text{dom}(Q)$ har sitt eksakte additive bidrag i bits til $\text{JSD}(P \,\|\, Q)$.

---

## 3. Ytelsesregler og «Gotchas» med `pyroaring` + `numpy`

1. **Null-kopi uthenting fra `BitMap` til `numpy`:**
   `bm.to_array()` i `pyroaring` returnerer en `array.array('I')`, **ikke** en `np.ndarray` (som gir `TypeError` ved boolsk maskering `arr[mask]`). Pakk den alltid inn null-kopi med:
   ```python
   arr = np.frombuffer(bm.to_array(), dtype=np.uint32)
   ```
   (Dette gjøres automatisk og caches i `pf.dom_array`).
2. **Frekvensnummerering ($1 \dots M$) og Roaring Containere:**
   Nummerer alltid det globale ordforrådet etter **fallende frekvens** (fra `totals_500k.parquet`). Da havner ~88 % av ordene i en bok i Roaring **Container 0** ($\text{ID} < 65\,536$, lagret som en 8 KB SIMD-bitmaske), mens sjeldne ord og OCR-feil ($\ge 65\,536$) havner i kompakte 16-bits array-containere.
3. **Valg av metode for verdi-oppslag ved snitt $I = \text{dom}(f) \cap \text{dom}(g)$:**
   * **Ved 1-mot-mange eller alle-mot-alle sammenligninger:** Bruk [`SparseWorkspace`](file:///Users/larsj/Github/partial-functions/partial_func.py#L652-L680) (Gustavsons algoritme). `ws.scatter(f)` legger verdiene til $f$ ut i et flatt arbeidsarray (`work[dom_f] = val_f`), slik at alle påfølgende sammenligninger med $g_1, g_2, \dots$ skjer ved direkte indeksering i `~5.5 µs` per bokpar uten søk!
   * **Ved enkeltstående parvise snitt:** Bruk `f.align_intersection(g, method="gallop")` (`~12 µs`) eller 64-bits `BitBlock + POPCNT` ([`_bitblock_intersect_dot_and_jsd`](file:///Users/larsj/Github/partial-functions/partial_func.py#L223-L276), `~7.3 µs`) som finner eksakt indeks i `val` med maskinvare-popcount `offset + popcnt(mask & (lowbit - 1))`.
   * **Unngå** å kalle `[f.dom.rank(w) - 1 for w in inter_bm]` i en Python-løkke når $|I|$ er stor.
4. **Ikke pakk `(word_id << 32) | val` inn i `BitMap64`:**
   Det ødelegger domenesnitt (fordi to bøker med ulike verdier for samme ord får ulike 64-bits tall) og gir 1-elements containere i `BitMap64` (~4× større minnebruk). Hold deg til 32-bit `BitMap` + alignet `ndarray` («Fortran-modellen»), eventuelt [`BSIPartialFunc`](file:///Users/larsj/Github/partial-functions/bsi_roaring.py) (Bit-Sliced Roaring) kun når du skal gjøre lynraske `sum_on(S)` (`1.5 µs`) eller terskelsøk på ferdig kvantiserte heltall.

---

## 4. Kobling mot søster-repoene (`doc-to-vec` og `metadata-roaring` / `sqlite-backend`)

### A. Kollokasjoner og trendlinjer uten bokoppslag ([collocations_and_trends.py](file:///Users/larsj/Github/partial-functions/collocations_and_trends.py))
I en Roaring-korpusindeks der hvert ord $w$ har $C_w = \text{RM}(\text{chunk\_id})$ (med $\text{chunk\_id} = (\text{doc\_id} \ll 11) \mid \text{local\_chunk}$) eller $D_w = \text{RM}(\text{doc\_id})$, og hvert tidssteg $t$ har en Run-container bitmap $K_t$:
* **Kollokasjonsprofil for en kontekst-bitmap $A$ (f.eks. $C_x \cap K_t$):**
  $$f_A(w) = \texttt{A.intersection\_cardinality(C\_w)} \quad\longrightarrow\quad \texttt{PartialFunc}\text{ i } \mathcal{F}(\text{Words}, \mathbb{R})$$
* **Trendlinje for $A$ over tid:**
  $$\tau_A(t) = \texttt{A.intersection\_cardinality(K\_t)} \quad\longrightarrow\quad \texttt{PartialFunc}\text{ i } \mathcal{F}(\text{Time}, \mathbb{R})$$
* **Relativ trendlinje:** $\tau_{\text{rel}} = \tau_A / \tau_{\text{total}}$ (punktvis kvotient av `PartialFunc`).
* **Diakron PMI-trendlinje:** $\log_2\!\big(\tau_{x \wedge y} / ((\tau_x * \tau_y) / \tau_{\text{total}})\big)$.

### B. Filer i dette repoet
* [partial_func.py](file:///Users/larsj/Github/partial-functions/partial_func.py) — Kjerneklassen `PartialFunc`, Numba-kjerner (`gallop`, `twopointer`, `bitblock_popcnt`), `SparseWorkspace`, PMI, JSD/TV/Hellinger og ettersporbar dekomponering.
* [test_mat_and_totals.py](file:///Users/larsj/Github/partial-functions/test_mat_and_totals.py) — Innlasting av `totals_500k.parquet` og `mat_bundle.json`, verifikasjon av vektorromsaksiomer og benchmark av snitt-metodene.
* [collocations_and_trends.py](file:///Users/larsj/Github/partial-functions/collocations_and_trends.py) — `RoaringCorpusIndex` for ren Roaring-aggregering av kollokasjoner, diakron drift og trendlinjer (testet på både DDK 641 og Shard 0 med 8 906 bøker).
* [codec_and_traceability.py](file:///Users/larsj/Github/partial-functions/codec_and_traceability.py) — `AlgebraCodec` med full pipeline fra inn-koding via funksjons-algebra til ettersporbar dekoding av topp-ord og konkordans-pekere.
* [bsi_roaring.py](file:///Users/larsj/Github/partial-functions/bsi_roaring.py) — Referanseimplementasjon og benchmark av Bit-Sliced Roaring (`BSIPartialFunc`).
