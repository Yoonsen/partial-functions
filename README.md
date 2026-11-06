# Partielle Funksjoner som Vektorrom og Distribusjoner over Roaring Bitmaps

Dette repoet utforsker en algebraisk og algoritmisk modell der ordlister, kollokasjoner og bøker representeres som **partielle funksjoner**

$$f : \text{Words} \rightharpoonup \mathbb{R}$$

med endelig støtte ($\text{dom}(f)$), **uten** å gå veien om fikserte $N$-tupler i $\mathbb{R}^N$.

---

## 1. Den matematiske historien: Vektorrom og distribusjoner uten tupler

I klassisk «bag-of-words» eller lineæralgebra tvinges man til å definere et fast univers av ord $W = \{w_1, \dots, w_N\}$ på forhånd, slik at en tekst blir en $N$-tuppel $(x_1, \dots, x_N) \in \mathbb{R}^N$ der nesten alle koordinater er $0$.

Hvis vi i stedet starter direkte med mengden $\mathcal{F}(W, \mathbb{R})$ av **partielle funksjoner med endelig domene**:

* En partiell funksjon $f$ er fullstendig gitt ved sitt domene $\text{dom}(f)$ (en endelig ordmengde) og sine verdier $f(w)$ for $w \in \text{dom}(f)$.
* **Vektorromsaksiomene (over $\mathbb{R}$):**
  * **Nullvektoren $\mathbf{0}$:** Den tomme funksjonen ($\text{dom}(\mathbf{0}) = \emptyset$).
  * **Addisjon $(f + g)$:** Definert på unionen $\text{dom}(f) \cup \text{dom}(g)$, der $w \notin \text{dom}(f)$ bidrar med $0$.
  * **Skalarmultiplikasjon $(c \cdot f)$:** Definert på $\text{dom}(f)$ ved $(cf)(w) = c \cdot f(w)$ (og $\emptyset$ for $c = 0$).
  * Dette oppfyller alle de 8 aksiomene for et vektorrom over $\mathbb{R}$ — helt uten at et omgivende univers $W$ eller en dimensjon $N$ trenger å være fiksert på forhånd!

### Fra geometriske vinkler til PMI og distribusjonsdistanser (kun over snittet!)

Når $f$ og $g$ starter som tellinger (multisets), induserer hver funksjon en sannsynlighetsfordeling $P_f(w) = \frac{f(w)}{\|f\|_1}$ på $\text{dom}(f)$.

1. **PMI som partiell funksjonstransformasjon:**
   Gitt en referansefordeling $P_b$ (f.eks. `nb.totals`), er Shifted (P)PMI en partiell funksjon definert på snittet $\text{dom}(f) \cap \text{dom}(b)$:
   $$\text{PMI}(f \,\|\, b)(w) = \log_2 \frac{P_f(w)}{P_b(w)^\alpha} - \log_2 k$$

2. **Alle sentrale distribusjonsdistanser reduserer til en sum over *snittet* $\text{dom}(P) \cap \text{dom}(Q)$:**
   Det er en vanlig misforståelse at informasjonsteoretiske avstander som Jensen-Shannon Divergens (JSD) eller $L_1$-avstand (Total Variation) krever at man itererer over *unionen* $\text{dom}(P) \cup \text{dom}(Q)$. Siden $\sum_{w \in \text{dom}(P)} P(w) = 1$ og $\sum_{w \in \text{dom}(Q)} Q(w) = 1$, kansellerer bidragene fra $\text{dom}(P) \setminus \text{dom}(Q)$ og $\text{dom}(Q) \setminus \text{dom}(P)$ eksakt:

   * **Jensen-Shannon Divergens ($\text{JSD} \in [0, 1]$ bits):**
     $$\text{JSD}(P \,\|\, Q) = 1 + \frac{1}{2} \sum_{w \in \text{dom}(P) \cap \text{dom}(Q)} \left[ P(w) \log_2 \frac{P(w)}{P(w)+Q(w)} + Q(w) \log_2 \frac{Q(w)}{P(w)+Q(w)} \right]$$
   * **Total Variation ($\text{TV} = \frac{1}{2}\|P - Q\|_1 \in [0, 1]$):**
     $$\text{TV}(P, Q) = 1 - \sum_{w \in \text{dom}(P) \cap \text{dom}(Q)} \min\!\big(P(w), Q(w)\big)$$
   * **Hellinger-distanse ($H \in [0, 1]$):**
     $$H(P, Q) = \sqrt{1 - \sum_{w \in \text{dom}(P) \cap \text{dom}(Q)} \sqrt{P(w)\,Q(w)}}$$

Dermed styres **både** indreprodukt, PMI, JSD, Total Variation og Hellinger av én og samme kjerneoperasjon: **finn snittet $I = \text{dom}(f) \cap \text{dom}(g)$ og hent ut verdiene $f|_I$ og $g|_I$**.

---

## 2. Datastruktur: `dom(f)` som Roaring Bitmap + alignet `val(f)`

Vi nummererer ordene med heltall $1, 2, \dots, M$ sortert etter fallende global frekvens (`totals_500k.parquet`).
En partiell funksjon representeres i [partial_func.py](file:///Users/larsj/Github/partial-functions/partial_func.py) som:
* `dom: pyroaring.BitMap` — delmengden av ord-ID-er i $f$.
* `val: np.ndarray[float64]` — verdier av lengde $|\text{dom}(f)|$, alignet med den sorterte rekkefølgen i `dom`.

### Hvorfor frekvensnummerering spiller på lag med Roaring Bitmaps
Roaring Bitmaps deler heltallsrommet i blokker (containere) på $2^{16} = 65\,536$. Når ord-ID $1 \dots 65\,535$ er de vanligste ordene i språket, havner i snitt **88,1 %** av en boks unike ord i **Container 0** (tett 64-bits bitmaske), mens sjeldne ord og OCR-feil ($> 65\,536$) havner i kompakte 16-bits array-containere.

### 5 metoder for å hente verdiene i `val(f)` og `val(g)` ved snitt $I = \text{dom}(f) \cap \text{dom}(g)$

| Metode | Prinsipp | Tid per bokpar (DDK 641) | Når bruke hva? |
| :--- | :--- | ---: | :--- |
| **1. `bm.rank(w)`** | Roarings innebygde $\text{idx} = \text{dom}(f).\text{rank}(w) - 1$ | `51.4 µs` | Enkeltoppslag $f(w)$ for ett eller noen få ord. |
| **2. Roaring `&` + Galloping** | $I = \text{dom}(f) \cap \text{dom}(g)$ i C/SIMD + eksponentielt søk i `dom_array` | `13.5 µs` | Når snittet $\|I\|$ er svært lite i forhold til $\|f\|$ og $\|g\|$. |
| **3. Roaring `&` + Two-pointer** | Lineær to-peker fletting over de sorterte domenene | `11.8 µs` | Enkel, allokasjonsfri parvis sammenligning. |
| **4. 64-bit BitBlock + `POPCNT`** | $\text{idx}_f = \text{offset}_f + \text{popcnt}(\text{mask}_f \mathbin{\&} (\text{lowbit} - 1))$ | **`7.3 µs`** | **0 søk og 0 arbeidsminne!** Direkte maskinvare-indeksering under bit-AND. |
| **5. `SparseWorkspace` (Gustavson)** | `work[dom_f] = val_f` $\rightarrow$ direkte oppslag `work[dom_g]` | **`5.5 µs`** | **Raskest for 1-mot-mange / alle-mot-alle** (56 953 bokpar på 383 ms). |

---

## 3. Kollokasjoner og trendlinjer som partielle funksjoner (0 bokoppslag)

I [collocations_and_trends.py](file:///Users/larsj/Github/partial-functions/collocations_and_trends.py) kobles Roaring-indekser (både på dokument- og chunk-nivå) direkte til `PartialFunc`-algebraen uten ett eneste oppslag i enkeltbøker (`unigrams`):

### A. Ett felles funksjonsrom for bøker, delkorpus og kollokasjoner: $\mathcal{F}(\text{Words}, \mathbb{R})$
La $C_w = \text{RM}(\text{chunk\_id})$ (eller $\text{RM}(\text{doc\_id})$) være de inverterte bitmappene for hvert ord $w$, og la $K_t$ være bitmappet for en tidsperiode eller et delkorpus.
Enhver kontekst eller spørring $A$ (f.eks. et enkeltord $C_x$, et ordpar $C_x \cap C_y$, eller et ord i en epoke $C_x \cap K_t$) er selv en Roaring Bitmap.

* **Kollokasjonsfunksjonen $f_A : \text{Words} \rightharpoonup \mathbb{R}$** oppstår direkte fra `intersection_cardinality`:
  $$f_A(w) = |A \cap C_w| = \texttt{A.intersection\_cardinality(C\_w)}$$
* **Kollokasjons-PMI** blir da en ren kvotient i `PartialFunc`-rommet mellom $P(w \mid A) = f_A / |A|$ og $P(w \mid K) = f_K / |K|$:
  ```python
  ratio_pf = (f_A / len(A)) / ((f_K / len(K)) ** alpha)
  pmi_pf = PartialFunc(ratio_pf.dom, np.log2(ratio_pf.val))
  ```
* **Diakron semantisk drift:** Sammenlign kollokasjonsprofilen til et ord i to tidsepoker ($A_1 = C_x \cap K_{\le 1940}$ og $A_2 = C_x \cap K_{\ge 1970}$) direkte med:
  * Globalt mål på endring: `f_A1.to_prob().jsd(f_A2.to_prob())` (i bits, kun over snittet!).
  * Hvilke kollokater som har vokst/falt mest: vektordifferansen `pmi_A2 - pmi_A1` i `PartialFunc`-rommet!

### B. Trendlinjer som partielle funksjoner over tid: $\mathcal{F}(\text{Time}, \mathbb{R})$
Gitt tids-bitmaps $K_t$ for hvert år eller tiår $t \in \text{Time}$, er trendlinjen til en hvilken som helst bitmap $A$ (ord, frase eller kollokasjon $C_x \cap C_y$) en partiell funksjon:
$$\tau_A : \text{Time} \rightharpoonup \mathbb{R}, \qquad \tau_A(t) = |A \cap K_t|$$
* **Relativ frekvens over tid:** Punktvis kvotient $\tau_{\text{rel}} = \tau_A / \tau_{\text{total}}$ (automatisk begrenset til $\text{dom}(\tau_A) \cap \text{dom}(\tau_{\text{total}})$).
* **Diakron PMI-trendlinje mellom $x$ og $y$:**
  $$\text{PMI}_{x,y}(t) = \log_2 \!\left(\frac{\tau_{x \wedge y}}{(\tau_x * \tau_y) / \tau_{\text{total}}}\right)(t)$$
* **Historisk samvariasjon mellom to ord/begreper:** `tau_x.to_prob().jsd(tau_y.to_prob())` måler om to ord følger samme historiske konjunktur (f.eks. `JSD(hvitløk, olivenolje) = 0.039 bits` mot `JSD(hvitløk, potetesmel) = 0.764 bits`).

---

## 4. Alternativ: 100 % Roaring via Bit-Sliced Indexing (`BSIPartialFunc`)

I [bsi_roaring.py](file:///Users/larsj/Github/partial-functions/bsi_roaring.py) testes også **Bit-Sliced Roaring (BSI)**, der selv verdiene lagres som en stabel av $B$ stk. 32-bits Roaring Bitmaps (`slices[0..B-1]`) i stedet for et flyttalls-array:

| Operasjon | `PartialFunc` (`BitMap` + `float[]`) | `BSIPartialFunc` (6 bitplan, 100 % Roaring) | Vinner & Hvorfor |
| :--- | ---: | ---: | :--- |
| **Minne per bok (PPMI)** | **3 252 B** | 3 743 B (6-bit) / 4 694 B (8-bit) | **`PartialFunc`** (ett bitmap + tett array) |
| **Sum over ordgalakse $\sum_{w \in S} f(w)$** | 5,05 µs | **1,54 µs** | **`BSI` (3,3× raskere)** — kun 6 `intersection_cardinality`-kall! |
| **Terskelsøk $\{w \mid f(w) \ge T\}$** | 19,18 µs | **13,48 µs** | **`BSI`** — top-down `AND`/`OR` gir ferdig `BitMap`. |
| **Restriksjon $f\|_S$** | **3,80 µs** | 6,68 µs | **`PartialFunc`** — kun 1 bitmap-snitt i stedet for $B+1$. |
| **Parvis $\langle f, g \rangle$ / JSD / PMI** | **5,5–15,0 µs** | 44,6 µs (6-bit) / 67,8 µs (8-bit) | **`PartialFunc`** — $O(B^2)$ kryss-snitt i BSI vs. lineært array-sveip. |

---

## 5. Kjøre tester og benchmarks

```bash
.venv/bin/python test_mat_and_totals.py
.venv/bin/python collocations_and_trends.py
.venv/bin/python codec_and_traceability.py
.venv/bin/python bsi_roaring.py
```
