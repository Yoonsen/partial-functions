# Handoff til Agent på `dhlab1`: Rank-indeksert `word_unigrams` («Fortran-modellen»)

Dette dokumentet beskriver neste steg etter `HANDOFF_SQLITE_BACKEND_ROARING_AND_CHUNKS.md`: hvordan man bytter ut 2D B-treet `unigrams (book_id, cf_id)` med en Rank-indeksert tabell `word_unigrams (cf_id INTEGER PRIMARY KEY)` styrt av `docpost.rank(book_id)`.

---

## 1. Hvorfor gjøre dette? (Målt på full Shard 0: 8 906 bøker)

I dagens hovedshard (`imag_roaring_main_0.db`, `6,66 GB`) har `unigrams` **77 303 348 rader** med `PRIMARY KEY (book_id, cf_id) WITHOUT ROWID`.
Når et søk som `NEAR` eller `sequence` skal sjekke $K$ kandidatbøker for to ord, må SQLite gjøre $2K$ tilfeldige B-tre-oppslag på kryss og tvers i en 6,7 GB fil.

Ved å snu tabellen til **1 rad per `cf_id`** (der alle bøkenes posisjons-bitmaps for `cf_id` ligger limt kant-i-kant i `pos_arena`, indeksert via `r = docpost.rank(book_id) - 1`):

1. **84,7 % færre B-tre-rader:** `77 303 348` rader $\longrightarrow$ **`11 794 233` rader** (kun 1 rad per `cf_id`).
2. **72,9 % av alle `cf_id` forekommer kun i 1 bok (`df = 1`):** For disse settes `offsets = NULL` (hele `pos_arena` er bokas posisjons-bitmap).
3. **26,4 % mindre hovedshard:** `6,66 GB` $\longrightarrow$ **`4,90 GB`** (sparer `1,76 GB` per shard alene i hoveddatabasen).
4. **Chunk-sidecar blir valgfri (sparer ytterligere `1,90 GB` per shard):**
   Fordi det kun gjøres **1 B-tre-oppslag per søkeord** (i stedet for ett per bok), går `NEAR_5` direkte på `word_unigrams` *uten* chunk-sidecar i **1–90 ms** — raskere enn det gamle B-treet var *med* chunk-sidecar!

| Ordpar | `Doc∩` | Gamle `unigrams` (`6,66 GB`) | **Nye `word_unigrams` (`4,90 GB`, uten sidecar)** | **Speedup** | Eksakte `NEAR_5` treff |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `penger + kjerlighed` | 21 | `24,7 ms` | **`1,2 ms`** | **21,5×** | 0 |
| `demokrati + frihed` | 264 | `221,1 ms` | **`6,0 ms`** | **36,9×** | 8 |
| `penger + kongen` | 351 | `275,8 ms` | **`8,5 ms`** | **32,6×** | 16 |
| `frihed + kongen` | 3 088 | `1 661,0 ms` | **`59,7 ms`** | **27,8×** | 161 |
| `norge + kongen` | 2 870 | `804,8 ms` | **`52,7 ms`** | **15,3×** | 1 331 |
| `gud + kongen` | 3 596 | `524,8 ms` | **`64,8 ms`** | **8,1×** | 1 329 |
| `lov + ret` | 6 042 | `2 243,0 ms` | **`90,6 ms`** | **24,8×** | 3 079 |

---

## 2. Databasedesign (`word_unigrams`)

```sql
CREATE TABLE word_unigrams (
    cf_id     INTEGER PRIMARY KEY,  -- Selve rowid i SQLite B-treet!
    offsets   BLOB,                 -- NULL når df == 1, ellers uint32[k + 1] byte-offsets (little-endian)
    pos_arena BLOB NOT NULL         -- Alle k posisjons-bitmaps (Roaring) limt kant-i-kant i stigende book_id-rekkefølge
);
```

### Invariant mellom `words.docpost` og `word_unigrams`
For enhver `cf_id` inneholder `docpost` (etter ev. komplement-inversjon `all_books_bm - bm` når `docpost_is_complement == 1`) de eksakt $k$ `book_id`-ene som har posisjoner i `pos_arena`, i **strengt stigende `book_id`-rekkefølge**:
* Hvis `offsets IS NULL` ($k = 1$): Bokas posisjons-blob er hele `pos_arena`.
* Hvis `offsets IS NOT NULL` ($k > 1$): `offsets` er et `uint32`-array av lengde $k + 1$ (`np.frombuffer(offsets, dtype=np.uint32)`).
  For enhver bok $b \in \text{docpost}$:
  $$r = \text{docpost}.\text{rank}(b) - 1$$
  $$\text{pos\_blob}(b) = \text{pos\_arena}\big[\,\text{offsets}[r] : \text{offsets}[r + 1]\,\big]$$

### Konvertering av eksisterende sharder
Bruk [convert_shard_to_rank.py](file:///Users/larsj/Github/partial-functions/convert_shard_to_rank.py) direkte. Den strømmer `unigrams` 100 % sekvensielt i `book_id`-rekkefølge ut til 32 midlertidige bøttefiler (krever kun ~200 MB RAM) og bygger `word_unigrams` ferdig sortert på `cf_id` på ca. 4–5 minutter per shard:
```bash
python convert_shard_to_rank.py \
  --src /sti/til/imag_roaring_main_0.db \
  --dst /sti/til/imag_roaring_rank_0.db \
  --tmp-dir /sti/til/_tmp_buckets
```

---

## 3. Drop-in oppslagskode for `api_python/postings_queries.py`

I stedet for å kalle `SELECT post FROM unigrams WHERE book_id = ? AND cf_id = ?` inne i en løkke over `candidate_books`, åpnes hver `cf_id` **én gang** før bok-løkken:

```python
from typing import Dict, List, Optional, Sequence, Tuple
import sqlite3
import numpy as np
from pyroaring import BitMap as RoaringBitMap


class RankWordPositions:
    """
    Holder (doc_bm, offsets, pos_arena / blobopen-handle) for én cf_id,
    og henter posisjons-BitMap for enhver book_id i O(1) via doc_bm.rank(book_id).
    """

    __slots__ = ("cf_id", "doc_bm", "offsets", "_mv", "_blob_handle")

    def __init__(
        self,
        con: sqlite3.Connection,
        cf_id: int,
        doc_bm: RoaringBitMap,
        *,
        use_blobopen: bool = False,
    ) -> None:
        self.cf_id = int(cf_id)
        self.doc_bm = doc_bm
        self._mv: Optional[memoryview] = None
        self._blob_handle: Optional[sqlite3.Blob] = None

        if use_blobopen:
            row = con.execute(
                "SELECT offsets FROM word_unigrams WHERE cf_id = ?", (self.cf_id,)
            ).fetchone()
            self.offsets = (
                np.frombuffer(row[0], dtype=np.uint32) if (row and row[0]) else None
            )
            self._blob_handle = con.blobopen(
                "word_unigrams", "pos_arena", self.cf_id, readonly=True
            )
        else:
            row = con.execute(
                "SELECT offsets, pos_arena FROM word_unigrams WHERE cf_id = ?",
                (self.cf_id,),
            ).fetchone()
            if not row:
                self.offsets = None
                self._mv = None
            else:
                off_b, arena_b = row
                self.offsets = (
                    np.frombuffer(off_b, dtype=np.uint32) if off_b else None
                )
                self._mv = memoryview(arena_b)

    def get_positions_bitmap(self, book_id: int) -> Optional[RoaringBitMap]:
        if book_id not in self.doc_bm:
            return None
        if self.offsets is None:
            # df == 1: Hele pos_arena tilhører denne ene boka
            raw = self._blob_handle[:] if self._blob_handle is not None else self._mv
            return RoaringBitMap.deserialize(raw) if raw is not None else None

        r = self.doc_bm.rank(book_id) - 1
        start = int(self.offsets[r])
        end = int(self.offsets[r + 1])
        if self._blob_handle is not None:
            return RoaringBitMap.deserialize(self._blob_handle[start:end])
        return RoaringBitMap.deserialize(self._mv[start:end])

    def close(self) -> None:
        if self._blob_handle is not None:
            self._blob_handle.close()
            self._blob_handle = None
```

### Tips for valg av `use_blobopen`
* Hvis antall overlevende bøker `len(candidate_books)` er **lite** i forhold til ordets totale dokumentfrekvens `len(doc_bm)` (f.eks. `< 100` bøker for et høyfrekvent ord som `"kongen"` eller `"."`, typisk ved `2-års filter` eller `presampling`), sett `use_blobopen = (len(candidate_books) * 8 < len(doc_bm))`. Da leser SQLite **kun** de få 4 KB-sidene i `pos_arena` som trengs!
* Ellers (`use_blobopen = False`) leser `SELECT offsets, pos_arena` hele stripen sekvensielt i ett kall med `memoryview` uten kopiering.
