# פרויקט דה-דופליקציה של מוצרים ואיחוד מחירים

מטלת בית לתפקיד **GenAI Exploration Lead** בקבוצת זאפ.<br> המשימה: לבנות
סקריפט שמאחד רישומי מוצרים כפולים (שיכולים להיות בעברית, באנגלית, או
תערובות מתועתקות של שתיהן) ומבטיח שהמחיר הזול ביותר מוצג עבור כל מוצר
מאוחד.

> **למי שבודק את ההגשה:**
> - `README.md` - תמונת המצב המלאה ב-2-3 דקות.
> - [APPROACH.md](APPROACH.md) - גרסת ההגשה הקצרה עם ההחלטות העיקריות.
> - [APPROACH-extended.md](APPROACH-extended.md) - גרסת ההרחבה עם גרפים, דוגמאות והיסטוריית איטרציות.

## תקציר מנהלים

- הפתרון הוא `pipeline` היברידי: `embeddings` בשביל הסינון הזול והסקיילבילי, ו-`LLM` רק בשביל המקרים המעורפלים.
- שלב 1 רץ מקומית עם `paraphrase-multilingual-mpnet-base-v2`; שלב 2 משתמש
  ב-`gpt-4o-mini`; שלב 3 הוא איחוד מחירים דטרמיניסטי.
- ברירת המחדל היא v2 - גרסה עם שערי ביטחון שמדלגת על קריאות `LLM` מיותרות.
- על הדאטהסט הסינתטי (131 רישומים, 33 מוצרים קנוניים, 6 קטגוריות), v2
  מגיע ל-`F1` מעט טוב יותר מ-v1 תוך חיסכון של כ-22% בקריאות `API`.

## הרצה מהירה

```bash
pip install -r requirements.txt
echo "OPENAI_API_KEY=sk-..." > .env
python main.py
```

הסקריפט קורא את [data/products.csv](data/products.csv) (131 רישומים מתוך 33
מוצרים קנוניים ב-6 קטגוריות), מריץ את ה-`pipeline` בן שלושת השלבים, וכותב את
הקובץ [data/deduplicated_products.csv](data/deduplicated_products.csv) - שורה
אחת לכל מוצר מאוחד עם `min_price`, `max_price` ו-`savings_pct`. בנוסף הוא כותב
את [data/deduplicated_products.audit.json](data/deduplicated_products.audit.json)
כקובץ explainability משלים.

ריצה טיפוסית משתמשת ב-~57 קריאות `API` של `OpenAI` (≈ `$0.005` במודל
`gpt-4o-mini`) ומסתיימת הרבה מתחת לדקה.

## ה-`pipeline` במבט אחד

```
data/products.csv  (131 listings)
        |
        v
[Stage 1: Embeddings]   ──  multilingual sentence-transformer + cosine similarity
        |                   pairwise (0.87) + centroid (0.65), per-category
        v
[Stage 2: LLM refinement v2]  ── gpt-4o-mini, confidence-gated
        |                       Pass 1: split mixed-variant clusters
        |                       Pass 2: match leftover singletons
        |                       Pass 3: merge Hebrew/English siblings
        v
[Stage 3: Consolidation]  ──  group-by + min(price), pick canonical name
        |
        v
data/deduplicated_products.csv  (~30 unified products)
```

לכל שלב יש תפקיד ברור וחוזה קלט/פלט ברור. ההיגיון המלא מאחורי כל החלטה נמצא
ב-[APPROACH-extended.md](APPROACH-extended.md).

## למה העיצוב הזה

שתי החלטות מפתח שכדאי להדגיש:

1. **שילוב של `embeddings` ו-`LLM`, לא `LLM` בלבד.**<br> לשלוח כל זוג ל-`LLM`
   היה עובד אבל זה עולה `O(n²)` קריאות וזה איטי.<br> ה-`embeddings` מטפלים
   ב-95% הקלים בחינם (כפילויות ברורות, לא-כפילויות ברורות), וה-`LLM` רואה
   רק את המקרים הקשים - תעתיקים עבריים, התנגשויות מספרי דגם, הבחנה בין
   וריאנטים.

2. **גרסת v2 - שיפור `LLM` עם שערי ביטחון.**<br> הרעיון: לקרוא ל-`LLM` רק
   כשיש אי-ודאות אמיתית.<br> לפני כל קריאה מתבצעת בדיקה זולה - אם טוקני
   הוריאנט בתוך קלאסטר אחידים, אם סינגלטון יושב מאוד קרוב לסנטרואיד של
   קלאסטר קיים, או אם זוג קלאסטרים נמצא במרחק עצום זה מזה - אפשר לסמוך
   על שלב ה-`embeddings` ולדלג על הקריאה.<br> שתי הגרסאות, v1 (תמיד-`LLM`)
   ו-v2 (עם השערים), נשמרו זו לצד זו ב-[src/llm_refine.py](src/llm_refine.py)
   ו-[src/llm_refine_v2.py](src/llm_refine_v2.py) כדי לאפשר השוואה
   ישירה: על דאטהסט הבדיקה v2 חוסכת ~22% מקריאות ה-`API` _וגם_ מקבלת
   ציון `F1` מעט טוב יותר, ולכן היא ברירת המחדל.

שתי ההחלטות מוסברות מקצה לקצה ב-[APPROACH.md](APPROACH.md), עם גרפים
והשוואה מלאה של v1 מול v2 ב-[APPROACH-extended.md](APPROACH-extended.md).

## מבנה הפרויקט

```
zapTask/
├── main.py                          # End-to-end pipeline entry point
├── requirements.txt
├── .env                             # OPENAI_API_KEY=sk-...  (you create this)
│
├── APPROACH.md                      # Short submission write-up
├── APPROACH-extended.md             # Long-form essay with charts
├── images/                          # Charts referenced by the extended write-ups
│
├── src/
│   ├── config.py                    # All thresholds + paths in one place
│   ├── embeddings.py                # Stage 1: multilingual embedding clustering
│   ├── llm_refine.py                # Stage 2 v1: always-LLM (kept for comparison)
│   ├── llm_refine_v2.py             # Stage 2 v2: confidence-gated (DEFAULT)
│   └── deduplicator.py              # Stage 3: pick canonical name + min price
│
└── data/
    ├── generate_dataset.py          # Synthetic data generator (real Israeli products)
    ├── products.csv                 # The input dataset
    ├── deduplicated_products.csv    # Final unified catalog (output)
    ├── deduplicated_products.audit.json  # Companion decision audit / explainability file
    ├── compare_approaches.py        # v1 vs v2 metrics + chart
    └── generate_charts.py           # Other visualization helpers
```

## איך מריצים

### ריצה רגילה (מומלץ - משתמש ב-v2)

```bash
python main.py
```

### ריצה על `CSV` משלך

```bash
python main.py --input my_products.csv --output my_unified.csv
```

קובץ ה-`CSV` חייב להכיל את העמודות: `product_id`, `product_name`, `price`,
`category`. ראה את [data/products.csv](data/products.csv) לפורמט.

### השוואה מול v1 (תמיד-`LLM`)

```bash
python main.py --v1                           # run the pipeline with v1
python data/compare_approaches.py             # full side-by-side benchmark + chart
```

### יצירה מחדש של הדאטהסט הסינתטי

```bash
python data/generate_dataset.py
```

## פורמט הפלט

לקובץ [data/deduplicated_products.csv](data/deduplicated_products.csv) יש שמונה עמודות:

| עמודה            | משמעות                                        |
| ---------------- | --------------------------------------------- |
| `canonical_name` | השם הארוך ביותר מהקלאסטר - מה שהקונה רואה     |
| `category`       | יורש מהרישומים המקוריים                       |
| `min_price`      | המחיר הזול ביותר בכל הרישומים - המספר הראשי   |
| `max_price`      | המחיר היקר ביותר בכל הרישומים                 |
| `savings_pct`    | `(1 - min/max) * 100`                         |
| `num_listings`   | כמה רישומים מקוריים אוחדו לשורה הזאת          |
| `listing_ids`    | ערכי ה-`product_id` המקוריים, מופרדים בפסיקים |
| `all_names`      | כל שמות הרישומים, מחוברים ב-`\|` (עזר לדיבאג) |

בנוסף נכתב גם
[data/deduplicated_products.audit.json](data/deduplicated_products.audit.json)
- קובץ JSON משלים עם החלטות explainability מתוך שלב 2. כל רשומה מתעדת מה הוחלט,
האם נעשה שימוש ב-`LLM`, איזה ציון דמיון תמך בהחלטה כשיש כזה, ומה הייתה הסיבה
הקצרה.

## תוצאות מדודות על הדאטהסט הסינתטי

הקובץ המחויב כרגע מכיל 30 מוצרים מאוחדים מתוך 131 רישומים. הטבלה למטה
מתארת ריצה מייצגת של v1 מול v2; בגלל אי-דטרמיניזם קל ב-`LLM`, מספר
הקלאסטרים וה-`F1` יכולים לזוז מעט בין ריצות.

| מטריקה                 | v1 (תמיד-`LLM`) |    v2 (שערים) |
| ---------------------- | --------------: | ------------: |
| קריאות `API` לריצה     |              73 |        **57** |
| `F1` ברמת זוגות        |          75.56% |    **77.53%** |
| `Precision` ברמת זוגות |          60.73% |    **64.57%** |
| `Recall` ברמת זוגות    |         100.00% |        97.01% |
| סינגלטונים לא-משובצים  |               0 |             0 |
| יציבות `F1` בין ריצות  |         69%–77% | **~77% קבוע** |

**הערכה: v2 הוא המנצח.** מדויק מעט יותר, 22% זול יותר, והרבה יותר עקבי בין
ריצות כי השערים מוציאים את ה-`LLM` מהמקרים שבהם הוא היה יכול אחרת לטעות
במזל רע.

## הסתייגויות ומה הייתי עושה אחרת עם יותר זמן

- **הספים לא עברו `grid search`.**<br> הם נבחרו לפי אינטואיציה ועוד סבב אחד
  של תיקון באגים. בדאטהסט סינתטי בן 131 שורות עם שונות של ±8 נקודות `F1`
  בין ריצות, סריקה הייתה מתאימה את עצמה יתר על המידה למוזרויות הדאטהסט.
  הזמן הנכון לכיוונון הוא על דאטה אמיתי של זאפ עם החלטות שערים מתועדות.
- **שם קנוני = המחרוזת הארוכה ביותר.** זה `proxy` חינמי ל"הכי אינפורמטיבי
  לאדם". אם כותרות מוכרים אמיתיות אי פעם ייראו כמו
  `!!! SALE !!! ... BEST PRICE !!!`, אפשר להחליף בבוחר מבוסס-`LLM`.
- **אין מטא-דאטה של מוכרים.** פריסה אמיתית בזאפ הייתה נושאת `seller_id`
  לכל רישום כדי שהשורה המאוחדת תוכל להציג "הזול ביותר אצל: מוכר X" במקום
  מזהה רישום עמום.
- **שלב 3 הוא במכוון מינימלי.** כל ערך המוצר נמצא בשלבים 1 ו-2 - ברגע
  שיודעים אילו רישומים הם כפילויות, ה-`group-by` הוא 30 שורות.
