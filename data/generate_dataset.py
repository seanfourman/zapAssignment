"""
Generate a realistic synthetic product dataset modeled after zap.co.il patterns.

Products include intentional duplicates with:
- Hebrew/English name variations
- Abbreviations and partial names
- Different price points for the same product
- Near-matches that are NOT duplicates (e.g., different storage, different model)
"""

import csv
import random
import os

random.seed(42)

# Each "canonical product" has a list of name variations and a price range.
# Variations simulate real-world inconsistencies seen on zap.co.il.
# price_range = (min, max) - each listing gets a random price in this range.

PRODUCTS = [
    # === PHONES ===
    {
        "canonical": "Samsung Galaxy S24 Ultra 256GB",
        "category": "סלולר",
        "variations": [
            "Samsung Galaxy S24 Ultra 256GB",
            "סמסונג גלקסי S24 אולטרה 256GB",
            "Galaxy S24 Ultra 256GB",
            "Samsung S24 Ultra 256 GB",
            "SAMSUNG Galaxy S24 Ultra 256GB SM-S928B",
        ],
        "price_range": (3200, 4100),
    },
    {
        "canonical": "Samsung Galaxy S24 Ultra 512GB",  # NOT a duplicate of 256GB
        "category": "סלולר",
        "variations": [
            "Samsung Galaxy S24 Ultra 512GB",
            "סמסונג גלקסי S24 אולטרה 512GB",
            "Galaxy S24 Ultra 512GB",
        ],
        "price_range": (3800, 4800),
    },
    {
        "canonical": "Samsung Galaxy S25 256GB",
        "category": "סלולר",
        "variations": [
            "Samsung Galaxy S25 256GB",
            "סמסונג גלקסי S25 256GB",
            "Galaxy S25 256 GB",
            "Samsung S25 256GB",
        ],
        "price_range": (2300, 2900),
    },
    {
        "canonical": "Samsung Galaxy S25 FE 256GB",  # NOT a duplicate of S25
        "category": "סלולר",
        "variations": [
            "Samsung Galaxy S25 FE 256GB",
            "סמסונג גלקסי S25 FE 256GB",
            "Galaxy S25 FE 256GB",
        ],
        "price_range": (1800, 2300),
    },
    {
        "canonical": "Apple iPhone 16 Pro Max 256GB",
        "category": "סלולר",
        "variations": [
            "Apple iPhone 16 Pro Max 256GB",
            "אייפון 16 פרו מקס 256GB",
            "iPhone 16 Pro Max 256GB",
            "אפל אייפון 16 פרו מקס 256 ג'יגה",
            "APPLE iPhone 16 Pro Max 256GB",
        ],
        "price_range": (4200, 5200),
    },
    {
        "canonical": "Apple iPhone 16 Pro 256GB",  # NOT a duplicate of Pro Max
        "category": "סלולר",
        "variations": [
            "Apple iPhone 16 Pro 256GB",
            "אייפון 16 פרו 256GB",
            "iPhone 16 Pro 256GB",
        ],
        "price_range": (3800, 4500),
    },
    {
        "canonical": "Apple iPhone 16 128GB",
        "category": "סלולר",
        "variations": [
            "Apple iPhone 16 128GB",
            "אייפון 16 128GB",
            "iPhone 16 128 GB",
            "אפל אייפון 16 128 ג'יגה",
        ],
        "price_range": (2400, 3100),
    },
    {
        "canonical": "Apple iPhone 15 128GB",
        "category": "סלולר",
        "variations": [
            "Apple iPhone 15 128GB",
            "אייפון 15 128GB",
            "iPhone 15 128GB",
            "אפל אייפון 15 128 ג'יגה",
        ],
        "price_range": (1900, 2500),
    },
    {
        "canonical": "Xiaomi 14T Pro 512GB",
        "category": "סלולר",
        "variations": [
            "Xiaomi 14T Pro 512GB",
            "שיאומי 14T פרו 512GB",
            "Xiaomi 14T Pro 512 GB 12GB RAM",
            "שיאומי 14T Pro 512GB",
        ],
        "price_range": (1800, 2400),
    },
    {
        "canonical": "Google Pixel 9 256GB",
        "category": "סלולר",
        "variations": [
            "Google Pixel 9 256GB",
            "גוגל פיקסל 9 256GB",
            "Pixel 9 256 GB 8GB RAM",
            "גוגל Pixel 9 256GB",
        ],
        "price_range": (2200, 2900),
    },
    {
        "canonical": "Samsung Galaxy A15 128GB",
        "category": "סלולר",
        "variations": [
            "Samsung Galaxy A15 128GB",
            "סמסונג גלקסי A15 128GB",
            "Galaxy A15 128 GB",
            "Samsung A15 128GB SM-A155F",
        ],
        "price_range": (450, 700),
    },

    # === TVs ===
    {
        "canonical": "Samsung 55\" Crystal UHD 4K TV",
        "category": "טלוויזיות",
        "variations": [
            "Samsung 55\" Crystal UHD 4K",
            "סמסונג 55 אינץ' Crystal UHD 4K",
            "Samsung UE55DU7100 Crystal UHD",
            "טלוויזיה סמסונג 55 אינץ 4K",
            "Samsung 55 inch Crystal UHD 4K Smart TV",
        ],
        "price_range": (1800, 2600),
    },
    {
        "canonical": "LG 65\" OLED C4",
        "category": "טלוויזיות",
        "variations": [
            "LG OLED65C4 65\"",
            "LG 65 אינץ' OLED C4",
            "טלוויזיה LG OLED C4 65 אינץ",
            "LG OLED C4 65\" 4K Smart TV",
        ],
        "price_range": (4500, 6200),
    },
    {
        "canonical": "LG 55\" OLED C4",  # NOT a duplicate of 65"
        "category": "טלוויזיות",
        "variations": [
            "LG OLED55C4 55\"",
            "LG 55 אינץ' OLED C4",
            "טלוויזיה LG OLED C4 55 אינץ",
        ],
        "price_range": (3500, 4800),
    },
    {
        "canonical": "Sony 65\" Bravia X90L 4K",
        "category": "טלוויזיות",
        "variations": [
            "Sony 65\" Bravia X90L",
            "סוני בראוויה 65 אינץ' X90L",
            "Sony XR-65X90L Bravia 4K",
            "טלוויזיה סוני 65 אינץ Bravia X90L",
        ],
        "price_range": (3800, 5000),
    },
    {
        "canonical": "Hisense 50\" 4K Smart TV",
        "category": "טלוויזיות",
        "variations": [
            "Hisense 50A6K 50\" 4K",
            "הייסנס 50 אינץ' 4K",
            "Hisense 50 inch 4K Smart TV",
            "טלוויזיה הייסנס 50 אינץ 4K",
        ],
        "price_range": (1200, 1800),
    },

    # === HEADPHONES ===
    {
        "canonical": "Sony WH-1000XM5",
        "category": "אוזניות",
        "variations": [
            "Sony WH-1000XM5",
            "סוני WH-1000XM5",
            "Sony WH1000XM5 Wireless",
            "אוזניות סוני WH-1000XM5 אלחוטיות",
            "Sony 1000XM5",
        ],
        "price_range": (900, 1400),
    },
    {
        "canonical": "Sony WH-1000XM4",  # NOT a duplicate of XM5
        "category": "אוזניות",
        "variations": [
            "Sony WH-1000XM4",
            "סוני WH-1000XM4",
            "Sony WH1000XM4",
        ],
        "price_range": (600, 950),
    },
    {
        "canonical": "Apple AirPods Pro 2",
        "category": "אוזניות",
        "variations": [
            "Apple AirPods Pro 2",
            "אפל אירפודס פרו 2",
            "AirPods Pro 2nd Generation",
            "אירפודז פרו 2",
            "Apple AirPods Pro (2nd Gen) USB-C",
        ],
        "price_range": (700, 1000),
    },
    {
        "canonical": "Apple AirPods 4",  # NOT a duplicate of AirPods Pro
        "category": "אוזניות",
        "variations": [
            "Apple AirPods 4",
            "אפל אירפודס 4",
            "AirPods 4th Generation",
        ],
        "price_range": (400, 600),
    },
    {
        "canonical": "Samsung Galaxy Buds 3 Pro",
        "category": "אוזניות",
        "variations": [
            "Samsung Galaxy Buds3 Pro",
            "סמסונג גלקסי באדס 3 פרו",
            "Galaxy Buds 3 Pro",
            "Samsung Buds 3 Pro",
        ],
        "price_range": (550, 850),
    },
    {
        "canonical": "JBL Tune 520BT",
        "category": "אוזניות",
        "variations": [
            "JBL Tune 520BT",
            "JBL T520BT",
            "אוזניות JBL Tune 520BT",
            "ג'יי בי אל Tune 520BT",
        ],
        "price_range": (120, 220),
    },

    # === LAPTOPS ===
    {
        "canonical": "Apple MacBook Air M3 13\" 256GB",
        "category": "מחשבים ניידים",
        "variations": [
            "Apple MacBook Air 13\" M3 256GB",
            "אפל מקבוק אייר 13 M3 256GB",
            "MacBook Air M3 13 inch 256GB",
            "מקבוק אייר 13 אינץ' M3 256 ג'יגה",
            "Apple MacBook Air 13.6\" M3 chip 8GB 256GB",
        ],
        "price_range": (3800, 4800),
    },
    {
        "canonical": "Apple MacBook Air M3 13\" 512GB",  # NOT a duplicate of 256GB
        "category": "מחשבים ניידים",
        "variations": [
            "Apple MacBook Air 13\" M3 512GB",
            "אפל מקבוק אייר 13 M3 512GB",
            "MacBook Air M3 13 inch 512GB",
        ],
        "price_range": (4800, 5800),
    },
    {
        "canonical": "Lenovo IdeaPad Slim 3 15.6\" i5",
        "category": "מחשבים ניידים",
        "variations": [
            "Lenovo IdeaPad Slim 3 15.6\" i5 512GB",
            "לנובו IdeaPad Slim 3 15.6 אינץ'",
            "Lenovo IdeaPad 3 15.6\" Intel Core i5",
            "לנובו אידיאפד סלים 3 15.6 אינץ i5",
        ],
        "price_range": (2200, 3000),
    },
    {
        "canonical": "Dell XPS 13 Plus i7",
        "category": "מחשבים ניידים",
        "variations": [
            "Dell XPS 13 Plus i7 512GB",
            "דל XPS 13 פלוס i7",
            "Dell XPS 13 Plus Intel Core i7 13th Gen",
            "DELL XPS13 Plus i7 512GB 16GB",
        ],
        "price_range": (5500, 7200),
    },
    {
        "canonical": "ASUS VivoBook 15 i5",
        "category": "מחשבים ניידים",
        "variations": [
            "ASUS VivoBook 15 X1504 i5 512GB",
            "אסוס ויוובוק 15 i5",
            "Asus Vivobook 15 Intel i5 15.6\"",
            "ASUS VivoBook X1504ZA i5 512GB",
        ],
        "price_range": (2000, 2800),
    },

    # === WASHING MACHINES ===
    {
        "canonical": "LG 9kg Front Load Washing Machine",
        "category": "מכונות כביסה",
        "variations": [
            "LG F1209SMS 9KG Front Load",
            "מכונת כביסה LG 9 ק\"ג פתח קדמי",
            "LG 9kg Washing Machine",
            "LG מכונת כביסה 9 קילו",
        ],
        "price_range": (1800, 2600),
    },
    {
        "canonical": "Samsung 8kg Front Load Washing Machine",
        "category": "מכונות כביסה",
        "variations": [
            "Samsung WW80T534 8kg",
            "סמסונג מכונת כביסה 8 ק\"ג",
            "Samsung 8kg Washing Machine EcoBubble",
            "מכונת כביסה סמסונג 8 קילו",
        ],
        "price_range": (1600, 2400),
    },
    {
        "canonical": "Bosch 9kg Front Load Washing Machine",
        "category": "מכונות כביסה",
        "variations": [
            "Bosch WAX32M40 9kg",
            "בוש מכונת כביסה 9 ק\"ג",
            "Bosch Series 8 9kg Washing Machine",
            "מכונת כביסה בוש 9 קילו סדרה 8",
        ],
        "price_range": (2800, 3800),
    },

    # === TABLETS ===
    {
        "canonical": "Apple iPad Air M2 11\" 128GB",
        "category": "טאבלטים",
        "variations": [
            "Apple iPad Air 11\" M2 128GB",
            "אפל אייפד אייר 11 M2 128GB",
            "iPad Air M2 11 inch 128GB Wi-Fi",
            "אייפד אייר 11 אינץ' M2 128 ג'יגה",
        ],
        "price_range": (2200, 2900),
    },
    {
        "canonical": "Samsung Galaxy Tab S9 FE 128GB",
        "category": "טאבלטים",
        "variations": [
            "Samsung Galaxy Tab S9 FE 128GB",
            "סמסונג גלקסי טאב S9 FE 128GB",
            "Galaxy Tab S9 FE 128 GB Wi-Fi",
            "Samsung Tab S9 FE 128GB",
        ],
        "price_range": (1200, 1700),
    },
    {
        "canonical": "Apple iPad 10th Gen 64GB",
        "category": "טאבלטים",
        "variations": [
            "Apple iPad 10.9\" 10th Gen 64GB",
            "אפל אייפד דור 10 64GB",
            "iPad 10th Generation 64 GB",
            "אייפד 10.9 אינץ' דור עשירי 64GB",
        ],
        "price_range": (1100, 1600),
    },
]


def generate_dataset(output_path: str):
    """Generate the product dataset as CSV."""
    rows = []
    product_id = 1

    for product in PRODUCTS:
        # Generate one listing per variation with a random price
        for variation in product["variations"]:
            price = random.randint(*product["price_range"])
            rows.append({
                "product_id": product_id,
                "product_name": variation,
                "price": price,
                "category": product["category"],
            })
            product_id += 1

    # Shuffle to make it realistic - duplicates won't be next to each other
    random.shuffle(rows)

    # Reassign IDs after shuffle
    for i, row in enumerate(rows, 1):
        row["product_id"] = i

    # Write CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["product_id", "product_name", "price", "category"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} product listings from {len(PRODUCTS)} canonical products")
    print(f"Saved to: {output_path}")

    # Print some stats
    categories = {}
    for p in PRODUCTS:
        cat = p["category"]
        categories[cat] = categories.get(cat, 0) + 1
    print(f"\nCanonical products by category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "products.csv")
    generate_dataset(output_path)
