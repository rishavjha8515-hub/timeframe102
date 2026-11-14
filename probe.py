from astroquery.ipac.irsa import Irsa

print("=== SPHEREx collections available ===")
try:
    cols = Irsa.list_collections()
    for row in cols:
        text = " | ".join(str(v) for v in row)
        if "spherex" in text.lower():
            print(text[:300])
except Exception as exc:
    print(f"LIST_FAILED {type(exc).__name__}: {exc}")

try:
    from astroquery.ipac.irsa.most import Most
except Exception as exc:
    Most = None
    print(f"MOST_IMPORT_FAILED {type(exc).__name__}: {exc}")

if Most is not None:
    for name in ("Ceres", "Vesta", "Pallas", "Juno", "Hygiea"):
        print(f"\n=== MOST {name} ===")
        try:
            result = Most.query_object(
                obj_name=name,
                catalog="spherex",
                input_type="name_input",
                obs_begin="2025-05-01",
                obs_end="2026-11-14",
                output_mode="Brief",
            )
            print("ROWS", len(result))
            print("COLUMNS", result.colnames)
            print(result[:8])
        except Exception as exc:
            print(f"QUERY_FAILED {type(exc).__name__}: {exc}")

for label, ra, dec in (
    ("north_ecliptic_pole", 270.0, 66.5607),
    ("south_ecliptic_pole", 90.0, -66.5607),
):
    print(f"\n=== SIA {label} ===")
    try:
        result = Irsa.query_sia(pos=f"CIRCLE {ra} {dec} 0.5", collection="spherex_qr2")
        print("ROWS", len(result))
        print("COLUMNS", result.colnames)
        print(result[:5])
    except Exception as exc:
        print(f"QUERY_FAILED {type(exc).__name__}: {exc}")
