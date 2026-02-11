import pandas as pd
import glob
import json
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).parent
LAST_OUTPUTS = SCRIPT_DIR / "last_outputs.json"
DOWNLOAD_DIR = SCRIPT_DIR / "datalogs"

# ================= INSTELLINGEN =================

kolom_volgorde = [
    "Tijd",
    "Temperatuur Fitness", "Recycle Temperatuur", "Dojo Temperatuur",
    "Temperatuur Zaal 1", "kantine Temperatuur", "Instuur Temperatuur",
    "Buiten Temperatuur", "dojo luchtvochtigheid", "dojo Co2",
    "Fitness luchtvochtigheid", "Fitness Co2", "buitenlucht klep",
    "Fitness bezetting"   # alleen voor daggrafiek
]

sensor_map = {
    "B040.Ax": "Temperatuur Fitness",
    "B060.Ax": "Recycle Temperatuur",
    "B061.Ax": "kantine Temperatuur",
    "B062.Ax": "Temperatuur Zaal 1",
    "B063.Ax": "Instuur Temperatuur",
    "B064.Ax": "Buiten Temperatuur",
    "AM2": "Dojo Temperatuur",
    "AM4": "dojo luchtvochtigheid",
    "AM6": "dojo Co2",
    "AM8": "Fitness luchtvochtigheid",
    "AM9": "Fitness Co2",
    "AQ1": "buitenlucht klep",
    "I21": "Fitness bezetting"
}

# ================= DATA INLEZEN =================

def main():
    files = sorted(
        glob.glob(str(DOWNLOAD_DIR / "*.csv")) +
        glob.glob(str(DOWNLOAD_DIR / "*.txt"))
    )


    if not files:
        print("Geen data gevonden")
        return

    frames = []

    for f in files:
        df = pd.read_csv(f, skipinitialspace=True)
        df = df.rename(columns=sensor_map)

        if "Time" not in df.columns:
            continue

        df["Tijd"] = pd.to_datetime(df["Time"])
        df.drop(columns=["Time"], inplace=True)

        df.replace(-500, pd.NA, inplace=True)

        for code, col in sensor_map.items():
            if col not in df.columns:
                continue

            df[col] = pd.to_numeric(df[col], errors="coerce")

            if code != "I21":
                df[col] = df[col] / 10

            if code == "AQ1":
                df[col] -= 6

        # I21 → 0/1 (alleen als aanwezig)
        if "Fitness bezetting" in df.columns:
            df["Fitness bezetting"] = (df["Fitness bezetting"] > 0).astype(int)

        df = df[[c for c in kolom_volgorde if c in df.columns]]
        df["Datum"] = df["Tijd"].dt.date
        df["Jaar"] = df["Tijd"].dt.year

        frames.append(df)

    if not frames:
        print("Geen geldige dataframes om te verwerken")
        return

    data = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["Tijd"])
        .sort_values("Tijd")
    )

    ALLE_JAREN = sorted(data["Jaar"].unique().tolist())

    # ================= HTML FUNCTIES =================

    # ================= dag generatie =================

    def write_day_html(path, datum, traces_main, bezetting_trace):
        html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <title>{datum}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>body {{ font-family: Arial; }}</style>
    </head>
    <body>

    <h3>{datum}</h3>

    <select id="type">
      <option value="sensoren">Sensoren</option>
      <option value="co2">CO2 & Luchtvochtigheid</option>
      <option value="klep">Buitenlucht klep</option>
    </select>

    <div id="plot" style="height:450px;"></div>
    <div id="bezetting" style="height:180px;"></div>

    <script>
    const mainData = {json.dumps(traces_main)};
    const bezetting = {json.dumps(bezetting_trace)};

    function draw(t) {{
      Plotly.newPlot("plot", mainData[t], {{
        hovermode: 'x unified',
        template: 'plotly_white'
      }});

      if (bezetting.length > 0) {{
        Plotly.newPlot("bezetting", bezetting, {{
          template: 'plotly_white',
          yaxis: {{
            range: [-0.1, 1.1],
            tickvals: [0, 1],
            ticktext: ["Niemand", "Iemand"],
            title: "Aanwezigheid"
          }},
          margin: {{ t: 20 }}
        }});
      }} else {{
        Plotly.purge("bezetting");
      }}
    }}

    draw("sensoren");
    document.getElementById("type").onchange = e => draw(e.target.value);
    </script>

    </body>
    </html>
    """
        path.write_text(html, encoding="utf-8")

    # ============== overzigt generatie ===============

    def write_overview_html(path, jaar, sensor, co2, klep, datums, alle_jaren):
        html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <title>Overzicht {jaar}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
    body {{ font-family: Arial; }}
    #error {{ color: red; font-weight: bold; }}
    .controls {{ margin-bottom: 10px; }}
    </style>
    </head>
    <body>

    <h2>Gemiddelden per dag</h2>

    <div class="controls">
      Jaar:
      <select id="jaar">
        {''.join(f'<option value="{y}" {"selected" if y == jaar else ""}>{y}</option>' for y in alle_jaren)}
      </select>

      Type:
      <select id="type">
        <option value="sensoren">Sensoren</option>
        <option value="co2">CO2 & Luchtvochtigheid</option>
        <option value="klep">Buitenlucht klep</option>
      </select>

      Datum:
      <input type="date" id="date">
      <span id="error"></span>
    </div>

    <div id="plot" style="height:600px;"></div>

    <script>
    const dataSets = {{
      sensoren: {json.dumps(sensor)},
      co2: {json.dumps(co2)},
      klep: {json.dumps(klep)}
    }};

    const datums = {json.dumps(datums)};
    const error = document.getElementById("error");

    function draw(type) {{
      Plotly.newPlot("plot", dataSets[type], {{
        hovermode: 'x unified',
        template: 'plotly_white'
      }});
    }}

    draw("sensoren");

    document.getElementById("type").onchange = e => draw(e.target.value);

    document.getElementById("jaar").onchange = e => {{
      const y = e.target.value;
      window.location.href = `../html_${{y}}/Temperature_${{y}}_overview.html`;
    }};

    document.getElementById("date").onchange = e => {{
      const d = e.target.value;
      error.innerText = "";

      if (!datums.includes(d)) {{
        error.innerText = "Geen data voor deze datum";
        return;
      }}
      window.open(`per_dag/grafiek_${{d}}.html`);
    }};

    document.getElementById("plot").on('plotly_click', d => {{
      const day = d.points[0].x;
      window.open(`per_dag/grafiek_${{day}}.html`);
    }});
    </script>

    </body>
    </html>
    """
        path.write_text(html, encoding="utf-8")

    # ================= GENERATIE =================

    for jaar, dfy in data.groupby("Jaar"):
        jaar_dir = SCRIPT_DIR / f"html_{jaar}"
        dag_dir = jaar_dir / "per_dag"
        jaar_dir.mkdir(exist_ok=True)
        dag_dir.mkdir(exist_ok=True)

        # ❗ I21 expliciet uitsluiten van jaaroverzicht
        overzicht = (
            dfy.drop(columns=["Fitness bezetting"], errors="ignore")
            .groupby("Datum")
            .mean(numeric_only=True)
            .reset_index()
        )

        def make_traces(cols):
            traces = []
            for c in cols:
                if c not in overzicht.columns:
                    continue
                traces.append({
                    "x": overzicht["Datum"].astype(str).tolist(),
                    "y": overzicht[c].tolist(),
                    "name": c,
                    "mode": "lines+markers"
                })
            return traces

        sensor_cols = [
            c for c in overzicht.columns
            if c not in ("Datum",)
               and "Co2" not in c
               and "lucht" not in c.lower()
               and "klep" not in c.lower()
               and "jaar" not in c.lower()
        ]

        co2_cols = [c for c in overzicht.columns if "Co2" in c or "luchtvochtigheid" in c.lower()]
        klep_cols = [c for c in overzicht.columns if "klep" in c.lower()]

        write_overview_html(
            jaar_dir / f"Temperature_{jaar}_overview.html",
            jaar,
            make_traces(sensor_cols),
            make_traces(co2_cols),
            make_traces(klep_cols),
            overzicht["Datum"].astype(str).tolist(),
            ALLE_JAREN
        )

        for datum, dfd in dfy.groupby("Datum"):

            def make_day(cols):
                traces = []
                for c in cols:
                    if c not in dfd.columns:
                        continue
                    traces.append({
                        "x": dfd["Tijd"].astype(str).tolist(),
                        "y": dfd[c].tolist(),
                        "name": c,
                        "mode": "lines"
                    })
                return traces

            bezetting = []
            if "Fitness bezetting" in dfd.columns:
                y = dfd["Fitness bezetting"]

                # 🔁 omdraaien: 1 = aanwezig (boven), 0 = niemand (beneden)
                y = y.replace({1: 0, 0: 1})

                bezetting = [{
                    "x": dfd["Tijd"].astype(str).tolist(),
                    "y": y.tolist(),
                    "mode": "lines",
                    "line": {
                        "width": 4,
                        "shape": "hv"  # 👈 geen schuine lijnen
                    },
                    "name": "Aanwezig"
                }]

            write_day_html(
                dag_dir / f"grafiek_{datum}.html",
                datum,
                {
                    "sensoren": make_day(sensor_cols),
                    "co2": make_day(co2_cols),
                    "klep": make_day(klep_cols),
                },
                bezetting
            )


if __name__ == "__main__":
    main()
