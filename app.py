import json
import os
from dotenv import load_dotenv
load_dotenv()
from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI
from dataset.download import ensure_dataset
from preprocessing.preprocess import run_pipeline

app = Flask(__name__, static_folder="frontend")

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
OLLAMA_URL   = os.environ.get("OLLAMA_URL",   "http://localhost:11434/v1")

_llm = OpenAI(base_url=OLLAMA_URL, api_key="ollama")

WEEK_TO_MONTH = {
    1: "Jan", 2: "Jan", 3: "Jan", 4: "Jan",
    5: "Feb", 6: "Feb", 7: "Feb", 8: "Feb",
    9: "Mar", 10: "Mar", 11: "Mar", 12: "Mar",
    13: "Apr", 14: "Apr", 15: "Apr", 16: "Apr",
    17: "May", 18: "May", 19: "May", 20: "May",
    21: "Jun", 22: "Jun", 23: "Jun", 24: "Jun",
    25: "Jul", 26: "Jul", 27: "Jul", 28: "Jul",
    29: "Aug", 30: "Aug", 31: "Aug", 32: "Aug",
    33: "Sep", 34: "Sep", 35: "Sep", 36: "Sep",
    37: "Oct", 38: "Oct", 39: "Oct", 40: "Oct",
    41: "Nov", 42: "Nov", 43: "Nov", 44: "Nov",
    45: "Dec", 46: "Dec", 47: "Dec", 48: "Dec",
    49: "Dec", 50: "Dec", 51: "Dec", 52: "Dec",
}


def parse_query(user_message: str) -> dict:
    """Use the local LLM to extract species, location, lat, and lon from a natural language query."""
    resp = _llm.chat.completions.create(
        model=OLLAMA_MODEL,
        max_tokens=256,
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract bird observation query parameters from the user's message. "
                    "Respond ONLY with a JSON object containing these fields:\n"
                    '  "species": string (the bird\'s simple common name, e.g. "hummingbird", "sandhill crane", "robin" — no qualifiers like "(general)" or "(species)"),\n'
                    '  "location": string (city, region, or location name, empty string if not found),\n'
                    '  "lat": number or null (approximate latitude of the location),\n'
                    '  "lon": number or null (approximate longitude of the location).\n'
                    "Provide lat/lon only for well-known cities or regions you are confident about. "
                    "Return valid JSON only. No explanation, no markdown, no code fences."
                ),
            },
            {"role": "user", "content": user_message},
        ],
    )
    raw = resp.choices[0].message.content.strip()
    # Strip markdown code fences if the model adds them anyway
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    # Extract the first JSON object if the model added extra text
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1:
        raw = raw[start:end + 1]
    return json.loads(raw)


def generate_response(
    user_message: str,
    pipeline_result: dict,
    parsed_query: dict,
) -> str:
    """Use Claude to turn pipeline results into a natural language answer."""
    peak_summary = ", ".join(
        f"week {p['week']} ({WEEK_TO_MONTH.get(p['week'], '?')}, "
        f"{p['normalized_frequency']*100:.1f}% of checklists)"
        for p in pipeline_result.get("peak_weeks", [])
    )
    species_list = ", ".join(pipeline_result.get("species_found", [])[:3])
    total = pipeline_result.get("total_records", 0)
    location_note = pipeline_result.get("location_note", "")

    context = (
        f"User asked: {user_message}\n\n"
        f"Analysis results:\n"
        f"- Species matched: {species_list}\n"
        f"- Total observations in dataset: {total}\n"
        f"- Peak observation weeks: {peak_summary or 'none identified'}\n"
    )
    if location_note:
        context += f"- Note: {location_note}\n"

    resp = _llm.chat.completions.create(
        model=OLLAMA_MODEL,
        max_tokens=512,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful bird observation assistant. "
                    "Given the analysis results from an eBird dataset, answer the user's question "
                    "in 2-4 sentences. Be specific about peak weeks and months. "
                    "Mention the probability as a percentage. Be friendly and informative."
                ),
            },
            {"role": "user", "content": context},
        ],
    )
    return resp.choices[0].message.content.strip()


@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory("frontend", filename)


@app.route("/assets/<path:filename>")
def asset_files(filename):
    return send_from_directory("assets", filename)


@app.route("/api/query", methods=["POST"])
def query():
    data = request.get_json(force=True)
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    # Step 1: Parse query with LLM
    try:
        parsed = parse_query(user_message)
    except Exception as e:
        return jsonify({"error": f"Query parsing failed: {e}"}), 500

    species = parsed.get("species", "").strip()
    location = parsed.get("location", "").strip()
    lat = parsed.get("lat")
    lon = parsed.get("lon")

    if not species:
        return jsonify({
            "response": "I couldn't identify a bird species in your question. "
                        "Try asking something like: \"When is the best time to see "
                        "a Robin in Knoxville?\"",
            "chart_data": None,
            "peak_weeks": [],
        })

    # Step 2: Run preprocessing pipeline
    result = run_pipeline(
        species_query=species,
        location_query=location or None,
        lat=float(lat) if lat is not None else None,
        lon=float(lon) if lon is not None else None,
    )

    if "error" in result:
        return jsonify({
            "response": result["error"],
            "chart_data": None,
            "peak_weeks": [],
        })

    # Step 3: Generate natural language response
    try:
        explanation = generate_response(user_message, result, parsed)
    except Exception as e:
        explanation = f"Analysis complete. Peak weeks: {result.get('peak_weeks')}"

    # Step 4: Build chart data
    weekly = result["weekly_frequencies"]
    peak_week_nums = {p["week"] for p in result["peak_weeks"]}
    chart_data = {
        "labels": [f"W{r['week']}" for r in weekly],
        "frequencies": [round(r["normalized_frequency"] * 100, 2) for r in weekly],
        "peak_weeks": list(peak_week_nums),
        "species": result["species_found"][0] if result["species_found"] else species,
        "location": location or "All locations",
        "yearly_data": result.get("yearly_data", {}),
    }

    return jsonify({
        "response": explanation,
        "chart_data": chart_data,
        "peak_weeks": result["peak_weeks"],
        "location_note": result.get("location_note"),
        "total_records": result["total_records"],
        "parsed_query": {"species": species, "location": location},
    })


if __name__ == "__main__":
    ensure_dataset()
    app.run(debug=True, port=5000)
