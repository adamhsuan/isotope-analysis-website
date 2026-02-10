import os
import json
import uuid
from flask import Flask, render_template, request
from analysis import analyze_spectrum  # import the function

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
ANALYSES_FOLDER = "analyses"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANALYSES_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["ANALYSES_FOLDER"] = ANALYSES_FOLDER

def _match_upload_for_analysis(analysis_path, uploads_dir, max_seconds=120):
    """Try to find an uploaded spectrum file whose modification time is near the analysis file's mtime.
    Returns the filename (basename) if a good match is found, else None."""
    try:
        analysis_mtime = os.path.getmtime(analysis_path)
    except Exception:
        return None
    best = None
    best_delta = None
    for fname in os.listdir(uploads_dir):
        fpath = os.path.join(uploads_dir, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            ft = os.path.getmtime(fpath)
        except Exception:
            continue
        delta = abs(ft - analysis_mtime)
        if delta <= max_seconds and (best_delta is None or delta < best_delta):
            best = fname
            best_delta = delta
    return best

@app.route("/")
def index():
    # list saved analyses (show id and timestamp from filename)
    analyses = []
    for fname in sorted(os.listdir(app.config["ANALYSES_FOLDER"]), reverse=True):
        if not fname.endswith('.json'):
            continue
        aid = os.path.splitext(fname)[0]
        path = os.path.join(app.config["ANALYSES_FOLDER"], fname)
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            label = data.get('label') or aid
            # If label looks like the default (e.g. 'Analysis <id>') try to recover an upload filename
            if label.startswith('Analysis') and 'spectrum_filename' not in data:
                candidate = _match_upload_for_analysis(path, app.config['UPLOAD_FOLDER'])
                if candidate:
                    # update the saved analysis file so it now includes the detected spectrum filename and a label
                    data['spectrum_filename'] = candidate
                    data['label'] = candidate
                    try:
                        with open(path, 'w') as out_f:
                            json.dump(data, out_f, default=str)
                        label = candidate
                    except Exception:
                        # if we cannot write, just fall back to the candidate for display
                        label = candidate
        except Exception:
            label = aid
        analyses.append({"id": aid, "label": label})
    return render_template("index.html", previous_analyses=analyses)

@app.route("/upload", methods=["POST"])
def upload_files():
    try:
        # --- Get files from the form ---
        spectrum_file = request.files.get("spectrum")
        background_file = request.files.get("background")
        efficiency_file = request.files.get("efficiency")

        if not spectrum_file or not background_file or not efficiency_file:
            return "Error: Please upload all three files."

        # --- Save files temporarily ---
        filepaths = []
        for f in [spectrum_file, background_file, efficiency_file]:
            path = os.path.join(app.config["UPLOAD_FOLDER"], f.filename)
            f.save(path)
            filepaths.append(path)

        spectrum_path, background_path, efficiency_path = filepaths

        # --- Run analysis ---
        the_results = analyze_spectrum(spectrum_path, background_path, efficiency_path)

        # persist this analysis so it can be revisited from the index page
        analysis_id = uuid.uuid4().hex
        outpath = os.path.join(app.config["ANALYSES_FOLDER"], f"{analysis_id}.json")
        # attach a human-readable label if possible (use original spectrum filename when available)
        the_results_to_save = dict(the_results)
        spectrum_filename = getattr(spectrum_file, 'filename', None) or os.path.basename(spectrum_path)
        the_results_to_save['spectrum_filename'] = spectrum_filename
        the_results_to_save['label'] = spectrum_filename or f"Analysis {analysis_id[:8]}"

        # normalize daughters_graphs to a dict for compatibility with older saved formats
        def _normalize_daughters_graphs(dg):
            if isinstance(dg, list):
                names = ["U-238", "Pu-239", "U-235", "Th-232"]
                return {names[i]: dg[i] for i in range(min(len(dg), len(names)))}
            return dg

        the_results_to_save['daughters_graphs'] = _normalize_daughters_graphs(the_results_to_save.get('daughters_graphs', {}))

        try:
            with open(outpath, 'w') as f:
                json.dump(the_results_to_save, f, default=str)
        except Exception:
            # non-fatal: continue without saving
            pass

        # ensure the variable we pass to the template is normalized
        daughters_graphs_var = _normalize_daughters_graphs(the_results.get('daughters_graphs', {}))

        # --- Return results page ---
        return render_template(
            "results.html",
            results = the_results["results"],
            results_graph = the_results["results_graph"],
            daughters_graphs = daughters_graphs_var,
            energy_graphs = the_results["energy_graphs"],
            isotopes_info = the_results["isotopes_info"],
            only_energies_graphs = the_results["only_energies_graphs"],
            daughters_energies = the_results["daughters_energies"],
            key_graphs = the_results["key_graphs"],
            conclusions = the_results["conclusions"],
            isotopes_dictionary = the_results.get("isotopes_dictionary"),
        )

    except Exception as e:
        return f"An error occurred: {e}"


@app.route('/analysis/<analysis_id>')
def view_analysis(analysis_id):
    path = os.path.join(app.config['ANALYSES_FOLDER'], f"{analysis_id}.json")
    if not os.path.exists(path):
        return f"Analysis {analysis_id} not found", 404
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        # normalize potential legacy list-format daughters_graphs into a dict
        dg = data.get("daughters_graphs", {})
        if isinstance(dg, list):
            names = ["U-238", "Pu-239", "U-235", "Th-232"]
            dg = {names[i]: dg[i] for i in range(min(len(dg), len(names)))}

        # render the same template as after upload
        label = data.get('label') or data.get('spectrum_filename')
        # persist label into the file if not present (help future UI and indexing)
        if 'label' not in data and label:
            data['label'] = label
            try:
                with open(path, 'w') as out_f:
                    json.dump(data, out_f, default=str)
            except Exception:
                pass

        return render_template(
            "results.html",
            results = data.get("results", []),
            results_graph = data.get("results_graph"),
            daughters_graphs = dg,
            energy_graphs = data.get("energy_graphs", {}),
            isotopes_info = data.get("isotopes_info", {}),
            only_energies_graphs = data.get("only_energies_graphs", {}),
            daughters_energies = data.get("daughters_energies", {}),
            key_graphs = data.get("key_graphs"),
            conclusions = data.get("conclusions"),
            isotopes_dictionary = data.get("isotopes_dictionary", {}),
            label = label,
        )
    except Exception as e:
        return f"Could not load analysis: {e}", 500

if __name__ == "__main__":
    app.run(debug=True)
