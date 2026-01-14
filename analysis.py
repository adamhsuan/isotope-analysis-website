import json
import becquerel as bq
from uncertainties import unumpy as unp
from scipy.interpolate import interp1d

def analyze_spectrum2(spectrum_path, background_path, efficiency_path):
    
    """
    Reads spectrum, background, and efficiency files,
    performs background subtraction and efficiency correction,
    and returns a single formatted string with summary results.
    """

    # --- Load spectra ---
    spectrum = bq.Spectrum.from_file(spectrum_path)
    background = bq.Spectrum.from_file(background_path)

    # --- Background subtraction ---
    counts_nominal = unp.nominal_values(spectrum.counts - background.counts)

    # --- Energy axis ---
    try:
        energies = spectrum.energies
        x_label = "Energy (keV)"
    except AttributeError:
        energies = spectrum.channels
        x_label = "Channel"

    # --- Load efficiency JSON ---
    with open(efficiency_path, "r") as f:
        eff_data = json.load(f)

    eff_energies = eff_data["energies"]
    eff_values = eff_data["efficiencies"]

    # --- Interpolate efficiency ---
    eff_interp = interp1d(
        eff_energies,
        eff_values,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate"
    )

    efficiencies = eff_interp(energies)
    corrected_counts = counts_nominal / efficiencies

    # --- Compute key metrics ---
    total_counts = sum(corrected_counts)
    peak_count = max(corrected_counts)
    peak_index = corrected_counts.tolist().index(peak_count)
    peak_energy = energies[peak_index]
    n_points = len(energies)

    # --- Return a single formatted string ---
    summary = (
        f"Gamma Spectrum Analysis Summary\n"
        f"--------------------------------\n"
        f"X-axis type: {x_label}\n"
        f"Total corrected counts: {total_counts:.2f}\n"
        f"Highest peak: {peak_count:.2f} at {peak_energy}\n"
        f"Number of points: {n_points}\n"
        f"Efficiency applied: Yes (interpolated from JSON)\n"
    )

    return summary
