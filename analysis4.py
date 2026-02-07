import os
import json
import becquerel as bq
from uncertainties import unumpy as unp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import becquerel as bq
from becquerel import Spectrum
import numpy as np
import matplotlib.pyplot as plt
import sys
import math
import json
from PF import PF
import analysis_methods as am
import math
from uncertainties import nominal_value
import pandas as pd
import os
import uuid

def analyze_spectrum(spectrum_path, background_path, efficiency_path, plot_output="static/plot.png"):

    NA = 6.022e23

    RADIUS = 2
    MIN_PEAK_COUNTS = 1000
    EFFICIENCY="2019_NAA_eff_calibration_parameters.json"
    CHECK_LIMIT=4
    MAX_UNCERTAINTY_PROPORTION = 0.1 #max uncertainty/counts
    MAX_DAUGHTERS_TO_FLAG_PARENT = 3 #max number of daughter isotopes to flag a parent isotope
    PEAK_STANDARD_DEVIATION_UNCERTAINTY_CUTOFF = 1000
    UNCALIBRATED_COUNTS_CUTOFF = 1000
    MIN_BOUNDS_MULTIPLIER = 1
    #change based on files used:
    #DETECTOR_EFFICIENCY = "2019_NAA_eff_calibration_parameters.json"
    DETECTOR_EFFICIENCY = "2019_NAA_eff_calibration_parameters.json"
    ENERGIES_TOO_CLOSE_CUTOFF = 3
    isotopes_dictionary = {
        "U-238": {
            "half_life": 4.468e9 * 365.25 * 24 * 3600,
            "molar_mass": 238.05078826,
            "daughter_isotopes": {

                "Th-234": [
                    [63.29, 4.50],   # commonly observed low-energy line
                    [92.58, 5.58],   # additional below 100 keV
                ],

                "Pa-234": [
                    # ground state / metamers combined
                    [131.3, 18.9],[883.24,10.0],[946.0,14.0],
                    [1001.03,0.84],[766.4,0.32],[742.8,0.096],
                ],

                "Pb-214": [
                    [53.2275,1.066],[241.997,7.19],[295.224,18.47],
                    [351.932,35.34],
                ],

                "Bi-214": [
                    [609.31,45.49],[934.06,3.11],[1120.28,14.92],
                    [1238.11,5.83],[1377.67,3.99],[1407.98,2.39],
                    [1729.59,2.98],[1764.49,15.30],[2204.21,4.92],
                ],

                "Tl-210": [
                    [296.0,79.0],[799.6,98.96],[1070.0,12.0],
                    [1210.0,17.0],[1316.0,21.0],
                    # Note: intensities vary in literature; these are typical
                ],

                "Hg-206": [
                    [304.896,26.0],  # weaker line but above 0.1%
                ]
            }
        },

        "U-235": {
            "half_life": 703.8e6 * 365.25 * 24 * 3600,
            "molar_mass": 235.0439299,
            "daughter_isotopes": {

                "U-235": [
                    [143.765,10.93],[163.356,5.5],[185.713,57.2],[202.12,5.0],[205.31,5.5],
                ],

                "Th-231": [
                    [25.65,13.7],
                    # Very few gamma rays; only the strongest above ~0.1% are shown
                ],

                "Pa-231": [
                    [27.36,10.5],  # low energy, often used for identification
                ],

                "Th-227": [
                    [235.96,12.9],
                    [256.2,7.2],
                ],

                "Fr-223": [
                    [50.094,34.0],[122.3,4.5],
                ],

                "Ra-223": [
                    [269.463,13.3],[154.2,5.5],
                ],

                "Rn-219": [
                    [271.23,10.8],  # only prominent gamma
                ],

                "Bi-215": [
                    [293.5,49.0],[438.6,10.0],
                ],

                "Bi-211": [
                    [351.06,12.91],[404.85,3.78],[832.01,3.52],
                ],
            }
        },

        "Th-232": {
            "half_life": 1.405e10 * 365.25 * 24 * 3600,
            "molar_mass": 232.0380553,
            "daughter_isotopes": {

                "Ac-228": [
                    [209.25,3.89],[271.24,3.46],[328.0,2.95],[338.32,11.27],
                    [463.00,4.40],[794.94,4.25],[911.20,25.8],[964.76,4.99],
                    [968.97,15.8],[1588.19,3.22],
                ],

                "Ra-224":[
                    [240.986,4.1],  # primary gamma
                ],

                "Pb-212": [
                    [115.183,0.623],[238.632,43.6],[300.09,3.18],
                ],

                "Bi-212": [
                    [727.33,6.74],[785.37,1.11],[1078.63,0.55],[1620.74,1.51],
                ],

                "Tl-208": [
                    [277.37,6.6],[510.77,22.6],[583.187,85.0],
                    [860.56,12.5],[2614.511,99.79],
                ]
            }
        },

        "Pu-239": {
            "half_life": 24110 * 365.25 * 24 * 3600,
            "molar_mass": 239.0521634,
            "daughter_isotopes": {
                # Pu-239 → U-235 series in secular equilibrium produces the same gammas as U-235 chain:
                "U-235": [[143.765,10.93],[163.356,5.5],[185.713,57.2],[202.12,5.0],[205.31,5.5]],
                "Th-231": [[25.65,13.7]],
                "Pa-231": [[27.36,10.5]],
                "Th-227": [[235.96,12.9]],
                "Fr-223": [[50.094,34.0]],
                "Ra-223": [[269.463,13.3]],
                "Rn-219": [[271.23,10.8]],
                "Bi-215": [[293.5,49.0]],
                "Bi-211": [[351.06,12.91]],
            }
        }
    }
    
    from types import BuiltinMethodType
    #finds the closest bin to a specific energy value
    def closest_bin(value, bins):
        difference = np.abs(value - bins)
        return bins[np.argmin(difference)]

    #graphs the spectrum with lines indicating energies we are checking
    def graph_spectrum(spec, energies):
        energies = np.array(energies)
        fix, ax = plt.subplots(figsize = (10, 6))
        ax.set_yscale('log')
        ax.set_title('Background Subtracted Soil Spectrum')
        print("TEST>>>>")

        print(spec.bin_centers_kev)
        ax.set_xlim(0, np.max(spec.bin_centers_kev))
        spec.plot(ax=ax)
        indexes = [closest_bin(energy, spec.bin_centers_kev) for energy in energies]
        ax.vlines(indexes, ymin=0, ymax=np.max(spec.cps_vals) * 1.5, colors = "red", linewidth=0.5)

    #graphs the spectrum zoomed in by a certain amount:
    def graph_peak(spec,energy,title,bounds,baseline):
        global unorganized_energy_graphs
        plt.figure()
        fig, ax = plt.subplots(figsize = (10,3))
        ax.set_xlim(energy-20*RADIUS,energy+20*RADIUS)
        ax.set_ylim(-0.01,0.03)
        ax.set_title(title)
        spec.plot(ax=ax)
        ax.vlines(bounds,ymin=0,ymax=np.max(spec.cps_vals) * 1.5, colors = "red", linewidth=0.5)
        ax.vlines([energy],ymin=0,ymax=np.max(spec.cps_vals) * 1.5, colors = "green", linewidth=0.5)
        plt.axhline(y=baseline, color='y', linestyle='--', label='y=5')
        filename = f"plot_{uuid.uuid4().hex}.png"
        filepath = os.path.join("static", filename)
        plt.savefig(filepath, bbox_inches="tight", dpi=200)
        plt.close()
        return filename

    def graph_bounds(spec,energy,title,peak=None,leftbound=None,rightbound=None,baseline=None):
        fig, ax = plt.subplots(figsize = (10,3))
        ax.set_xlim(energy-15*RADIUS,energy+15*RADIUS)
        ax.set_ylim(-0.01,0.02)
        ax.set_title(title)
        spec.plot(ax=ax)
        if baseline!=None:
            plt.axhline(y=baseline, color='y', linestyle='--', label='y=5')
        if peak!=None:
            ax.vlines([peak],ymin=0,ymax=np.max(spec.cps_vals) * 1.5, colors = "red", linewidth=0.5)
        if leftbound!=None and rightbound!=None:
            min_bounds = [leftbound, rightbound]
            ax.vlines(min_bounds,ymin=0,ymax=np.max(spec.cps_vals) * 1.5, colors = "green", linewidth=0.5)

    #inverts the isotopes dictionary, such that it obtains the format: {daughter_isotope_energy: [daughter_isotope_name, parent_isotope]}
    def get_inverted_isotopes_dictionary(isotopes_dictionary):
        inverted_isotopes_dictionary = {}
        for parent_isotope in isotopes_dictionary:
            for daughter_isotope in isotopes_dictionary[parent_isotope]["daughter_isotopes"]:
                for energy_and_gamma_yield in isotopes_dictionary[parent_isotope]["daughter_isotopes"][daughter_isotope]:
                    inverted_isotopes_dictionary[energy_and_gamma_yield[0]] = [daughter_isotope, parent_isotope]
        return inverted_isotopes_dictionary
    #takes dictionary of the form {ene: [daughter, parent, graph]} and returns {parent: {daughter: {ene: graph}}}
    
    def reorganize_energy_graphs(energy_graphs):
        new_energy_graphs = {}
        for ene, vals in energy_graphs.items():
            # vals should be [daughter, parent, graph]; if graph hasn't been appended yet skip
            if not isinstance(vals, (list, tuple)) or len(vals) < 3:
                continue
            daughter_isotope, parent_isotope, graph = vals[0], vals[1], vals[2]
            if parent_isotope not in new_energy_graphs:
                new_energy_graphs[parent_isotope] = {}
            if daughter_isotope not in new_energy_graphs[parent_isotope]:
                new_energy_graphs[parent_isotope][daughter_isotope] = {}
            new_energy_graphs[parent_isotope][daughter_isotope][ene] = graph
        return new_energy_graphs
                
    def get_counts(spec,energies,RADIUS,livetime):
        data = dict(zip(spec.energies_kev,spec.cps))
        counts=[]
        counts_unc=[]
        bounds=[]
        baselines=[]
        for energy in energies:
            the_energy = energy
            n=1
            max_baseline_adjusted_counts=0
            while True:
                integral=0
                radius_left = closest_bin(the_energy-RADIUS,spec.energies_kev)
                radius_right = closest_bin(the_energy+RADIUS,spec.energies_kev)
                #greatest_counts=0
                least_counts_left=data[radius_left]
                least_counts_bin_left=radius_left
                least_counts_right=data[radius_right]
                least_counts_bin_right=radius_right
                """
                for bin in data.keys():
                    if bin >= radius_left and bin <= radius_right:
                        if data[bin]>greatest_counts:
                            greatest_counts=data[bin]
                            greatest_counts_bin = bin
                for bin in data.keys():
                    if bin >= radius_left*MIN_BOUNDS_MULTIPLIER and bin <= radius_right*MIN_BOUNDS_MULTIPLIER:
                        if bin<greatest_counts_bin and data[bin]<least_counts_left:
                            least_counts_left=data[bin]
                            least_counts_bin_left = bin
                        if bin>greatest_counts_bin and data[bin]<least_counts_right:
                            least_counts_right=data[bin]
                            least_counts_bin_right = bin
                """
                for bin in data.keys():
                    if bin >= radius_left and bin <= radius_right:
                        if data[bin]<least_counts_left and bin<=the_energy:
                            least_counts_left=data[bin]
                            least_counts_bin_left = bin
                        if data[bin]<least_counts_right and bin>=the_energy:
                            least_counts_right=data[bin]
                            least_counts_bin_right = bin
                baseline_adjusted_counts=0
                baseline=0
                unadjusted_integral=0
                #if greatest_counts_bin >= radius_left and greatest_counts_bin <= radius_right:
                baseline = max(float(nominal_value(least_counts_left)),float(nominal_value(least_counts_right)))
                for bin in data.keys():
                    if bin >= least_counts_bin_left and bin <= least_counts_bin_right:
                        delta = (nominal_value(data[bin]) - baseline) * livetime
                        if delta > 0:
                            baseline_adjusted_counts += delta
                        #baseline_adjusted_counts+=(nominal_value(data[bin])-baseline)*livetime
                        unadjusted_integral+=nominal_value(data[bin])*livetime
                        print(f"bin: {bin}, counts: {nominal_value(data[bin])}, baseline: {baseline}, baseline adjusted counts: {(nominal_value(data[bin])-baseline)*livetime}")    
                if baseline_adjusted_counts > max_baseline_adjusted_counts:
                    max_baseline_adjusted_counts=baseline_adjusted_counts
                if baseline_adjusted_counts > MIN_PEAK_COUNTS:
                    integral = baseline_adjusted_counts
                    print(f"Found sufficient counts for peak at {energy} keV: {integral} counts (baseline subtracted).")
                    break
                elif abs(energy-the_energy) > CHECK_LIMIT:
                    integral = max_baseline_adjusted_counts
                    print(f"Warning: could not find sufficient counts for peak at {energy} keV; using maximum found value of {max_baseline_adjusted_counts} counts.")   
                    break
                else:
                    the_energy+=0.1*(-1)**n*(n+1)
                    n+=1
            counts.append(integral)
            bounds.append([least_counts_bin_left,least_counts_bin_right])
            counts_unc.append(math.sqrt(abs(integral)))
            baselines.append(baseline)
            # graphs are created in get_isotopes_info tpyo include more context in titles; avoid duplicating here
        return [counts,counts_unc,bounds,baselines]

    # returns the predicted parent isotope mass and mass uncertainty based on the daughter isotope counts and counts uncertainty
    def get_mass_prediction(parent_isotope,daughter_isotope,energy,counts,unc,livetime):
        gamma_yield = 0
        for daughter_energy, gamma_yield_value in isotopes_dictionary[parent_isotope]["daughter_isotopes"][daughter_isotope]:
            if daughter_energy == energy:
                gamma_yield = gamma_yield_value
                break
        if gamma_yield == 0:
            print(f"Warning: gamma yield not found for {parent_isotope} -> {daughter_isotope} at {energy} keV; skipping mass prediction.")
            return [0,0]
        decay_constant=math.log(2) / isotopes_dictionary[parent_isotope]['half_life']
        molar_mass=isotopes_dictionary[parent_isotope]['molar_mass']
        #assuming secular equilibrium (parent isotope counts = daughter isotope counts)
        predicted_parent_mass = counts * molar_mass / (NA * gamma_yield * decay_constant * livetime)
        predicted_parent_mass_uncertainty = unc * molar_mass / (NA * gamma_yield * decay_constant * livetime)
        return [predicted_parent_mass,predicted_parent_mass_uncertainty]

    def energy_close_to_another(parent_isotope,daughter_isotope,energy):
        for parent in isotopes_dictionary:
            for daughter in isotopes_dictionary[parent]["daughter_isotopes"]:
                for energy_yield in isotopes_dictionary[parent]["daughter_isotopes"][daughter]:
                    energy2 = energy_yield[0]
                    if daughter_isotope != daughter and energy2 != energy and abs(energy2-energy)<ENERGIES_TOO_CLOSE_CUTOFF and (parent_isotope=="U-235" or parent_isotope=="Pu-239"):
                        return True
        return False
                        
    #returns a dictionary with format... parent isotope 1: {daugther isotope energy 1: [daughter isotope, counts, unc, predicted parent mass, predicted parent mass unc]}... for each parent and daughter isotope energy believed to be in the sample
    def get_isotopes_info(spec, bg, isotopes_dictionary,efficiency):
        eff_func = am.Efficiency()
        eff_func.set_parameters(efficiency)
        inverted_isotopes_dictionary = get_inverted_isotopes_dictionary(isotopes_dictionary)
        energies = list(inverted_isotopes_dictionary.keys())
        energy_graphs = get_inverted_isotopes_dictionary(isotopes_dictionary)
        subtracted_spec=spec-bg
        counts, counts_unc, bounds, baselines = get_counts(subtracted_spec,energies,RADIUS,spec.livetime)
        #gets the counts and count uncertainties
        #calibrates counts and counts_unc to detector efficiency (guard against zero efficiency)
        uncalibrated_counts = counts
        calibrated_counts = []
        calibrated_counts_unc = []
        for i in range(len(energies)):
            eff = eff_func.get_eff(energies[i])
            if eff == 0 or eff is None:
                print(f"Warning: zero detector efficiency for energy {energies[i]} keV; setting calibrated counts to 0.")
                calibrated_counts.append(0)
                calibrated_counts_unc.append(0)
            else:
                calibrated_counts.append(counts[i]/eff)
                calibrated_counts_unc.append(counts_unc[i]/eff)
        counts = calibrated_counts
        counts_unc = calibrated_counts_unc
        #creates dictionary containing isotope counts and mass info
        isotopes_in_sample_info = {}
        """
        pf = PF(energies, spec, bg)
        pf_uncalibrated_counts, pf_counts_unc = pf.get_counts()
        pf_counts = []
        for i in range(len(energies)):
            eff = eff_func.get_eff(energies[i])
            if eff == 0 or eff is None:
                pf_counts.append(0)
            else:
                pf_counts.append(pf_uncalibrated_counts[i]/eff)
        """
        for i in range(len(energies)):
            title=""
            ene, unc, daughter_counts, uncalibrated_daughter_counts, daughter_bounds, baseline = energies[i], counts_unc[i], counts[i], uncalibrated_counts[i], bounds[i], baselines[i]
            
            #pf_uncalibrated_daughter_counts, pf_daughter_counts, pf_daughter_counts_unc = pf_uncalibrated_counts[i], pf_counts[i], pf_counts_unc[i]
            parent_isotope, daughter_isotope = inverted_isotopes_dictionary[ene][1], inverted_isotopes_dictionary[ene][0]
            predicted_parent_mass, predicted_parent_mass_unc = get_mass_prediction(parent_isotope,daughter_isotope,ene,daughter_counts,unc,spec.livetime)
            #pf_predicted_parent_mass, pf_predicted_parent_mass_unc = get_mass_prediction(parent_isotope,daughter_isotope,ene,pf_daughter_counts,pf_daughter_counts_unc,spec.livetime)
            if parent_isotope not in isotopes_in_sample_info:
                isotopes_in_sample_info[parent_isotope] = {}
            #if daughter_counts!=0 and uncalibrated_daughter_counts>=UNCALIBRATED_COUNTS_CUTOFF:
            isotopes_in_sample_info[parent_isotope][ene]={"daughter_isotope":daughter_isotope, "counts": daughter_counts, "counts unc": unc, "predicted_parent_mass": predicted_parent_mass, "predicted_parent_mass_unc": predicted_parent_mass_unc}
            #title+=f"ene: {ene}keV, parent: {parent_isotope}, daughter: {daughter_isotope}, uncalibrated counts: {uncalibrated_daughter_counts} counts: {daughter_counts:.2e}, uncertainty: {unc}, unc/ud_counts: {unc/uncalibrated_daughter_counts}, predicted parent mass: {predicted_parent_mass:.2e}g"
            #graph_peak(subtracted_spec,ene,title,RADIUS)
            #print(f"Results for {ene}keV peak using PF: uncalibrated counts: {pf_uncalibrated_daughter_counts}, counts: {pf_daughter_counts}, counts_unc: {pf_daughter_counts_unc}, predicted_parent_mass: {pf_predicted_parent_mass}g, predicted_parent_mass_unc: {pf_predicted_parent_mass_unc}g")
            #pf.plot_roi(ene)
            title = f"{ene} daughter_counts: {daughter_counts} unc: {unc} uncalibrated_daughter_counts: {uncalibrated_daughter_counts} predicted_parent_mass: {predicted_parent_mass} predicted_parent_mass_unc: {predicted_parent_mass_unc}"
            graph = graph_peak(subtracted_spec, ene, title, daughter_bounds, baseline)
            energy_graphs[ene].append(graph)
        energy_graphs = reorganize_energy_graphs(energy_graphs)
        return isotopes_in_sample_info,energy_graphs
    # returns a dictionary with parent isotopes and their estimated masses and uncertainties
    def estimate_aggregated_masses_and_uncertainties(isotopes_info, isotopes_dictionary):
        flagged_isotopes = []
        estimated_parent_masses = {}
        for parent_isotope in isotopes_info:
            sum=0
            denominator=0
            if len(isotopes_info[parent_isotope]) < MAX_DAUGHTERS_TO_FLAG_PARENT:
                flagged_isotopes.append(parent_isotope)

            for ene in isotopes_info[parent_isotope]:
                daughter_isotope, counts, unc, predicted_parent_mass, predicted_parent_mass_unc = isotopes_info[parent_isotope][ene].values()
                if unc==0 or predicted_parent_mass_unc==0:
                    predicted_parent_mass_unc=1e12 #arbitrarily large uncertainty to effectively ignore this value
                sum += predicted_parent_mass/predicted_parent_mass_unc**2
                denominator += 1/predicted_parent_mass_unc**2
                    
            if len(isotopes_info[parent_isotope])!=0:
                if denominator != 0:
                    sum /= denominator
                    total_unc = 1/denominator**0.5
                else:
                    # no valid entries with non-zero uncertainty -> no estimate
                    sum = 0
                    total_unc = 0

            estimated_parent_masses[parent_isotope] = [sum,total_unc]
        return [estimated_parent_masses,flagged_isotopes]
    
    def get_daughter_values_uncs(daughter_isotopes_info):
        values=[]
        errors=[]
        for daughter_isotope in daughter_isotopes_info:
            sum=0
            denominator=0
            total_unc=0
            for energy in daughter_isotopes_info[daughter_isotope]:
                parent_mass=energy[1]
                parent_mass_unc=energy[2]
                if parent_mass_unc!=0:
                    sum+=parent_mass/parent_mass_unc**2
                    denominator+=1/parent_mass_unc**2
            if denominator != 0:
                sum/=denominator
                total_unc = 1/denominator**0.5
            values.append(sum)
            errors.append(total_unc)
        return values, errors
    
    def create_results_graph(results):
        labels = [x[0] for x in results]
        values = [x[1] for x in results]
        errors = [x[2] for x in results]
        plt.figure()
        plt.bar(labels,values,yerr=errors,capsize=5)
        plt.xlabel("Parent Isotopes")
        plt.ylabel("Estimated Mass (g)")
        plt.tight_layout()
        plt.show()
        filename = f"plot_{uuid.uuid4().hex}.png"
        filepath = os.path.join("static", filename)
        plt.savefig(filepath, bbox_inches="tight", dpi=200)
        plt.close()
        return filename
    
    def create_daughters_graphs(isotopes_info):
        daughters_graphs = {}
        for parent in isotopes_info:
            daughter_isotopes_info = {}
            for ene in isotopes_info[parent]:
                daughter_isotope = isotopes_info[parent][ene]["daughter_isotope"]
                predicted_parent_mass = isotopes_info[parent][ene]["predicted_parent_mass"]
                predicted_parent_mass_unc = isotopes_info[parent][ene]["predicted_parent_mass_unc"]
                if not(daughter_isotope in daughter_isotopes_info):
                    daughter_isotopes_info[daughter_isotope] = []
                daughter_isotopes_info[daughter_isotope].append([ene,predicted_parent_mass,predicted_parent_mass_unc])
            labels = list(daughter_isotopes_info.keys())
            values, errors = get_daughter_values_uncs(daughter_isotopes_info)
            plt.figure()
            plt.bar(labels,values,yerr=errors,capsize=5)
            plt.title("Daughter Isotope Mass Predictions for "+parent)
            plt.xlabel("Daughter Isotopes")
            plt.ylabel("Estimated Mass (g)")
            plt.tight_layout()
            filename = f"plot_{uuid.uuid4().hex}.png"
            filepath = os.path.join("static", filename)
            plt.savefig(filepath, bbox_inches="tight", dpi=200)
            plt.close()
            daughters_graphs[parent] = filename

        return daughters_graphs
        
    def get_daughters_energies():
        daughters_energies={}
        for parent in isotopes_dictionary:
            # iterate the daughter_isotopes mapping for each parent
            for daughter in isotopes_dictionary[parent]["daughter_isotopes"]:
                if daughter not in daughters_energies:
                    daughters_energies[daughter]=[]
                for energy_and_yield in isotopes_dictionary[parent]["daughter_isotopes"][daughter]:
                    # energy_and_yield is [energy, gamma_yield]
                    energy = energy_and_yield[0]
                    if energy not in daughters_energies[daughter]:
                        daughters_energies[daughter].append(energy)
        return daughters_energies
    
    def get_only_energies_graphs(energies_graphs):
        only_energies_graphs={}
        for parent in energies_graphs:
            for daughter in energies_graphs[parent]:
                for energy in energies_graphs[parent][daughter]:
                    only_energies_graphs[energy]=energies_graphs[parent][daughter][energy]
        return only_energies_graphs
    
    def get_key_graphs(results):
        desired = ["U-238","U-235"]
        result_dict = {r[0]: r[1] for r in results}
        labels = [l for l in desired if l in result_dict]
        values = [result_dict[l] for l in labels]
        # if there are no matching values, skip plotting
        if not values:
            return None
        plt.figure()
        plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title("Uranium Isotope Composition")
        plt.tight_layout()
        filename = f"plot_{uuid.uuid4().hex}.png"
        filepath = os.path.join("static", filename)
        plt.savefig(filepath, bbox_inches="tight", dpi=200)
        plt.close()
        return filename
    
    def remove_close_energies(isotopes_dictionary):
        new_isotopes_dictionary = {}
        for parent in isotopes_dictionary:
            new_isotopes_dictionary[parent] = {"half_life": isotopes_dictionary[parent]["half_life"], "molar_mass": isotopes_dictionary[parent]["molar_mass"], "daughter_isotopes": {}}
            for daughter in isotopes_dictionary[parent]["daughter_isotopes"]:
                new_isotopes_dictionary[parent]["daughter_isotopes"][daughter] = []
                for energy_yield in isotopes_dictionary[parent]["daughter_isotopes"][daughter]:
                    energy = energy_yield[0]
                    if not(energy_close_to_another(parent, daughter, energy)):
                        new_isotopes_dictionary[parent]["daughter_isotopes"][daughter].append(energy_yield)
        print(new_isotopes_dictionary)
        return new_isotopes_dictionary

    # This is the function for the main program execution. calls the above functions to get the isotopes info and convert it into masses. Prints out the isotope masses.
    def analyze_the_spectrum(filename,spec,bg,efficiency,sample_mass):
        #spec.calibrate_like(bg)
        bg = bg.rebin(spec.bin_edges_kev)
        isotopes_info,energy_graphs = get_isotopes_info(spec,bg,isotopes_dictionary,efficiency)
        masses,flagged_isotopes = estimate_aggregated_masses_and_uncertainties(isotopes_info, isotopes_dictionary)
        returnstatement={}
        returnstatement["results"]=[]
        for parent_isotope in masses:
            mass,unc = masses[parent_isotope]
            returnstatement["results"].append([parent_isotope,mass,unc])
            #printstatement+=f"\n{parent_isotope}: mass: {mass:.2e}g, unc: {unc:.2e}"
            #printstatement+=f"\nflagged isotopes: {flagged_isotopes}"
        returnstatement["isotopes_info"]=isotopes_info
        returnstatement["isotopes_dictionary"]=isotopes_dictionary
        returnstatement["results_graph"] = create_results_graph(returnstatement["results"])
        returnstatement["daughters_graphs"] = create_daughters_graphs(isotopes_info)
        returnstatement["energy_graphs"] = energy_graphs
        returnstatement["only_energies_graphs"] = get_only_energies_graphs(energy_graphs)
        returnstatement["daughters_energies"] = get_daughters_energies()
        returnstatement["key_graphs"] = get_key_graphs(returnstatement["results"])
        returnstatement["conclusions"] = "no conclusions yet"
        return returnstatement
    
    #isotopes_dictionary = remove_close_energies(isotopes_dictionary)
    return analyze_the_spectrum(spectrum_path,Spectrum.from_file(spectrum_path),Spectrum.from_file(background_path),efficiency_path,"mass unknown")

