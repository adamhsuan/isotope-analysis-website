"""
This file contains the main analysis functions for the web app which take spectrum and background
files as input and return the estimated parent isotope masses and graphs to be rendered in the results page. 
The main function is analyze spectrum, which calls the other functions to get the isotope info, create the graphs,
and return the result in a format that can be easily rendered in the template.
"""

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
import analysis_methods as am
import math
from uncertainties import nominal_value
import pandas as pd
import os
import uuid
from types import BuiltinMethodType

NA = 6.022e23
RADIUS = 2
MIN_PEAK_COUNTS = 1000
CHECK_LIMIT=4
ENERGIES_TOO_CLOSE_CUTOFF = 3

#isotopes_dictionary contains info about parent isotopes and their daughter isotopes, which is used to estimate parent isotoepe masses
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

""" the following functions are used to estimate the masses of parent isotopes based on the gamma emmisions of their daughter isotopes
"""

#helper function for get_counts. finds the closest bin to a specific energy value
def closest_bin(value, bins):
    difference = np.abs(value - bins)
    return bins[np.argmin(difference)]

#get_counts determines the counts under the peaks for each energy.
def get_counts(spec,energies,RADIUS,livetime):

    #list containing the energy and cps per energy bin for the spectrum
    data = dict(zip(spec.energies_kev,spec.cps))
    
    counts=[]
    counts_unc=[]
    
    #bounds are the energy values of the left and right of the peak. We keep track of them for graphing
    bounds=[]
    
    #baselines are floors subtracted away from the peak to avoid counting background noise. We keep track of them for graphing as well.
    baselines=[]

    for energy in energies:

        #the_energy is the current peak center we are checking. we will adjust this value if we find a stronger peak nearby
        the_energy = energy
        
        #n is used to adjust the energy in a pattern spiraling outward from the original energy value, to find the strongest peak
        n=1
        
        #max_baseline_adjusted_counts keeps track of the strongest peak found
        max_baseline_adjusted_counts=0

        while True:
            
            integral=0

            #radius is the maximum distance from the current energy value we check for the bounds of the peak
            radius_left = closest_bin(the_energy-RADIUS,spec.energies_kev)
            radius_right = closest_bin(the_energy+RADIUS,spec.energies_kev)

            least_counts_left=data[radius_left]
            least_counts_bin_left=radius_left
            least_counts_right=data[radius_right]
            least_counts_bin_right=radius_right

            for bin in data.keys():
                #checks the cps at each bin within the radius of the current energy value
                if bin >= radius_left and bin <= radius_right:
                    #if bin on the left side of the peak and has less cps than the least counts on the left, move the least counts outward
                    if bin<=the_energy and data[bin]<least_counts_left:
                        least_counts_left=data[bin]
                        least_counts_bin_left = bin
                    #if bin on the right side of the peak and has less cps than the least counts on the right, move the least counts outward
                    if bin>=the_energy and data[bin]<least_counts_right:
                        least_counts_right=data[bin]
                        least_counts_bin_right = bin

            baseline_adjusted_counts=0
            baseline=0
            unadjusted_integral=0

            #baseline is set as the higher of the least counts on either side of the peak
            baseline = max(float(nominal_value(least_counts_left)),float(nominal_value(least_counts_right)))

            #iterates over the bins within the radius, adds the baseline adjusted counts for the particular energy (delta) to the integral
            for bin in data.keys():
                if bin >= least_counts_bin_left and bin <= least_counts_bin_right:
                    delta = (nominal_value(data[bin]) - baseline) * livetime
                    if delta > 0:
                        baseline_adjusted_counts += delta
                    #unadjusted_integral keeps track of the total counts without baseline adjustement
                    unadjusted_integral+=nominal_value(data[bin])*livetime

            #checks if the baseline adjusted counts for this energy is the strongest peak so far and updates if so
            if baseline_adjusted_counts > max_baseline_adjusted_counts:
                max_baseline_adjusted_counts=baseline_adjusted_counts
            
            #if the baseline adjusted counts is above the minimum peak counts, we consider it a peak and stop spiraling outward
            if baseline_adjusted_counts > MIN_PEAK_COUNTS:
                integral = baseline_adjusted_counts
                break

            #if we reach the end of the spiral pattern without finding a strong enough peak, we take the strongest peak found
            elif abs(energy-the_energy) > CHECK_LIMIT:
                integral = max_baseline_adjusted_counts
                break

            #otherwise continue spiraling out searching for peaks
            else:
                the_energy+=0.1*(-1)**n*(n+1)
                n+=1

        counts.append(integral)
        bounds.append([least_counts_bin_left,least_counts_bin_right])
        counts_unc.append(math.sqrt(abs(integral)))
        baselines.append(baseline)

    return [counts,counts_unc,bounds,baselines]

# returns the predicted parent isotope mass and mass uncertainty based on the daughter isotope counts and counts uncertainty
def get_mass_prediction(parent_isotope,daughter_isotope,energy,counts,unc,livetime):

    #gets the gamma yeild for the energy
    gamma_yield = 0
    for daughter_energy, gamma_yield_value in isotopes_dictionary[parent_isotope]["daughter_isotopes"][daughter_isotope]:
        if daughter_energy == energy:
            gamma_yield = gamma_yield_value
            break

    if gamma_yield == 0:
        return [0,0]
    
    #decay constant is calculated based on the half life of the parent isotope
    decay_constant=math.log(2) / isotopes_dictionary[parent_isotope]['half_life']

    molar_mass=isotopes_dictionary[parent_isotope]['molar_mass']
    
    predicted_parent_mass = counts * molar_mass / (NA * gamma_yield * decay_constant * livetime)
    predicted_parent_mass_uncertainty = unc * molar_mass / (NA * gamma_yield * decay_constant * livetime)

    return [predicted_parent_mass,predicted_parent_mass_uncertainty]

#helper function for get_isotopes_info. inverts the isotopes dictionary, such that it obtains the format: {daughter_isotope_energy: [daughter_isotope_name, parent_isotope]}
def get_inverted_isotopes_dictionary(isotopes_dictionary):
    inverted_isotopes_dictionary = {}
    for parent_isotope in isotopes_dictionary:
        for daughter_isotope in isotopes_dictionary[parent_isotope]["daughter_isotopes"]:
            for energy_and_gamma_yield in isotopes_dictionary[parent_isotope]["daughter_isotopes"][daughter_isotope]:
                inverted_isotopes_dictionary[energy_and_gamma_yield[0]] = [daughter_isotope, parent_isotope]
    return inverted_isotopes_dictionary

#helper function for get_isotopes_info. reorganizes the energy graphs into a nested dictionary format for easier access in the template
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

#returns a dictionary with format... parent isotope 1: {daugther isotope energy 1: [daughter isotope, counts, unc, predicted parent mass, predicted parent mass unc]}... for each parent and daughter isotope energy believed to be in the sample
def get_isotopes_info(spec, bg, isotopes_dictionary,efficiency):
    #sets up the efficiency function
    eff_func = am.Efficiency()
    eff_func.set_parameters(efficiency)

    #puts isotopes_dictionary in an order that eases use for the rest of the function
    inverted_isotopes_dictionary = get_inverted_isotopes_dictionary(isotopes_dictionary)

    energies = list(inverted_isotopes_dictionary.keys())
    energy_graphs = get_inverted_isotopes_dictionary(isotopes_dictionary)

    subtracted_spec=spec-bg

    #gets the counts at each energy
    counts, counts_unc, bounds, baselines = get_counts(subtracted_spec,energies,RADIUS,spec.livetime)

    #calibrates counts and counts_unc to detector efficiency (guard against zero efficiency)
    uncalibrated_counts = counts

    calibrated_counts = []
    calibrated_counts_unc = []

    #gets the actual counts based on the detector counts by incorperating the efficiency function
    for i in range(len(energies)):
        eff = eff_func.get_eff(energies[i])
        if eff == 0 or eff is None:
            calibrated_counts.append(0)
            calibrated_counts_unc.append(0)
        else:
            calibrated_counts.append(counts[i]/eff)
            calibrated_counts_unc.append(counts_unc[i]/eff)

    #sets the counts and counts_unc variables equal to the actual counts and counts_unc
    counts = calibrated_counts
    counts_unc = calibrated_counts_unc

    #creates dictionary containing isotope counts and mass info
    isotopes_in_sample_info = {}
    for i in range(len(energies)):
        title=""
        #gets the isotopes info values for each energy
        ene, unc, daughter_counts, uncalibrated_daughter_counts, daughter_bounds, baseline = energies[i], counts_unc[i], counts[i], uncalibrated_counts[i], bounds[i], baselines[i]
        
        #gets the parent and daughter isotopes for the energy to construct the dictionary
        parent_isotope, daughter_isotope = inverted_isotopes_dictionary[ene][1], inverted_isotopes_dictionary[ene][0]
        
        #gets the predicted parent mass and unc based on the isotopes info for that energy
        predicted_parent_mass, predicted_parent_mass_unc = get_mass_prediction(parent_isotope,daughter_isotope,ene,daughter_counts,unc,spec.livetime)
        
        #creates the dictionary by adding isotope info by energy to parent isotope
        if parent_isotope not in isotopes_in_sample_info:
            isotopes_in_sample_info[parent_isotope] = {}
        isotopes_in_sample_info[parent_isotope][ene]={"daughter_isotope":daughter_isotope, "counts": daughter_counts, "counts unc": unc, "predicted_parent_mass": predicted_parent_mass, "predicted_parent_mass_unc": predicted_parent_mass_unc}
        
        #creates another dictionary with the energy graphs for each energy
        title = f"{ene} daughter_counts: {daughter_counts:.2e} unc: {unc:.2e} uncalibrated_daughter_counts: {uncalibrated_daughter_counts:.2e} predicted_parent_mass: {predicted_parent_mass:.2e} predicted_parent_mass_unc: {predicted_parent_mass_unc:.2e}"
        graph = graph_peak(subtracted_spec, ene, title, daughter_bounds, baseline)
        energy_graphs[ene].append(graph)

    # reorganizes the energy graph data structure for easier access in the template    
    energy_graphs = reorganize_energy_graphs(energy_graphs)

    return isotopes_in_sample_info,energy_graphs

# returns a dictionary with parent isotopes and their estimated masses and uncertainties
def estimate_aggregated_masses_and_uncertainties(isotopes_info):
    estimated_parent_masses = {}
    for parent_isotope in isotopes_info:
        sum=0
        denominator=0
        
        #loops through each energy corresponding to a parent isotope, and combines the predicted parent mass values and uncertainties while maintaining error propegation
        for ene in isotopes_info[parent_isotope]:
            entry = isotopes_info[parent_isotope][ene]
            daughter_isotope = entry["daughter_isotope"]
            counts = entry["counts"]
            unc = entry["counts unc"]
            predicted_parent_mass = entry["predicted_parent_mass"]
            predicted_parent_mass_unc = entry["predicted_parent_mass_unc"]
            if unc == 0 or predicted_parent_mass_unc == 0:
                predicted_parent_mass_unc = 1e12  # arbitrarily large uncertainty to effectively ignore this value
            sum += predicted_parent_mass / predicted_parent_mass_unc ** 2
            denominator += 1 / predicted_parent_mass_unc ** 2
                
        # ensures no division by zero. completes average via error propegation formula
        if len(isotopes_info[parent_isotope])!=0:
            if denominator != 0:
                sum /= denominator
                total_unc = 1/denominator**0.5
            else:
                # no valid entries with non-zero uncertainty -> no estimate
                sum = 0
                total_unc = 0

        estimated_parent_masses[parent_isotope] = [sum,total_unc]
    return estimated_parent_masses

#helper function for remove_close energies. Checks if a given energy is within a certain cutoff of another energy from a different daughter isotope of the same parent (only for U-235 and Pu-239, where there are many low-energy gammas that can be close to each other and cause confusion in analysis)
def energy_close_to_another(parent_isotope,daughter_isotope,energy):
    for parent in isotopes_dictionary:
        for daughter in isotopes_dictionary[parent]["daughter_isotopes"]:
            for energy_yield in isotopes_dictionary[parent]["daughter_isotopes"][daughter]:
                energy2 = energy_yield[0]
                if daughter_isotope != daughter and energy2 != energy and abs(energy2-energy)<ENERGIES_TOO_CLOSE_CUTOFF and (parent_isotope=="U-235" or parent_isotope=="Pu-239"):
                    return True
    return False

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

""" the following functions are for creating graphs
"""

#graphs the spectrum with lines indicating energies we are checking
def graph_spectrum(spec, energies):
    energies = np.array(energies)
    fix, ax = plt.subplots(figsize = (10, 6))
    ax.set_yscale('log')
    ax.set_title('Background Subtracted Soil Spectrum')
    ax.set_xlim(0, np.max(spec.bin_centers_kev))
    spec.plot(ax=ax)
    indexes = [closest_bin(energy, spec.bin_centers_kev) for energy in energies]
    ax.vlines(indexes, ymin=0, ymax=np.max(spec.cps_vals) * 1.5, colors = "red", linewidth=0.5)
    filename = f"plot_{uuid.uuid4().hex}.png"
    filepath = os.path.join("static", filename)
    plt.savefig(filepath, bbox_inches="tight", dpi=200)
    plt.close()
    return filename

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

#helper function for create_daughters_graphs. takes the daughter isotopes info and returns lists of the predicted parent mass values and uncertainties for each daughter isotope, which can then be plotted in a graph with error bars
def get_daughter_mass_predictions_uncs(daughter_isotopes_info):
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
        values, errors = get_daughter_mass_predictions_uncs(daughter_isotopes_info)
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

""" the following function creates new data structures based on the isotopes info for easier rendering in the template
"""
def get_daughters_energies():
    daughters_energies={}
    for parent in isotopes_dictionary:
        # iterate the daughter_isotopes mapping for each parent
        for daughter in isotopes_dictionary[parent]["daughter_isotopes"]:
            if daughter not in daughters_energies:
                daughters_energies[daughter]=[]
            for energy_and_yield in isotopes_dictionary[parent]["daughter_isotopes"][daughter]:
                # energy_and_yield is [energy, gamma_yield]
                energy = str(energy_and_yield[0])
                if energy not in daughters_energies[daughter]:
                    daughters_energies[daughter].append(energy)
    return daughters_energies

""" analyze_the_spectrum gets the isotope info and creates the graphs, then returns it into app.py to be rendered in the results page
"""
def analyze_spectrum(spectrum_path,background_path,efficiency):
    #uncomment the following line to remove energies that are close to each other
    #isotopes_dictionary = remove_close_energies(isotopes_dictionary)

    #creates spectrum objects from the spectrum and background files
    spec = Spectrum.from_file(spectrum_path)
    bg = Spectrum.from_file(background_path)
    
    #calibrates background and and spectrum to match energy bins
    bg = bg.rebin(spec.bin_edges_kev)

    #gets isotopes info, which is the counts and predicted parent masses from each daughter energy
    isotopes_info,energy_graphs = get_isotopes_info(spec,bg,isotopes_dictionary,efficiency)

    #estimates the parent masses based on the masses of the daughter isotopes
    masses = estimate_aggregated_masses_and_uncertainties(isotopes_info)

    returnstatement={}

    #spectrum graph is a graph that displays the entire spectrum counts and energies
    returnstatement["spectrum_graph"] = graph_spectrum(spec, list(get_inverted_isotopes_dictionary(isotopes_dictionary).keys()))

    #results is used to list the masses and uncertainties of the parent isotopes
    returnstatement["results"]=[]
    for parent_isotope in masses:
        mass,unc = masses[parent_isotope]
        returnstatement["results"].append([parent_isotope,mass,unc])

    #results graph displays the mass predictions in a bar graph
    returnstatement["results_graph"] = create_results_graph(returnstatement["results"])

    #daughters graphs are the graphs corresponding to each parent isotope which showing the mass predictions per daughter isotope
    returnstatement["daughters_graphs"] = create_daughters_graphs(isotopes_info)

    #daughters_energies gives the energies for each daughter isotope, used to create buttons for eneriges that pop up when daughter energy button is clicked
    returnstatement["daughters_energies"] = get_daughters_energies()

    #energy graphs are the graphs of individual energies, zoomed in on the spectrum. only_energies_graphs contains that info in a different data structure (energy->graph) for ease of access 
    returnstatement["energy_graphs"] = energy_graphs
    returnstatement["only_energies_graphs"] = get_only_energies_graphs(energy_graphs)

    #key graphs contains important graphs like U-235 to U-238 ratio which are displayed seperately
    returnstatement["key_graphs"] = get_key_graphs(returnstatement["results"])
    returnstatement["conclusions"] = "no conclusions yet"

    #the following keys of returnstatement contain info for the debug panel 
    returnstatement["isotopes_info"]=isotopes_info
    returnstatement["isotopes_dictionary"]=isotopes_dictionary

    return returnstatement

