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
MIN_PEAK_COUNTS = 3000
CHECK_LIMIT=4
ENERGIES_TOO_CLOSE_CUTOFF = 4
MIN_ENERGY_CUTOFF = 60 

#isotopes_dictionary contains info about parent isotopes and their daughter isotopes, which is used to estimate parent isotoepe masses
isotopes_dictionary = {
    "U-238": {
        "half_life": 4.468e9 * 365.25 * 24 * 3600,
        "molar_mass": 238.05078826,
        "daughter_isotopes": {
            #[decay energy, gamma yield]
            "Th-234":  [[63.29,3.7],[92.38,2.13],[92.80,2.1]],
            "Pa-234": [[131.30,18.9],[946.00,14.0],[883.24,10.0],[569.5,8.6],[925.0,8.2],[926.72,7.6],[733.39,7.2],[880.5,6.5],[152.71,6.3],[227.25,6.0],[226.5,4.4],[880.5,4.4],[831.5,4.3],[699.03,3.8],[568.9,3.8],[898.67,3.4],[99.86,3.3],[293.79,3.1]],
            "Pb-214": [[351.9320,35.72],[295.224,18.47],[241.995,7.26]],
            "Bi-214": [[609.321,45.44],[1764.491,15.29],[1120.294,14.90],[1238.122,5.83],[2204.10,4.92],[768.360,4.89],[1377.669,3.985],[934.056,3.094]],
            "Tl-210": [[799.6,98.96],[296,79],[1316,21],[1210,17],[1070,12],[2430,9],[2360,8],[2010,6.9],[860,6.9],[1110,6.9],[1410,4.9],[2090,4.9],[356,4.0],[97,4.0],[2270,3.0],[910,3.0],[382,3.0]],
            "Pb-210": [[46.539,4.26]],
            "Hg-206": [[304.896,26.0]]
        }
    },

    "U-235": {
        "half_life": 703.8e6 * 365.25 * 24 * 3600,
        "molar_mass": 235.0439299,
        "daughter_isotopes": {

            "U-235": [
                #removed 184.713
                #[143.765,10.93],[163.356,5.5],[185.713,57.2],[202.12,5.0],[205.31,5.5],
                [185.713,57.2],[143.765,10.93],[163.357,5.04],[205.311,5.03]],
            "Th-231": [[25.65,13.7],[84.2140,6.80]],
            "Pa-231": [
                [27.36,10.5],
            ],

            "Th-227": [
                #temporarily removed
                #[235.96,12.9],
                [50.13,8.4],[256.23,7.0]
            ],

            "Fr-223": [
                [50.094,34.0],[79.651,8.7],
                #removed 234.75
                #[234.75,3.0]
            ],

            "Ra-223": [
                [269.463,13.3],[154.208,6.02],[323.871,3.64],[144.235,3.47]
            ],

            "Rn-219": [
                [271.23,10.8],[401.81,6.6]
            ],

            "Bi-215": [
                [293.5,49.0],[271.1,4.0],[1104.5,3.1]
            ],
            "Pb-211": [
                [404.853,3.78],[832.01,3.52]
            ],
            "Bi-211": [
                #temporarily removed
                #[351.06,12.91],
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
                [240.986,4.1],
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
    }

}




"""
Pu-239 not in secular equilibrium, so can't be used
"Pu-239": {
    "half_life": 24110 * 365.25 * 24 * 3600,
    "molar_mass": 239.0521634,
    "daughter_isotopes": {
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
"""

#helper function for get_counts. finds the closest bin to a specific energy value

def closest_bin(value, bins):
    difference = np.abs(value - bins)
    return bins[np.argmin(difference)]

#get_energy_counts determines the counts under the peaks for each energy.
"""
- iterates through each expected energy corresponding to a gamma emmision
  - beginning at the expected energy, iterates spiraling outward checking for peaks centered at energies further and further out, until CHECK_LIMIT is reached (the maximum distance the peak center can be from the expected energy)
    - determines the maximum peak radius i.e. half width (function of energy)
    - sets the bounds of integration as the closest local minima on either side of the central energy bin, within the max peak radius. if no minimum, set to max peak radius.
    - sets the baseline as the average of the cps of the bounds
    - integrates over the region, subtracting out the baseline.
    - converts cps to counts, incorperating efficiency.
    - checks whether counts is greater than max_counts, and if so sets max_counts to counts
  - adds max_counts to energy_counts, a list containing the counts for all the enegies
- returns energy_counts as well as lists containing uncertaintines, bounds, and baselines
Note: counts returned are not adjusted for efficiency. This adjustment is done later in get_isotopes_info
"""

def get_counts(spec,bg,energies,livetime):

    subtracted_spec = spec-bg

    #lists containing the energy and cps per energy bin for the spec, bg, and subtracted_spec
    spectrum_data = dict(zip(spec.energies_kev,spec.cps))
    background_data = dict(zip(bg.energies_kev,bg.cps))
    subtracted_spectrum_data = dict(zip(subtracted_spec.energies_kev,subtracted_spec.cps))

    energies_counts=[]
    energies_counts_unc=[]

    #bounds are the energy values of the left and right of the peak. We keep track of them for graphing
    bounds=[]

    #baselines are floors subtracted away from the peak to avoid counting background noise. We keep track of them for graphing as well.
    baselines=[]

    #peak_energies are the energies of the actual peaks found (not the energies from the dictionary)
    peak_energies=[]

    #stores lists containing close energies for each energy
    close_energies_list = []

    for energy in energies:
        #num close peaks tracks the number of peaks that are close and discounts those energies in uncertainty calculation (we don't know for sure that that peak came from the energy)
        close_energies = []

        #max_radius is the maximum half-width of the peak we're checking. The bounds of integration may reside inside the radius of a local minimum occurs
        max_radius = 1.06*math.sqrt(0.4+0.0024*energy)

        """
        #max_counts and max_counts_unc keep track of the strongest peak found
        max_counts=0
        max_counts_unc=0
        """

        #dictionary containing the counts under each peak centered at a given energy (dictionary key)
        peak_counts_per_energy = {}

        #the_energy is the current peak center we are checking. we start at the left side of the check limit and work out way to the right
        the_energy = energy - CHECK_LIMIT

        """
        left_bound = 0
        right_bound = 0
        the_baseline_cps = 0
        peak_energy = energy
        """

        while energy-the_energy < CHECK_LIMIT:

            #bin values corresponding to maximum peak radii
            max_radius_left = closest_bin(the_energy-max_radius,spec.energies_kev)
            max_radius_right = closest_bin(the_energy+max_radius,spec.energies_kev)

            #sets initial values for locla minima to maximum possible (cps at bounds of max peak radius)
            least_cps_left = subtracted_spectrum_data[max_radius_left]
            least_cps_bin_left = max_radius_left
            least_cps_right = subtracted_spectrum_data[max_radius_right]
            least_cps_bin_right = max_radius_right

            #adjusts the bounds of integration (least_counts_bin_left and right) to the closest local minimum on either side of the central energy bin, if one exists within the radius
            for bin in subtracted_spectrum_data.keys():
                #checks the cps at each bin within the radius of the current energy value
                if bin >= max_radius_left and bin <= max_radius_right:
                    #if bin on the left side of the peak and has less cps than the least counts on the left, move the least counts outward
                    if bin<=the_energy and subtracted_spectrum_data[bin]<least_cps_left:
                        least_cps_left=subtracted_spectrum_data[bin]
                        least_cps_bin_left = bin
                    #if bin on the right side of the peak and has less cps than the least counts on the right, move the least counts outward
                    if bin>=the_energy and subtracted_spectrum_data[bin]<least_cps_right:
                        least_cps_right=subtracted_spectrum_data[bin]
                        least_cps_bin_right = bin

            total_peak_cps=0
            total_spec_cps=0
            total_bg_cps=0

            #baseline is set as the average of the least counts on either side of the peak
            baseline_cps = (float(nominal_value(least_cps_left)) + float(nominal_value(least_cps_right)))/2

            #integrates region. iterates over the bins within the radius, adds the baseline adjusted counts for the particular energy (delta) to the integral
            for bin in subtracted_spectrum_data.keys():
                if bin >= least_cps_bin_left and bin <= least_cps_bin_right:
                    peak_cps_per_bin = (nominal_value(subtracted_spectrum_data[bin]) - baseline_cps)
                    if peak_cps_per_bin > 0:
                        total_peak_cps += peak_cps_per_bin
                    total_spec_cps += nominal_value(spectrum_data[bin])
                    total_bg_cps += nominal_value(background_data[bin])

            total_peak_counts = total_peak_cps * livetime
            total_bg_counts = total_bg_cps * livetime
            total_spec_counts = total_spec_cps * livetime

            total_counts_unc = math.sqrt(total_spec_counts+total_bg_counts)

            """
            #checks if the counts for this energy is the strongest peak so far and updates if so
            if total_peak_counts > max_counts:
                max_counts = total_peak_counts
                max_counts_unc = total_counts_unc
                left_bound = least_cps_bin_left
                right_bound = least_cps_bin_right
                the_baseline_cps = baseline_cps
                peak_energy = the_energy
            
            #if the peak counts are above the cutoff and we haven't already counted the current peak (current_peak_left or current_peak_right must be false), add one to num_close_energies
            if total_peak_counts > MIN_PEAK_COUNTS and abs(energy-the_energy) < ENERGIES_TOO_CLOSE_CUTOFF:
                if the_energy<energy:
                    if not current_peak_left:
                        current_peak_left = True
                        close_energies.append(energy)
                else:
                    if not current_peak_right:
                        current_peak_right = True
                        close_energies.append(energy)
            else:
                if the_energy<energy:
                    current_peak_left = False
                else:
                    current_peak_right = False
            """
            #adds the peak counts to the peak_counts_per_energy dictionary
            peak_counts_per_energy[the_energy] = {"peak_energy": total_peak_counts, "total_counts_unc": total_counts_unc, "bounds": [least_cps_bin_left, least_cps_bin_right], "baseline": baseline_cps}
        
        #finds all the peaks within the check limit



        energies_counts.append(max_counts)
        energies_counts_unc.append(max_counts_unc)
        bounds.append([left_bound,right_bound])
        baselines.append(the_baseline_cps)
        peak_energies.append(peak_energy)
        close_energies_list.append(close_energies)

    return [energies_counts,energies_counts_unc,bounds,baselines,peak_energies,close_energies_list]

#functions for relative uncertainty calculations in error propegation of mass prediction formula. Change these later if more precision required
def relative_efficiency_unc(energy):
    return 0.01
def relative_decay_intensity_unc(energy):
    return 0.01
def relative_decay_constant_unc(energy):
    return 0.001
def relative_livetime_unc(energy):
    return 0.001

# returns the predicted parent isotope mass and mass uncertainty based on the daughter isotope counts and counts uncertainty
def get_mass_prediction(parent_isotope,daughter_isotope,energy,counts,unc,close_energies,livetime,eff_func):
    
    #gets the gamma yeild for the energy
    decay_intensity = 0
    for daughter_energy, decay_intensity_value in isotopes_dictionary[parent_isotope]["daughter_isotopes"][daughter_isotope]:
        if daughter_energy == energy:
            decay_intensity = decay_intensity_value
            break
        
    if decay_intensity == 0:
        return [0,0]

    #decay constant is calculated based on the half life of the parent isotope
    decay_constant=math.log(2) / isotopes_dictionary[parent_isotope]['half_life']
    molar_mass=isotopes_dictionary[parent_isotope]['molar_mass']
    eff = eff_func.get_eff(energy)

    #predicted parent mass calculated by decay physics
    predicted_parent_mass = counts * molar_mass / (eff *NA * decay_intensity * decay_constant * livetime)

    #uncertainty calculated by error propegation of mass formula, assumes error in unc, efficiency, decay intensity, decay constant, and livetime
    predicted_parent_mass_uncertainty =  predicted_parent_mass*math.sqrt((unc/counts)**2)
                                                                         #+(relative_efficiency_unc(energy))**2+(relative_decay_intensity_unc(energy))**2+(relative_decay_constant_unc(energy))**2+(relative_livetime_unc(energy))**2)
    
    #discounts the uncertainty if there are close peaks
    num_close_peaks = len(close_energies)
    predicted_parent_mass_uncertainty *= num_close_peaks+1

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

    #gets the counts at each energy
    the_counts, the_counts_unc, the_bounds, baselines, peak_energies, close_energies_list = get_counts(spec,bg,energies,spec.livetime)

    the_calibrated_counts = []
    the_calibrated_counts_unc = []

    #gets calibrated_counts based on counts by multiplying by the efficiency function
    for i in range(len(energies)):
        eff = eff_func.get_eff(energies[i])
        the_calibrated_counts.append(the_counts[i] / eff)
        the_calibrated_counts_unc.append(the_counts_unc[i] / eff)

    #creates dictionary containing isotope counts and mass info
    isotopes_in_sample_info = {}
    for i in range(len(energies)):
        title=""
        #gets the isotopes info values for each energy
        ene, counts, counts_unc, calibrated_counts, calibrated_counts_unc, bounds, baseline, peak_energy, close_energies = energies[i], the_counts[i], the_counts_unc[i], the_calibrated_counts[i], the_calibrated_counts_unc[i], the_bounds[i], baselines[i], peak_energies[i], close_energies_list[i]

        #gets the parent and daughter isotopes for the energy to construct the dictionary
        parent_isotope, daughter_isotope = inverted_isotopes_dictionary[ene][1], inverted_isotopes_dictionary[ene][0]

        #gets the predicted parent mass and unc based on the isotopes info for that energy
        predicted_parent_mass, predicted_parent_mass_unc = get_mass_prediction(parent_isotope,daughter_isotope,ene,calibrated_counts,calibrated_counts_unc,close_energies,spec.livetime,eff_func)

        #creates the dictionary by adding isotope info by energy to parent isotope
        if parent_isotope not in isotopes_in_sample_info:
            isotopes_in_sample_info[parent_isotope] = {}
        isotopes_in_sample_info[parent_isotope][ene]={"daughter_isotope":daughter_isotope, "uncalibrated_counts": counts, "uncalibrated_counts_unc": counts_unc, "calibrated_counts": calibrated_counts, "calibrated_counts_unc": calibrated_counts_unc, "predicted_parent_mass": predicted_parent_mass, "predicted_parent_mass_unc": predicted_parent_mass_unc, "close_energies": close_energies}
        title=str(ene)+"keV spectrum graph"
        subtracted_spec=spec-bg
        graph = create_peak_graph(subtracted_spec, ene, title, bounds, baseline, peak_energy, close_energies)
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
        total_unc_denominator=0
        #loops through each energy corresponding to a parent isotope, and combines the predicted parent mass values and uncertainties while maintaining error propegation
        for ene in isotopes_info[parent_isotope]:
            entry = isotopes_info[parent_isotope][ene]
            unc = entry["calibrated_counts_unc"]
            predicted_parent_mass = entry["predicted_parent_mass"]
            predicted_parent_mass_unc = entry["predicted_parent_mass_unc"]
            if unc == 0 or predicted_parent_mass_unc == 0:
                predicted_parent_mass_unc = 1e12  # arbitrarily large uncertainty to effectively ignore this value
            sum += predicted_parent_mass / (predicted_parent_mass_unc / predicted_parent_mass) ** 2
            denominator += 1 / (predicted_parent_mass_unc / predicted_parent_mass)** 2 
            total_unc_denominator += 1 / (predicted_parent_mass)**2 

        # ensures no division by zero. completes average via error propegation formula
        if len(isotopes_info[parent_isotope])!=0:
            if denominator != 0:
                sum /= denominator
                total_unc = 1/total_unc_denominator**0.5
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
    return new_isotopes_dictionary

def remove_small_energies(isotopes_dictionary):
    new_isotopes_dictionary = {}
    for parent in isotopes_dictionary:
        new_isotopes_dictionary[parent] = {"half_life": isotopes_dictionary[parent]["half_life"], "molar_mass": isotopes_dictionary[parent]["molar_mass"], "daughter_isotopes": {}}
        for daughter in isotopes_dictionary[parent]["daughter_isotopes"]:
            new_isotopes_dictionary[parent]["daughter_isotopes"][daughter] = []
            for energy_yield in isotopes_dictionary[parent]["daughter_isotopes"][daughter]:
                energy = energy_yield[0]
                if energy >= MIN_ENERGY_CUTOFF:
                      new_isotopes_dictionary[parent]["daughter_isotopes"][daughter].append(energy_yield)
    return new_isotopes_dictionary


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

def get_only_energies_graphs(energies_graphs):
    only_energies_graphs={}
    for parent in energies_graphs:
        for daughter in energies_graphs[parent]:
            for energy in energies_graphs[parent][daughter]:
                only_energies_graphs[energy]=energies_graphs[parent][daughter][energy]
    return only_energies_graphs

def get_energy_mass_predictions(isotopes_info):
    energy_mass_predictions = {}
    for parent in isotopes_info:
        energy_mass_predictions[parent]={}
        for ene in isotopes_info[parent]:
            daughter_isotope = isotopes_info[parent][ene]["daughter_isotope"]
            predicted_parent_mass = isotopes_info[parent][ene]["predicted_parent_mass"]
            predicted_parent_mass_unc = isotopes_info[parent][ene]["predicted_parent_mass_unc"]
            if not(ene in energy_mass_predictions):
                energy_mass_predictions[parent][ene] = {
                    "daughter_isotope": daughter_isotope,
                    "predicted_parent_mass": predicted_parent_mass,
                    "predicted_parent_mass_unc": predicted_parent_mass_unc
                }
    return energy_mass_predictions
""" the following functions are for creating graphs
"""

#graphs the spectrum with lines indicating energies we are checking
def create_spectrum_graph(spec, energies):
    energies = np.array(energies)
    fix, ax = plt.subplots(figsize = (10, 6))
    ax.set_yscale('log')
    ax.set_title('Background Subtracted Soil Spectrum')
    ax.set_xlim(0, np.max(spec.bin_centers_kev))
    spec.plot(ax=ax)
    indexes = [closest_bin(energy, spec.bin_centers_kev) for energy in energies]
    ax.vlines(indexes, ymin=0, ymax=np.max(spec.cps_vals) * 1.5, colors = "red", linewidth=0.5)
    filename = f"plot_{uuid.uuid4().hex}.png"

    #add these back to each graphing function when creating website
    #filepath = os.path.join("static", filename)
    filepath = filename

    plt.savefig(filepath, bbox_inches="tight", dpi=200)
    plt.close()
    return filename

def create_peak_graph(spec,energy,title,bounds,baseline,peak_energy,close_energies):
    global unorganized_energy_graphs
    plt.figure()
    fig, ax = plt.subplots(figsize = (10,3))
    ax.set_xlim(energy-40,energy+40)
    ax.set_ylim(-0.01,0.03)
    ax.set_title(title)
    spec.plot(ax=ax)
    """
    ax.vlines(bounds,ymin=0,ymax=np.max(spec.cps_vals) * 1.5, colors = "red", linewidth=0.5, linestyle='--', label="bounds of integration")
    ax.vlines([energy],ymin=0,ymax=np.max(spec.cps_vals) * 1.5, colors = "green", linewidth=0.7,label=f"expected energy: {energy}keV")
    ax.vlines([peak_energy],ymin=0,ymax=np.max(spec.cps_vals) * 1.5, colors = "blue", linewidth=0.5,label=f"peak_energy: {peak_energy}keV")
    ax.vlines([closest_bin(energy-ENERGIES_TOO_CLOSE_CUTOFF, spec.bin_centers_kev),closest_bin(energy+ENERGIES_TOO_CLOSE_CUTOFF, spec.bin_centers_kev)], ymin=0, ymax=np.max(spec.cps_vals) * 1.5, colors="orange", linewidth=0.5, linestyle='--', label=f"close energies range")
    ax.vlines([closest_bin(energy, spec.bin_centers_kev) for energy in close_energies], ymin=0, ymax=np.max(spec.cps_vals) * 1.5, colors="purple", linewidth=0.5, label=f"close energies")
    plt.legend()
    plt.axhline(y=baseline, color='y', linestyle='--', label='y=5')
    """
    filename = f"plot_{uuid.uuid4().hex}.png"

    #filepath = os.path.join("static", filename)
    filepath = filename

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

    #filepath = os.path.join("static", filename)
    filepath = filename

    plt.savefig(filepath, bbox_inches="tight", dpi=200)
    plt.close()
    return filename

def create_energy_mass_predictions_graphs(isotopes_info):
    energy_mass_predictions_graphs = {}
    for parent in isotopes_info:
        energy_mass_predictions = get_energy_mass_predictions(isotopes_info)
        labels = [str(key) for key in energy_mass_predictions[parent].keys()]
        masses = [energy_mass_predictions[parent][key]["predicted_parent_mass"] for key in energy_mass_predictions[parent].keys()]
        uncs = [energy_mass_predictions[parent][key]["predicted_parent_mass_unc"] for key in energy_mass_predictions[parent].keys()]
        plt.figure()
        plt.bar(labels,masses,yerr=uncs,capsize=5)
        plt.title("Energy-Parent Mass Predictions for "+parent)
        plt.xlabel("Energy (keV)")
        plt.ylabel("Estimated Mass (g)")
        plt.tight_layout()
        plt.xticks(rotation=90)
        filename = f"plot_{uuid.uuid4().hex}.png"

        #filepath = os.path.join("static", filename)
        filepath=filename

        plt.savefig(filepath, bbox_inches="tight", dpi=200)
        plt.close()
        energy_mass_predictions_graphs[parent] = filename
    return energy_mass_predictions_graphs


def create_key_graphs(results):
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

    #filepath = os.path.join("static", filename)
    filepath = filename

    plt.savefig(filepath, bbox_inches="tight", dpi=200)
    plt.close()
    return filename

""" analyze_the_spectrum gets the isotope info and creates the graphs, then returns it into app.py to be rendered in the results page
"""
def analyze_spectrum(spectrum_path,background_path,efficiency,isotopes_dictionary=isotopes_dictionary):

    #uncomment the following line to remove energies that are close to each other
    #isotopes_dictionary = remove_close_energies(isotopes_dictionary)

    #small energies can cause issues with peak identification
    isotopes_dictionary = remove_small_energies(isotopes_dictionary)

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
    returnstatement["spectrum_graph"] = create_spectrum_graph(spec, list(get_inverted_isotopes_dictionary(isotopes_dictionary).keys()))

    #results is used to list the masses and uncertainties of the parent isotopes
    returnstatement["results"]=[]
    for parent_isotope in masses:
        mass,unc = masses[parent_isotope]
        returnstatement["results"].append([parent_isotope,mass,unc])

    #results graph displays the mass predictions in a bar graph
    returnstatement["results_graph"] = create_results_graph(returnstatement["results"])

    #daughters graphs are the graphs corresponding to each parent isotope which showing the mass predictions per daughter isotope
    returnstatement["energy_mass_predictions_graphs"] = create_energy_mass_predictions_graphs(isotopes_info)
    returnstatement["energy_mass_predictions"] = get_energy_mass_predictions(isotopes_info)

    #daughters_energies gives the energies for each daughter isotope, used to create buttons for eneriges that pop up when daughter energy button is clicked
    returnstatement["daughters_energies"] = get_daughters_energies()

    #energy graphs are the graphs of individual energies, zoomed in on the spectrum. only_energies_graphs contains that info in a different data structure (energy->graph) for ease of access
    returnstatement["energy_graphs"] = energy_graphs
    returnstatement["only_energies_graphs"] = get_only_energies_graphs(energy_graphs)

    #key graphs contains important graphs like U-235 to U-238 ratio which are displayed seperately
    returnstatement["key_graphs"] = create_key_graphs(returnstatement["results"])
    returnstatement["conclusions"] = "no conclusions yet"

    #the following keys of returnstatement contain info for the debug panel
    returnstatement["isotopes_info"]=isotopes_info
    returnstatement["isotopes_dictionary"]=isotopes_dictionary

    return returnstatement

