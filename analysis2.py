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

NA = 6.022e23

RADIUS = 2.5
EFFICIENCY="2019_NAA_eff_calibration_parameters.json"
INTEGRATION_LENGTH=2
MAX_UNCERTAINTY_PROPORTION = 0.1 #max uncertainty/counts
MAX_DAUGHTERS_TO_FLAG_PARENT = 3 #max number of daughter isotopes to flag a parent isotope
PEAK_STANDARD_DEVIATION_UNCERTAINTY_CUTOFF = 1000
UNCALIBRATED_COUNTS_CUTOFF = 1000
MIN_BOUNDS_MULTIPLIER = 1.5
#change based on files used:
#DETECTOR_EFFICIENCY = "2019_NAA_eff_calibration_parameters.json"
DETECTOR_EFFICIENCY = "2019_NAA_eff_calibration_parameters.json"



"""
sample_files = {
    '31760': {
        'spec': '31760_Soil_Sample.Spe',
        'bg': 'Background_4_25_2025.Spe',
        'efficiency': "2019_NAA_eff_calibration_parameters.json",
        'mass': 1424.8
    },
    '67649': {
        'spec': '67649_soil_sample.Spe',
        'bg': 'rayleigh_background_6_12_25.Spe',
        'efficiency': "2019_NAA_eff_calibration_parameters.json",
        'mass': 1089.7
    },
    '74433': {
        'spec': 'sample74433_10_17-20_2025.Spe',
        'bg': 'rayleigh_background_6_12_25.Spe',
        'efficiency': "2019_NAA_eff_calibration_parameters.json",
        'mass': 298.8
    },
    '15598': {
        'spec': '15598_soil_sample.Spe',
        'bg': 'rayleigh_background_6_12_25.Spe',
        'efficiency': "2019_NAA_eff_calibration_parameters.json",
        'mass': 0
    },
    'berkeley': {
        'spec': 'berkeley-control-sample_10_20-22_2025.Spe.',
        'bg': 'background_10_06_2025.Spe',
        'efficiency': "2019_NAA_eff_calibration_parameters.json",
        'mass': 546.1
    },
    '77809': {
        'spec': 'sample77809_10_22-24_2025.Spe',
        'bg': 'background_10_06_2025.Spe',
        'efficiency': "2019_NAA_eff_calibration_parameters.json",
        'mass': 334.4
    }
"""


    # ['31760_Soil_Sample.Spe','Background_4_25_2025.Spe'],['67649_soil_sample.Spe','rayleigh_background_6_12_25.Spe'],["15598_soil_sample.Spe",'rayleigh_background_6_12_25.Spe'],["sample74433_10_17-20_2025.Spe",'rayleigh_background_6_12_25.Spe']]

#dictionary containing information on isotopes to be analyzed
#note: gamma yield is the probability of a gamma emmision per decay event
#note: the peaks of daughter isotopes were only included if their gamma yields were greater than 10%. (reducing the chance of small random fluctuations being inaccurately included)
"""
isotopes_dictionary = {
    "U-238": {
        "half_life": 4.468e9 * 365.25 * 24 * 3600,
        "molar_mass": 238.05078826,  # g/mol
        "daughter_isotopes": {
            #format... daughter isotope: [[energy1,gamma yeild1],[energy2,gamma yeild2]...]
            "Pa-234": [[131.3,0.189],[946.0,0.14],[883.24,0.10]],
            "Pb-214": [[351.93,0.3572],[295.224,0.1847]],
            "Bi-214": [[609.3,0.4544],[1745.491,0.1529],[1120,290.1490]],
            "Tl-210": [[799.6,0.9896],[296,0.79],[1316,0.21],[1210,0.17],[1070,0.12]],
            "Hg-206": [[304.896,0.26]],
            }
    },
    "U-235": {
        "half_life": 703.8e6 * 365.25 * 24 * 3600,
        "molar_mass": 235.0439299,
        "daughter_isotopes": {
            "U-235": [[185.713,0.572],[143.765,0.1093]],
            "Th-231": [[25.65,0.137]],
            "Pa-231": [[27.36,0.105]],
            "Th-227": [[235.96,0.129]],
            "Fr-223": [[50.094,0.34]],
            "Ra-223": [[269.463,0.133]],
            "Rn-219": [[271.23,0.108]],
            "Bi-215": [[293.5,0.49]],
            "Bi-211": [[351.07,0.1302]],
            }
    },
    "Th-232": {
        "half_life": 1.405e10 * 365.25 * 24 * 3600,
        "molar_mass": 232.0380553,
        "daughter_isotopes": {
            "Ac-228": [[911.204,0.258],[968.971,0.158],[338.32,0.1127],[964.766,0.0499],[463.004,0.0440],[794.947,0.0425]],
            "Ra-224": [[240.986,0.041]],
            "Pb-212": [[238.632,0.436]],
            "Tl-208": [[2614.511,0.99754],[583.187,0.85],[510.77,0.226],[860.557,0.125],[277.371,0.066]]
        }
    },
    "Pu-239": {
        "half_life": 24110 * 365.25 * 24 * 3600,
        "molar_mass": 239.0521634,
        "daughter_isotopes": {
            "U-235": [[185.713,0.572],[143.765,0.1093]],
            "Th-231": [[25.65,0.137]],
            "Pa-231": [[27.36,0.105]],
            "Th-227": [[235.96,0.129]],
            "Fr-223": [[50.094,0.34]],
            "Ra-223": [[269.463,0.133]],
            "Rn-219": [[271.23,0.108]],
            "Bi-215": [[293.5,0.49]],
            "Bi-211": [[351.07,0.1302]],
        }
    }
}
"""
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
def graph_peak(spec,energy,title,RADIUS):
    fig, ax = plt.subplots(figsize = (10,3))
    ax.set_xlim(energy-20*RADIUS,energy+20*RADIUS)
    ax.set_ylim(-0.01,0.03)
    ax.set_title(title)
    spec.plot(ax=ax)
    max_bounds = [closest_bin(energy-RADIUS, spec.bin_centers_kev),closest_bin(energy+RADIUS, spec.bin_centers_kev)]
    ax.vlines(max_bounds,ymin=0,ymax=np.max(spec.cps_vals) * 1.5, colors = "red", linewidth=0.5)
    min_bounds = [closest_bin(energy-MIN_BOUNDS_MULTIPLIER*RADIUS, spec.bin_centers_kev),closest_bin(energy+MIN_BOUNDS_MULTIPLIER*RADIUS, spec.bin_centers_kev)]
    ax.vlines(min_bounds,ymin=0,ymax=np.max(spec.cps_vals) * 1.5, colors = "green", linewidth=0.5)

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

def get_counts(spec,energies,RADIUS,INTEGRATION_LENGTH,livetime):
    data = dict(zip(spec.energies_kev,spec.cps))
    counts=[]
    counts_unc=[]
    for energy in energies:
        integral=0
        radius_left = closest_bin(energy-RADIUS,spec.energies_kev)
        radius_right = closest_bin(energy+RADIUS,spec.energies_kev)
        check_left = closest_bin(energy-MIN_BOUNDS_MULTIPLIER*RADIUS,spec.energies_kev)
        check_right = closest_bin(energy+MIN_BOUNDS_MULTIPLIER*RADIUS,spec.energies_kev)
        greatest_counts=0
        greatest_counts_bin=0
        least_counts_left=data[check_left]
        least_counts_bin_left=check_left
        least_counts_right=data[check_right]
        least_counts_bin_right=check_right
        for bin in data.keys():
            if bin >= check_left and bin <= check_right:
                if data[bin]>greatest_counts:
                    greatest_counts=data[bin]
                    greatest_counts_bin = bin
        for bin in data.keys():
            if bin >= check_left and bin <= check_right:
                if bin<greatest_counts_bin and data[bin]<least_counts_left:
                    least_counts_left=data[bin]
                    least_counts_bin_left = bin
                if bin>greatest_counts_bin and data[bin]<least_counts_right:
                    least_counts_right=data[bin]
                    least_counts_bin_right = bin
        baseline_adjusted_counts=""
        baseline=None
        unadjusted_integral=0
        left_bound,right_bound=None,None
        if greatest_counts_bin >= radius_left and greatest_counts_bin <= radius_right:
            baseline=max(nominal_value(least_counts_left),nominal_value(least_counts_right))
            left_bound = least_counts_bin_left
            right_bound = least_counts_bin_right
            for bin in data.keys():
                if bin >= left_bound and bin <= right_bound:
                    baseline_adjusted_counts=(nominal_value(data[bin])-baseline)*livetime
                    unadjusted_integral+=nominal_value(data[bin])*livetime
                    if baseline_adjusted_counts > 0:
                        integral += baseline_adjusted_counts
        counts.append(integral)
        counts_unc.append(math.sqrt(abs(unadjusted_integral)))
        #graph_bounds(spec,energy,str(energy)+" counts: "+str(integral),closest_bin(greatest_counts_bin,spec.energies_kev),left_bound,right_bound,baseline)

    return [counts,counts_unc]

def get_counts_method_two(spec,energies,RADIUS,INTEGRATION_LENGTH,livetime):
    counts=[]
    counts_unc=[]
    spec_energies_kev = spec.energies_kev.tolist()
    spec_cps = spec.cps.tolist()
    for energy in energies:
        integral=0
        radius_left = spec_energies_kev.index(closest_bin(energy-RADIUS,spec.energies_kev))
        radius_right = spec_energies_kev.index(closest_bin(energy+RADIUS,spec.energies_kev))

        greatest_counts=0
        greatest_counts_index=0

        for index in range(radius_left,radius_right+1):
            if spec_cps[index]>greatest_counts:
                greatest_counts=spec_cps[index]
                greatest_counts_index = index

        left_bound_index=greatest_counts_index
        previous_bin_counts=greatest_counts

        while True:
            left_bound_index-=1
            if spec_cps[left_bound_index]>previous_bin_counts:
                break
        right_bound_index=greatest_counts_index
        previous_bin_counts=greatest_counts
        while True:
            right_bound_index+=1
            if spec_cps[right_bound_index]>previous_bin_counts:
                break

        baseline=max(nominal_value(spec_cps[left_bound_index]),nominal_value(spec_cps[right_bound_index]))
        baseline_adjusted_counts=""
        unadjusted_integral=0

        for i in range(left_bound_index,right_bound_index+1):
            baseline_adjusted_counts=(nominal_value(spec_cps[i])-baseline)*livetime
            unadjusted_integral+=nominal_value(spec_cps[i])*livetime
            if baseline_adjusted_counts > 0:
                        integral += baseline_adjusted_counts



        """
        if greatest_counts_index >= radius_left and greatest_counts_index <= radius_right:
            for bin in data.keys():
                if bin >= left_bound_index and bin <= right_bound_index:
                    baseline_adjusted_counts=(nominal_value(data[bin])-baseline)*livetime
                    unadjusted_integral+=nominal_value(data[bin])*livetime
                    if baseline_adjusted_counts > 0:
                        integral += baseline_adjusted_counts

        """
        counts.append(integral)
        counts_unc.append(1)
        #graph_bounds(spec,energy,str(energy)+" counts: "+str(integral),closest_bin(greatest_counts_index,spec.energies_kev),left_bound_index,right_bound_index,baseline)

    return [counts,counts_unc]

# returns the predicted parent isotope mass and mass uncertainty based on the daughter isotope counts and counts uncertainty
def get_mass_prediction(parent_isotope,daughter_isotope,energy,counts,unc,livetime):
    gamma_yield = 0
    for daughter_energy, gamma_yield_value in isotopes_dictionary[parent_isotope]["daughter_isotopes"][daughter_isotope]:
        if daughter_energy == energy:
            gamma_yield = gamma_yield_value
            break
    decay_constant=math.log(2) / isotopes_dictionary[parent_isotope]['half_life']
    molar_mass=isotopes_dictionary[parent_isotope]['molar_mass']
    #assuming secular equilibrium (parent isotope counts = daughter isotope counts)
    predicted_parent_mass = counts * molar_mass / (NA * gamma_yield * decay_constant * livetime)
    predicted_parent_mass_uncertainty = unc * molar_mass / (NA * gamma_yield * decay_constant * livetime)
    return [predicted_parent_mass,predicted_parent_mass_uncertainty]

#returns a dictionary with format... parent isotope 1: {daugther isotope energy 1: [daughter isotope, counts, unc, predicted parent mass, predicted parent mass unc]}... for each parent and daughter isotope energy believed to be in the sample
def get_isotopes_info(spec, bg, isotopes_dictionary,efficiency):
    eff_func = am.Efficiency()
    eff_func.set_parameters(efficiency)
    inverted_isotopes_dictionary = get_inverted_isotopes_dictionary(isotopes_dictionary)
    energies = list(inverted_isotopes_dictionary.keys())
    subtracted_spec=spec-bg
    counts, counts_unc = get_counts(subtracted_spec,energies,RADIUS,INTEGRATION_LENGTH,spec.livetime)
    subtracted_spec = spec-bg
    #gets the counts and count uncertainties
    #calibrates counts and counts_unc to detector efficiency
    uncalibrated_counts = counts
    counts = [counts[i]/eff_func.get_eff(energies[i]) for i in range(len(energies))]
    counts_unc = [counts_unc[i]/eff_func.get_eff(energies[i]) for i in range(len(energies))]
    #creates dictionary containing isotope counts and mass info
    isotopes_in_sample_info = {}
    pf = PF(energies, spec, bg)
    pf_uncalibrated_counts, pf_counts_unc = pf.get_counts()
    pf_counts = [pf_uncalibrated_counts[i]/eff_func.get_eff(energies[i]) for i in range(len(energies))]
    for i in range(len(energies)):
        title=""
        ene, unc, daughter_counts, uncalibrated_daughter_counts = energies[i], counts_unc[i], counts[i], uncalibrated_counts[i]
        pf_uncalibrated_daughter_counts, pf_daughter_counts, pf_daughter_counts_unc = pf_uncalibrated_counts[i], pf_counts[i], pf_counts_unc[i]
        parent_isotope, daughter_isotope = inverted_isotopes_dictionary[ene][1], inverted_isotopes_dictionary[ene][0]
        predicted_parent_mass, predicted_parent_mass_unc = get_mass_prediction(parent_isotope,daughter_isotope,ene,daughter_counts,unc,spec.livetime)
        pf_predicted_parent_mass, pf_predicted_parent_mass_unc = get_mass_prediction(parent_isotope,daughter_isotope,ene,pf_daughter_counts,pf_daughter_counts_unc,spec.livetime)
        if parent_isotope not in isotopes_in_sample_info:
            isotopes_in_sample_info[parent_isotope] = {}
        #if daughter_counts!=0 and uncalibrated_daughter_counts>=UNCALIBRATED_COUNTS_CUTOFF:
        isotopes_in_sample_info[parent_isotope][ene]={"daughter_isotope":daughter_isotope, "counts": daughter_counts, "counts unc": unc, "predicted parent mass": predicted_parent_mass, "predicted parent mass unc": predicted_parent_mass_unc}
        #title+=f"ene: {ene}keV, parent: {parent_isotope}, daughter: {daughter_isotope}, uncalibrated counts: {uncalibrated_daughter_counts} counts: {daughter_counts:.2e}, uncertainty: {unc}, unc/ud_counts: {unc/uncalibrated_daughter_counts}, predicted parent mass: {predicted_parent_mass:.2e}g"
        #graph_peak(subtracted_spec,ene,title,RADIUS)
        #print(f"Results for {ene}keV peak using PF: uncalibrated counts: {pf_uncalibrated_daughter_counts}, counts: {pf_daughter_counts}, counts_unc: {pf_daughter_counts_unc}, predicted_parent_mass: {pf_predicted_parent_mass}g, predicted_parent_mass_unc: {pf_predicted_parent_mass_unc}g")
        #pf.plot_roi(ene)
    return isotopes_in_sample_info
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
            if unc!=0 and predicted_parent_mass_unc!=0:
                sum += predicted_parent_mass/predicted_parent_mass_unc**2
                denominator += 1/predicted_parent_mass_unc**2
        if len(isotopes_info[parent_isotope])!=0:
            sum /= denominator
            total_unc = 1/denominator**0.5

        estimated_parent_masses[parent_isotope] = [sum,total_unc]
    return [estimated_parent_masses,flagged_isotopes]
    """
    masses_and_uncertainties = {}
    for parent_isotope in isotopes_info:
        for ene in isotopes_info[parent_isotope]:

        gamma_yield=energy_expectations[parent_isotope]["daughter_isotopes"][daughter_isotope]['gamma_yield']
        decay_constant=energy_expectations[parent_isotope]['decay_constant']
        molar_mass=energy_expectations[parent_isotope]['molar_mass']
        if parent_isotope not in masses_and_uncertainties:
            masses_and_uncertainties[parent_isotope] = []
        masses_and_uncertainties[parent_isotope].append([counts * molar_mass / (86400 * NA * gamma_yield * decay_constant), unc * molar_mass / (86400 * NA * gamma_yield * decay_constant)])
    for parent_isotope in masses_and_uncertainties:
        weighted_avg = 0
        denominator = 0
        for mass, unc in masses_and_uncertainties[parent_isotope]:
            weighted_avg += mass/unc**2
            denominator += 1/unc**2
        weighted_avg /= denominator
        total_uncertainty = 1/denominator**0.5
        masses_and_uncertainties[parent_isotope] = [weighted_avg, total_uncertainty]
    return masses_and_uncertainties
    """

# This is the function for the main program execution. calls the above functions to get the isotopes info and convert it into masses. Prints out the isotope masses.
def analyze_spectrum(filename,spec,bg,efficiency,sample_mass):
    #spec.calibrate_like(bg)
    bg = bg.rebin(spec.bin_edges_kev)
    isotopes_info = get_isotopes_info(spec,bg,isotopes_dictionary,efficiency)
    masses,flagged_isotopes = estimate_aggregated_masses_and_uncertainties(isotopes_info, isotopes_dictionary)
    printstatement=""
    printstatement+="\n\nIsotopes in sample "+filename
    printstatement+=" ("+str(sample_mass)+"g): \n"

    for parent_isotope in masses:

      mass,unc = masses[parent_isotope]
      printstatement+=f"\n{parent_isotope}: mass: {mass:.2e}g, unc: {unc:.2e}"
    printstatement+=f"\nflagged isotopes: {flagged_isotopes}"
    return printstatement

sample_data = pd.read_csv('sample_data.csv')

bigprintstatement=''
"""
for i in range(len(sample_data)):
    #for i in range(1):
    spectrum_file=sample_data["Spectrum file"].iloc[i]
    if not os.path.exists(spectrum_file):
        print(spectrum_file)
        continue
    spec = Spectrum.from_file(spectrum_file)
    bg = Spectrum.from_file(sample_data['Background spectrum'].iloc[i])
    bigprintstatement += analyze_spectrum(sample_data["Sample ID"].iloc[i],spec,bg,EFFICIENCY,sample_data["Sample Weight"].iloc[i])
#bigprintstatement+= analyze_spectrum("67649",Spectrum.from_file("67649_soil_sample.Spe"),Spectrum.from_file("rayleigh_background_6_12_25.Spe"),EFFICIENCY,"unknown")
#bigprintstatement+= analyze_spectrum("31760",Spectrum.from_file("31760_Soil_Sample.Spe"),Spectrum.from_file("Background_4_25_2025.Spe"),EFFICIENCY,"unknown")
#bigprintstatement+= analyze_spectrum("15598",Spectrum.from_file("15598_soil_sample.Spe"),Spectrum.from_file("rayleigh_background_6_12_25.Spe"),EFFICIENCY,"unknown")
"""

print("\nRESULTS")
print(bigprintstatement)
