# Austin's PF script
import becquerel as bq
from becquerel import Spectrum
import numpy as np
import scipy.integrate as integrate
import math as m
import matplotlib.pyplot as plt


# INPUT: source_energies, spectrum, background, branching_ratio


def f_near(energy_array, energy):
    """
    Finds index of closest energy in spectrum to that of the characteristic energy

    # Arguments: 
    energy_array - list of energies in a spectrum \n 
    energy - the energy to find the index of
    """
    idx = np.abs(energy_array - energy).argmin()
    return idx


class PF(object):
    """
    Peak Fitter object
    """
    def __init__(self, source_energies, spectrum, background=None,
                 source_activities=None, source_isotopes=None, branching_ratio=None):
        """
        Initialization function of the Peak Fitter (PF) object

        # Arguments:
        source_energies - the energies of the peaks \n
        spectrum - becquerel spectra object \n
        background - becquerel spectra object of the background \n
        source_activities - the activities of the isotopes in the source \n
        source_isotopes - the isotopes in the source \n
        branching_ratio - the branching ratios of the gamma ray energies of the source
        """
        self.integrals_unc = None
        self.spectrum = spectrum
        self.background = background
        self.source_energies = source_energies

        self.source_activities = source_activities
        self.source_isotopes = source_isotopes
        self.branching_ratio = branching_ratio
        self.fitters = []
        self.sub_spec = None

        self.integrals = []

    def calibration(self):
        """
        Returns efficiencies using the values of counts from becquerel's fit to a gaussian
        """
        efficiencies = []
        for x in range(0, len(self.source_activities)):
            efficiency = self.integrals[x] / (self.source_activities[x] * self.branching_ratio[x])
            efficiencies = np.append(efficiencies, efficiency)
        return efficiencies

    def get_counts(self):
        """
        Returns the counts for each energy in source_energies as well as the uncertainties. Integrates
        over a Gaussian fit to find the counts and uncertainties
        """
        if self.background is not None:
            self.sub_spec = self.spectrum - self.background  # background subtraction
        else:
            self.sub_spec = self.spectrum  # background is None

        spec_energies = self.sub_spec.bin_centers_kev  # all energies
        integrals = []
        integrals_unc = []
        model = ['gauss', 'line', 'erf'] 
        for i, n in enumerate(self.source_energies):
            offset = (np.sqrt(n/50)) + 10
            self.fitters.append(
                bq.core.fitting.Fitter(model, x=self.sub_spec.bin_indices, y=self.sub_spec.cps_vals, y_unc=self.sub_spec.cps_uncs))
            idx = f_near(spec_energies, n)
            
            self.fitters[i].set_roi(idx - offset, idx + offset)
            self.fitters[i].fit()
            amp = self.fitters[i].result.params['gauss_amp']
            mu = self.fitters[i].result.params['gauss_mu']
            sigma = self.fitters[i].result.params['gauss_sigma']

            def gaussian(x):
                return (self.spectrum.livetime * amp.value / (m.sqrt(2 * m.pi) * sigma.value)) * m.exp(
                    - ((x - mu.value) ** 2) / (2 * sigma.value ** 2))
            def gaussian_up(x):
                return (self.spectrum.livetime * amp_up / (m.sqrt(2 * m.pi) * sigma_up)) * m.exp(
                    - ((x - mu.value) ** 2) / (2 * sigma_up ** 2))
            def gaussian_low(x):
                return (self.spectrum.livetime * amp_low / (m.sqrt(2 * m.pi) * sigma_low)) * m.exp(
                    - ((x - mu.value) ** 2) / (2 * sigma_low ** 2))
            
            integral = integrate.quad(gaussian, idx - offset, idx + offset)
            integrals = np.append(integrals, integral[0])
            # calculate amp_up by amp_up = amp + amp_unc
            if amp.stderr is not None:
                amp_up = amp.value + amp.stderr
                amp_low = amp.value - amp.stderr
            else:
                amp_up = 2.0 * amp.value
                amp_low = 0

            if sigma.stderr is not None:
                sigma_up = sigma.value + sigma.stderr
                sigma_low = sigma.value - sigma.stderr
            else:
                sigma_up = 2.0 * sigma.value
                sigma_low = 0

            integral_up = integrate.quad(gaussian_up, idx - offset, idx + offset)

            if amp_low == 0:
                integral_low = [0]
            else:
                integral_low = integrate.quad(gaussian_low, idx - offset, idx + offset)
            # calculate integral_unc
            integral_unc = (integral_up[0] - integral_low[0]) / 2
            integrals_unc = np.append(integrals_unc, integral_unc)
        self.integrals = integrals
        self.integrals_unc = integrals_unc
        return integrals, integrals_unc

    def plot_roi(self, energy_kev):
        """
        Plots the fitted peak for a given energy.
        'get_counts' must be run first.

        # Arguments:
        energy_kev - The energy of the peak to plot.
        """
        if not self.fitters or self.sub_spec is None:
            print("Please run get_counts() before plotting.")
            return

        try:
            if isinstance(self.source_energies, np.ndarray):
                energies_list = self.source_energies.tolist()
            else:
                energies_list = self.source_energies
            idx = energies_list.index(energy_kev)
        except ValueError:
            print(f"Energy {energy_kev} keV not found in source energies.")
            return

        fitter = self.fitters[idx]
        roi_x = fitter.x_roi
        plt.figure(figsize=(10, 6))
        plt.plot(roi_x, self.sub_spec.cps_vals[roi_x], 'k.', label='Spectrum Data')
        fitter.plot(color='r', label='Fit')
        plt.title(f'Fit for {energy_kev:.2f} keV peak')
        plt.xlabel('Index')
        plt.ylabel('Counts per Second')
        plt.legend()
        plt.show()

    def Efficiency(self):
        """
        Performs a Gaussian fit and returns the efficiencies using the integrals
        """
        spec = Spectrum.from_file(self.spectrum)  # import spectrum data
        bg = Spectrum.from_file(self.background)  # import spectrum data
        sub_spec = spec - bg  # background subtraction
        spec_energies = sub_spec.energies_kev  # all energies
        integrals = []
        model = ['gauss', 'line', 'erf']
        for n in self.source_energies:
            offset = (np.sqrt(n/50)) + 10
            fit = bq.core.fitting.Fitter(model, x=sub_spec.bin_indices, y=sub_spec.cps_vals, y_unc=sub_spec.cps_uncs)
            idx = f_near(spec_energies, n)
            fit.set_roi(idx - offset, idx + offset)
            fit.fit()
            amp = fit.result.params['gauss_amp'].value
            mu = fit.result.params['gauss_mu'].value
            sigma = fit.result.params['gauss_sigma'].value

            def gaussian(x):
                return (spec.livetime * amp / (m.sqrt(2 * m.pi) * sigma)) * m.exp(- ((x - mu) ** 2) / (2 * sigma ** 2))

            integral = integrate.quad(gaussian, idx - offset, idx + offset)
            integrals = np.append(integrals, integral[0])
        efficiencies = self.calibration()
        return efficiencies