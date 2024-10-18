import numpy as np
import yaml
from scipy.optimize import brentq
import matplotlib.pyplot as plt


class Generator:
    def __init__(
        self, struc_file="structure.yml", set_file="sets.yml", cap_file="cap.yaml"
    ):
        # Initialize constants and parameters based on physical constants and experimental settings
        self.Ns = 1e15  # Surface site density [sites/m^2]
        self.Eo = 8.8541878128e-12  # Vacuum permittivity [F/m]
        self.q = 1.60218e-19  # Elementary charge [C]
        self.Kb = 1.38064852e-23  # Boltzmann constant [J/K]
        self.Navo = 6.023e23  # Avogadro's number [1/mol]
        self.T = 298  # Temperature [K] (Room temperature)
        self.B = (self.Kb * self.T) / self.q  # Thermal voltage [V]
        self.conc = 0.001  # Bulk electrolyte concentration [mol/L]
        self.Io = self.conc / 0.001  # Ionic strength [mol/m^3], assuming unit volume
        self.Ew = 78.3810 * self.Eo  # Permittivity of water/electrolyte [F/m]
        # Debye length parameter for the electrolyte solution
        self.Qo = np.sqrt(8 * self.Ew * self.B * self.Io * self.Navo * self.q)  # [C/m^2]
        self.c_s = 0.8  # Stern layer capacitance [F/m^2], can be adjusted for oxide layers
        self.n = 0  # Number of positive (amine) chargeable groups
        self.m = 0  # Number of negative (carboxyl, phenol) chargeable groups
        self.chain = None  # Amino acid sequence (peptide chain)
# Load data from YAML files containing amino acid properties and sets
        with open(struc_file, "r") as outfile:
            self.db_dict = yaml.safe_load(outfile)  # Dictionary of amino acid dissociation constants
        with open(set_file, "r") as outfile:
            self.sets = yaml.safe_load(outfile)  # Lists of amino acids categorized by their side chains
        with open(cap_file, "r") as outfile:
            self.cap = yaml.safe_load(outfile)  # Capacitance parameters for amino acids

    def get_aas(self):
        return [*self.sets[0], *self.sets[1], *self.sets[2]]

    def get_kvals(self, chain):
        aa = None
        if chain is None:
            raise ValueError("Chain sequence is required")
        if type(chain) == str:
            chain = [*chain]
        if type(chain) != list:
            raise ValueError("Chain must be string or list")
        aa = chain
        self.chain = chain
        k_vals = None
        # ind determines whether the N-terminal (amine) or C-terminal (carboxyl) is immobilized
        # For this model, we assume the chain is immobilized by the carboxyl end (C-terminal)

        ind = 0
        # N  M
        # DYK -  2, 2
        # DYKA - 2 ,2
        # DYND - 1, 3

        # Initialize counts of chargeable groups based on the immobilization
        self.n = 1 - ind  # Number of amine groups (positive charges)
        self.m = ind      # Number of carboxyl groups (negative charges)
        connection = (ind) % 2  # Determines which terminal is available for charging

        # Iterate over each amino acid in the chain to collect dissociation constants
        for _, k in enumerate(aa):
            if k in self.sets[1]:
                # Amino acid has a carboxylic side chain (negative chargeable group)
                self.m += 1
            elif k in self.sets[2]:
                # Amino acid has an amine side chain (positive chargeable group)
                self.n += 1
            if k_vals is None:
                # For the first amino acid, initialize k_vals
                k_vals = self.db_dict[k]
                # Remove the dissociation constant corresponding to the immobilized terminal
                k_vals = np.delete(k_vals, [connection], axis=0)
            else:
                # For subsequent amino acids, concatenate their dissociation constants
                # Exclude the immobilized terminal - can we do this???
                k_vals = np.concatenate(
                    (
                        np.delete(k_vals, [len(k_vals) - 2], axis=0),
                        np.delete(self.db_dict[k], [connection], axis=0),
                    )
                )

        if k_vals is None:
            return
        k_vals.sort()  # Sort dissociation constants in ascending order
        k_vals = k_vals[k_vals != 0]  # Remove any zero values
        # Convert pKa/pKb values to actual Ka/Kb values
        k_vals = np.power(10, -k_vals)
        k_vals = np.insert(k_vals, 0, 1)  # Insert a leading 1 for calculation purposes
        return k_vals

        # print(k_vals)
        k_vals = np.power(10, -k_vals)
        k_vals = np.insert(k_vals, 0, 1)
        return k_vals

    def get_tokens(self, k_vals):
        """
        Calculates the surface potential and capacitance curves for the given dissociation constants.

        Parameters:
        k_vals (numpy array): Array of dissociation constants

        Returns:
        pot_curve (numpy array): Second derivative of surface potential with respect to pH
        cap_curve (numpy array): Second derivative of capacitance with respect to pH
        """
        
        pot_plot = []
        pot = 0.0
        pot_plot = np.zeros(1400)
        cnt = 0

        # Define the function whose root we want to find (for surface potential calculation)
        def f(x):
            # Calculate the diffuse layer surface charge density using the Gouy-Chapman model
            sig_dl = self.Qo * np.sinh(x / (2 * self.B))
            # Calculate the surface potential including the Stern layer capacitance
            pot_o = sig_dl / self.c_s + x
            # Calculate the proton concentration at the surface using the Boltzmann distribution
            H_s = H_bulk * np.exp(-pot_o / self.B)
            # Numerator for positive charges (amine groups)
            num = 0
            for i in range(self.n):
                num += np.power(H_s, self.n + self.m - i) * np.prod(k_vals[0 : i + 1])
            num *= self.n
            # Numerator for negative charges (carboxyl, phenol groups)
            num1 = 0
            for i in range(self.m):
                num1 += np.power(H_s, i) * np.prod(k_vals[0 : self.n + self.m + 1 - i])
            num1 *= self.m
            # Denominator for total charge calculation
            den = 0
            for i in range(self.n + self.m + 1):
                den += np.power(H_s, self.n + self.m - i) * np.prod(k_vals[0 : i + 1])
            # Calculate the site-binding surface charge density
            sig_o = self.q * self.Ns * ((num - num1) / den)
            return sig_dl - sig_o  # The root of this function gives the surface potential

        # Function to calculate the total capacitance at a given surface potential
        def get_cap(zetaPot):
            curr_cap = 0
            curr_len = 0
            if not isinstance(self.chain, list):
                raise TypeError("chain should be list")
            for j in range(len(self.chain)):
                # Retrieve the permittivity and length for each amino acid
                mol_per, mol_length = self.cap[self.chain[j]]
                mol_per *= self.Eo  # Convert relative permittivity to absolute permittivity
                mol_length *= 1e-10  # Convert length from Angstroms to meters
                C = mol_per / mol_length  # Intrinsic capacitance of the amino acid
                if j == 0:
                    curr_cap = C  # Initialize the cumulative capacitance
                    curr_len = mol_length  # Initialize the cumulative length
                else:
                    # Combine capacitances in series (since they are stacked)
                    curr_cap = (curr_cap * C) / (curr_cap + C)
                    curr_len = curr_len + mol_length  # Sum the lengths

            # Calculate the diffuse layer capacitance using the Gouy-Chapman model
            diffuseLayerCap = self.Qo / (2 * self.B) * np.cosh(zetaPot / (2 * self.B))
            # Combine diffuse layer and Stern layer capacitances in series
            diffLayerCap = (diffuseLayerCap * self.c_s) / (diffuseLayerCap + self.c_s)
            # Adjust capacitance based on surface site density (convert to F/μm^2)
            capBio2 = curr_cap * self.Ns * 1e-16  # Convert to [F/μm^2]
            return diffLayerCap + capBio2  # Total capacitance is the sum of contributions

            diffuseLayerCap = self.Qo / (2 * self.B) * np.cosh(zetaPot / (2 * self.B))

            diffLayerCap = (diffuseLayerCap * self.c_s) / (diffuseLayerCap + self.c_s)

            capBio2 = curr_cap * self.Ns * 1e-16
            # return curr_cap*curr_len
            return diffLayerCap + capBio2

        cap_plot = []
        ph_array = np.linspace(0, 14, 1400)
        for pH in ph_array:
            H_bulk = 10 ** (-pH)

            # Minimum and maximum of the entire curve

            pot = brentq(f, -0.28, 0.28)  # this is zeta pot for capacitance calc
            if not isinstance(pot, float):
                raise TypeError("pot not float")
            cap_plot.append(get_cap(pot))
            sig_dl = self.Qo * np.sinh(pot / (2 * self.B))
            pot_o = sig_dl / self.c_s + pot  # surface potential for capacitance
            pot_plot[cnt] = pot_o
            cnt += 1

        pot_curve = np.gradient(
            np.gradient(pot_plot, ph_array[1] - ph_array[0]), ph_array[1] - ph_array[0]
        )
        plt.plot(ph_array, pot_curve)
        # tokens_pot = np.where(
        #     np.logical_and(np.abs(np.diff(out)) >= 1e-10, np.diff(np.sign(out)) != 0)
        # )[0]
        # # for i in range(0,len(tokens_pot)):
        # #     plt.scatter(tokens_pot[i], out[i], s = 8, c='b')
        # tokens_pot = np.append(tokens_pot, 0)

        cap_curve = np.gradient(
            np.gradient(cap_plot, ph_array[1] - ph_array[0]), ph_array[1] - ph_array[0]
        )
        plt.plot(ph_array, cap_curve)
        # tokens_cap = np.where(
        #     np.logical_and(np.abs(np.diff(out)) >= 1e-10, np.diff(np.sign(out)) != 0)
        # )[0]
        # tokens_cap = np.append(tokens_cap, 0)

        # plt.show()
        plt.savefig("test.svg")
        return pot_curve, cap_curve

    def get(self, chain):
        k_vals = self.get_kvals(chain[:-1])
        # print(k_vals)
        return self.get_tokens(k_vals)
