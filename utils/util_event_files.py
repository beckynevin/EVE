# Better handling for event files
# This is useful for understanding the properties of various datasets (in a comparison way)
# This takes a lot from Grant's hyperscreen code
from astropy.io import fits
from astropy.table import Table
import numpy as np
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

class HRCevt1:
    """This is a conceptual class representation of a Chandra High Resolution Camera (HRC) Level 1 Event File
    :return: HRCevt1 object
    :rtype: pandas.DataFrame or astropy.table.table.Table
    """

    def __init__(self, evt1file, verbose=False, as_astropy_table=False):
        """The constructor method for the HRCevt1 class
        :param evt1file: A .fits (or fits.gz) file containing the level 1 event list. If downloaded from the Chandra database, this file always has a *evt1.fits extension. This event list includes all events telemetered.
        :type evt1file: .fits or .fits.gz
        :param verbose: Set verbose=True to make the constructor chatty on the command line, defaults to False
        :type verbose: bool, optional
        :param as_astropy_table: Set as_astropy_table to True in order to have the HRCevt1 constructor method return an Astropy Table object, rather than a Pandas DataFrame. Defaults to False.
        :type as_astropy_table: bool, optional
        """

        # Define how chatty to be
        self.verbose = verbose

        if self.verbose is True:
            print(colorama.Fore.BLUE + '\nParsing HRC EVT1 file...', end=" ")
        # Do a standard read in of the EVT1 fits table
        self.filename = evt1file
        self.hdulist = fits.open(evt1file)
        self.data = Table(self.hdulist[1].data)
        self.header = self.hdulist[1].header
        self.gti = self.hdulist[2].data
        self.hdulist.close()  # Don't forget to close your fits file!

        # Make sure the user isn't running this on an ACIS observation!
        if self.header["DETNAM"][:4] == 'ACIS':
            raise Exception(
                "ERROR: HRCevt1 objects can only be initialized for Chandra/HRC observations. This is a Chandra/ACIS observation.")

        # Populate the fp, fb values for ever event
        if self.verbose is True:
            print(colorama.Fore.BLUE + 'Calculating fp, fb values...', end=" ")
        fp_u, fb_u, fp_v, fb_v = self.calculate_fp_fb()

        # Populate the fp, fb values for ever event
        if self.verbose is True:
            print(colorama.Fore.BLUE + 'Applying GTI mask... ', end=" ")
        self.gti.starts = self.gti['START']
        self.gti.stops = self.gti['STOP']

        self.gtimask = (self.data["time"] > self.gti.starts[0]) & (
            self.data["time"] < self.gti.stops[-1])

        # Populate the fp, fb values for every event
        if self.verbose is True:
            print(colorama.Fore.BLUE + 'Populating metadata columns...', end=" ")
        self.data["fp_u"] = fp_u
        self.data["fb_u"] = fb_u
        self.data["fp_v"] = fp_v
        self.data["fb_v"] = fb_v

        # Make individual status bit columns with legible names
        self.data["AV3 corrected for ringing"] = self.data["status"][:, 0]
        self.data["AU3 corrected for ringing"] = self.data["status"][:, 1]
        self.data["Event impacted by prior event (piled up)"] = self.data["status"][:, 2]
        # Bit 4 (Python 3) is spare
        self.data["Shifted event time"] = self.data["status"][:, 4]
        self.data["Event telemetered in NIL mode"] = self.data["status"][:, 5]
        self.data["V axis not triggered"] = self.data["status"][:, 6]
        self.data["U axis not triggered"] = self.data["status"][:, 7]
        self.data["V axis center blank event"] = self.data["status"][:, 8]
        self.data["U axis center blank event"] = self.data["status"][:, 9]
        self.data["V axis width exceeded"] = self.data["status"][:, 10]
        self.data["U axis width exceeded"] = self.data["status"][:, 11]
        self.data["Shield PMT active"] = self.data["status"][:, 12]
        # Bit 14 (Python 13) is hardware spare
        self.data["Upper level discriminator not exceeded"] = self.data["status"][:, 14]
        self.data["Lower level discriminator not exceeded"] = self.data["status"][:, 15]
        self.data["Event in bad region"] = self.data["status"][:, 16]
        self.data["Amp total on V or U = 0"] = self.data["status"][:, 17]
        self.data["Incorrect V center"] = self.data["status"][:, 18]
        self.data["Incorrect U center"] = self.data["status"][:, 19]
        self.data["PHA ratio test failed"] = self.data["status"][:, 20]
        self.data["Sum of 6 taps = 0"] = self.data["status"][:, 21]
        self.data["Grid ratio test failed"] = self.data["status"][:, 22]
        self.data["ADC sum on V or U = 0"] = self.data["status"][:, 23]
        self.data["PI exceeding 255"] = self.data["status"][:, 24]
        self.data["Event time tag is out of sequence"] = self.data["status"][:, 25]
        self.data["V amp flatness test failed"] = self.data["status"][:, 26]
        self.data["U amp flatness test failed"] = self.data["status"][:, 27]
        self.data["V amp saturation test failed"] = self.data["status"][:, 28]
        self.data["U amp saturation test failed"] = self.data["status"][:, 29]
        self.data["V hyperbolic test failed"] = self.data["status"][:, 30]
        self.data["U hyperbolic test failed"] = self.data["status"][:, 31]
        self.data["Hyperbola test passed"] = np.logical_not(np.logical_or(
            self.data['U hyperbolic test failed'], self.data['V hyperbolic test failed']))
        self.data["Hyperbola test failed"] = np.logical_or(
            self.data['U hyperbolic test failed'], self.data['V hyperbolic test failed'])

        self.obsid = self.header["OBS_ID"]
        self.obs_date = self.header["DATE"]
        self.target = self.header["OBJECT"]
        self.detector = self.header["DETNAM"]
        self.grating = self.header["GRATING"]
        self.exptime = self.header["EXPOSURE"]

        self.numevents = len(self.data["time"])
        if gti:
            self.goodtimeevents = len(self.data["time"][self.gtimask])
        self.badtimeevents = self.numevents - self.goodtimeevents

        self.hyperbola_passes = np.sum(np.logical_or(
            self.data['U hyperbolic test failed'], self.data['V hyperbolic test failed']))
        self.hyperbola_failures = np.sum(np.logical_not(np.logical_or(
            self.data['U hyperbolic test failed'], self.data['V hyperbolic test failed'])))

        if self.hyperbola_passes + self.hyperbola_failures != self.numevents:
            warnings.warn("Number of Hyperbola Test Failures and Passes ({}) does not equal total number of events ({}).".format(
                self.hyperbola_passes + self.hyperbola_failures, self.numevents))

        if self.verbose is True:
            print(colorama.Fore.GREEN + 'Done')

        
        if as_astropy_table is False:
            # Multidimensional columns don't grok with Pandas
            self.data.remove_column('status')
            self.data = self.data.to_pandas()

        if self.verbose is True:
            print(colorama.Fore.GREEN + 'Done')

    def __str__(self):
        """This method returns the string representation of the HRCevt1 object. It is called when the print() or str() function is invoked on an HRCevt1 object.
        :return: A string describing the HRCevt1 object
        :rtype: str
        """
        return "HRC EVT1 object with {} events. Data is packaged as a Pandas Dataframe (or an Astropy Table if as_astropy_table=True on initialization.)".format(self.numevents)

    def calculate_fp_fb(self):
        """Method to calculate the Fine Position (f_p) and normalized central tap amplitude (fb) for the HRC U- and V- axes.
        :return: fp_u, fb_u, fp_v, fb_v; the calculated fine positions and normalized central tap amplitudes, respectively, for the HRC U- and V- axes of the I or S detector
        :rtype: float
        """

        a_u = self.data["au1"]  # otherwise known as "a1"
        b_u = self.data["au2"]  # "a2"
        c_u = self.data["au3"]  # "a3"

        a_v = self.data["av1"]
        b_v = self.data["av2"]
        c_v = self.data["av3"]

        with np.errstate(invalid='ignore'):
            # Do the U axis
            fp_u = ((c_u - a_u) / (a_u + b_u + c_u))
            fb_u = b_u / (a_u + b_u + c_u)

            # Do the V axis
            fp_v = ((c_v - a_v) / (a_v + b_v + c_v))
            fb_v = b_v / (a_v + b_v + c_v)

        return fp_u, fb_u, fp_v, fb_v

    def threshold(self, img, bins, softening=None):
        """HyperScreen (a) separates events by both axis and tap, (b) creates an
        image (a 2D histogram, with bin sizes dependent on the total number of counts
        in the observation), then (c) performs efficient image segmentation on that image
        by applying Otsu's Method (https://en.wikipedia.org/wiki/Otsu%27s_method) to
        create a bespoke threshold for each tap image. This efficiently separates the 'boomerang'
        locus for 'real' events with the 'everything else' region for background events. This function
        performs operation (c).
        Arguments:
            img {[type]} -- [description]
            bins {[type]} -- [description]
        """

        # You don't want to be verbose in this function; it's called many times

        thresh_img = img.copy()
        thresh_img[img == 0] = np.nan

        if softening is None:
            thresh = filters.threshold_otsu(img)
        elif isinstance(softening, float):
            otsu_thresh = filters.threshold_otsu(img)
            thresh = otsu_thresh - (otsu_thresh * softening)

        # "If you don't ignore the warning, you'll get a warning" ~~ G. Tremblay
        with np.errstate(invalid='ignore'):
            thresh_img[thresh_img < thresh] = np.nan
        thresh_img[:int(bins[1] / 2), :] = np.nan
    #     thresh_img[:,int(bins[1]-5):] = np.nan

        return thresh_img

    def hyperscreen(self, softening=1.0):
        """[summary]
        Returns:
            [type] -- [description]
        """

        data = self.data[self.data['Hyperbola test passed']]

        # taprange = range(data['crsu'].min(), data['crsu'].max() + 1)
        taprange_u = range(data['crsu'].min() - 1, data['crsu'].max() + 1)
        taprange_v = range(data['crsv'].min() - 1, data['crsv'].max() + 1)

        if self.numevents < 100000:
            bins = [50, 50]  # number of bins
        else:
            bins = [200, 200]

        # Instantiate these empty dictionaries to hold our results
        u_axis_survivals = {}
        v_axis_survivals = {}

        if self.verbose is False:
            progressbar_disable = True
        elif self.verbose is True:
            progressbar_disable = False

        if self.verbose is True:
            print(colorama.Fore.YELLOW + "\nApplying Otsu's Method to every Tap-specific boomerang across U-axis taps {} through {}".format(taprange_u[0] + 1, taprange_u[-1] + 1))

        skiptaps_u = []
        skiptaps_v = []

        for tap in progressbar(taprange_u, disable=progressbar_disable, ascii=False):
            # Do the U axis
            tapmask_u = data[data['crsu'] == tap].index.values
            if len(tapmask_u) < 20:
                skiptaps_u.append((tap + 1, len(tapmask_u)))
                continue
            keep_u = np.isfinite(data['fb_u'][tapmask_u])

            hist_u, xbounds_u, ybounds_u = np.histogram2d(
                data['fb_u'][tapmask_u][keep_u], data['fp_u'][tapmask_u][keep_u], bins=bins)
            thresh_hist_u = self.threshold(
                hist_u, bins=bins, softening=softening)

            posx_u = np.digitize(data['fb_u'][tapmask_u], xbounds_u)
            posy_u = np.digitize(data['fp_u'][tapmask_u], ybounds_u)
            hist_mask_u = (posx_u > 0) & (posx_u <= bins[0]) & (
                posy_u > -1) & (posy_u <= bins[1])

            # Values of the histogram where the points are
            hhsub_u = thresh_hist_u[posx_u[hist_mask_u] -
                                    1, posy_u[hist_mask_u] - 1]
            pass_fb_u = data['fb_u'][tapmask_u][hist_mask_u][np.isfinite(
                hhsub_u)]

            u_axis_survivals["U Axis Tap {:02d}".format(
                tap)] = pass_fb_u.index.values

        if self.verbose is True:
            print("\nThe following {} U-axis taps were skipped due to a (very) low number of counts: ".format(len(skiptaps_u)))
            for skipped_tap in skiptaps_u:
                tapnum, counts = skipped_tap
                print("Skipped U-axis Tap {}, which had {} count(s)".format(tapnum, counts))
            print(colorama.Fore.MAGENTA + "\n... doing the same for the V axis taps {} through {}".format(taprange_v[0] + 1, taprange_v[-1] + 1))

        for tap in progressbar(taprange_v, disable=progressbar_disable, ascii=False):
            # Now do the V axis:
            tapmask_v = data[data['crsv'] == tap].index.values
            if len(tapmask_v) < 20:
                skiptaps_v.append((tap + 1, len(tapmask_v)))
                continue
            keep_v = np.isfinite(data['fb_v'][tapmask_v])

            hist_v, xbounds_v, ybounds_v = np.histogram2d(
                data['fb_v'][tapmask_v][keep_v], data['fp_v'][tapmask_v][keep_v], bins=bins)
            thresh_hist_v = self.threshold(
                hist_v, bins=bins, softening=softening)

            posx_v = np.digitize(data['fb_v'][tapmask_v], xbounds_v)
            posy_v = np.digitize(data['fp_v'][tapmask_v], ybounds_v)
            hist_mask_v = (posx_v > 0) & (posx_v <= bins[0]) & (
                posy_v > -1) & (posy_v <= bins[1])

            # Values of the histogram where the points are
            hhsub_v = thresh_hist_v[posx_v[hist_mask_v] -
                                    1, posy_v[hist_mask_v] - 1]
            pass_fb_v = data['fb_v'][tapmask_v][hist_mask_v][np.isfinite(
                hhsub_v)]

            v_axis_survivals["V Axis Tap {:02d}".format(
                tap)] = pass_fb_v.index.values

        if self.verbose is True:
            print("\nThe following {} V-axis taps were skipped due to a (very) low number of counts: ".format(len(skiptaps_v)))
            for skipped_tap in skiptaps_v:
                tapnum, counts = skipped_tap
                print("Skipped V-axis Tap {}, which had {} count(s)".format(tapnum, counts))

        # Done looping over taps

        if self.verbose is True:
            print(colorama.Fore.BLUE + "\nCollecting events that pass both U- and V-axis HyperScreen tests...", end=" ")

        u_all_survivals = np.concatenate(
            [x for x in u_axis_survivals.values()])
        v_all_survivals = np.concatenate(
            [x for x in v_axis_survivals.values()])

        # If the event passes both U- and V-axis tests, it survives
        all_survivals = np.intersect1d(u_all_survivals, v_all_survivals)
        survival_mask = np.isin(self.data.index.values, all_survivals)
        failure_mask = np.logical_not(survival_mask)

        num_survivals = sum(survival_mask)
        num_failures = sum(failure_mask)

        percent_hyperscreen_rejected = round(
            ((num_failures / self.numevents) * 100), 2)

        # Do a sanity check to look for lost events. Shouldn't be any.
        if num_survivals + num_failures != self.numevents:
            print("WARNING: Total Number of survivals and failures does \
            not equal total events in the EVT1 file. Something is wrong!")

        legacy_hyperbola_test_failures = sum(
            self.data['Hyperbola test failed'])
        percent_legacy_hyperbola_test_rejected = round(
            ((legacy_hyperbola_test_failures / self.numevents) * 100), 2)

        percent_improvement_over_legacy_test = round(
            (percent_hyperscreen_rejected - percent_legacy_hyperbola_test_rejected), 2)

        if self.verbose is True:
            print("Done")
            print(colorama.Fore.GREEN + "HyperScreen rejected" + colorama.Fore.YELLOW + " {}% of all events ({:,} bad events / {:,} total events)".format(percent_hyperscreen_rejected, sum(failure_mask), self.numevents) + colorama.Fore.GREEN +
                  "\nThe Murray+ algorithm rejects" + colorama.Fore.MAGENTA + " {}% of all events ({:,} bad events / {:,} total events)".format(percent_legacy_hyperbola_test_rejected, legacy_hyperbola_test_failures, self.numevents))

            print(colorama.Fore.GREEN + "As long as the results pass sanity checks, this is a POTENTIAL improvement of \n" +
                  colorama.Fore.BLUE + "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ POTENTIAL Improvement ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n" +
                  colorama.Fore.WHITE + "                                      {}%\n".format(percent_improvement_over_legacy_test) +
                  colorama.Fore.BLUE + "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

        hyperscreen_results_dict = {"ObsID": self.obsid,
                                    "Target": self.target,
                                    "Exposure Time": self.exptime,
                                    "Detector": self.detector,
                                    "Number of Events": self.numevents,
                                    "Number of Good Time Events": self.goodtimeevents,
                                    "U Axis Survivals by Tap": u_axis_survivals,
                                    "V Axis Survivals by Tap": v_axis_survivals,
                                    "U Axis All Survivals": u_all_survivals,
                                    "V Axis All Survivals": v_all_survivals,
                                    "All Survivals (event indices)": all_survivals,
                                    "All Survivals (boolean mask)": survival_mask,
                                    "All Failures (boolean mask)": failure_mask,
                                    "Percent rejected by Tapscreen": percent_hyperscreen_rejected,
                                    "Percent rejected by Hyperbola": percent_legacy_hyperbola_test_rejected,
                                    "Percent improvement": percent_improvement_over_legacy_test
                                    }

        return hyperscreen_results_dict

    def hyperbola(self, fb, a, b, h):
        """Given the normalized central tap amplitude, a, b, and h,
        return an array of length len(fb) that gives a hyperbola.
        Arguments:
            fb {[type]} -- [description]
            a {[type]} -- [description]
            b {[type]} -- [description]
            h {[type]} -- [description]
        Returns:
            [type] -- [description]
        """

        hyperbola = b * np.sqrt(((fb - h)**2 / a**2) - 1)

        return hyperbola

    def legacy_hyperbola_test(self, tolerance=0.035):
        """[summary]
        Keyword Arguments:
            tolerance {float} -- [description] (default: {0.035})
        Returns:
            [type] -- [description]
        """

        # Remind the user what tolerance they're using
        # print("{0: <25}| Using tolerance = {1}".format(" ", tolerance))

        # Set hyperbolic coefficients, depending on whether this is HRC-I or -S
        if self.detector == "HRC-I":
            a_u = 0.3110
            b_u = 0.3030
            h_u = 1.0580

            a_v = 0.3050
            b_v = 0.2730
            h_v = 1.1
            # print("{0: <25}| Using HRC-I hyperbolic coefficients: ".format(" "))
            # print("{0: <25}|    Au={1}, Bu={2}, Hu={3}".format(" ", a_u, b_u, h_u))
            # print("{0: <25}|    Av={1}, Bv={2}, Hv={3}".format(" ", a_v, b_v, h_v))

        if self.detector == "HRC-S":
            a_u = 0.2706
            b_u = 0.2620
            h_u = 1.0180

            a_v = 0.2706
            b_v = 0.2480
            h_v = 1.0710
            # print("{0: <25}| Using HRC-S hyperbolic coefficients: ".format(" "))
            # print("{0: <25}|    Au={1}, Bu={2}, Hu={3}".format(" ", a_u, b_u, h_u))
            # print("{0: <25}|    Av={1}, Bv={2}, Hv={3}".format(" ", a_v, b_v, h_v))

        # Set the tolerance boundary ("width" of the hyperbolic region)

        h_u_lowerbound = h_u * (1 + tolerance)
        h_u_upperbound = h_u * (1 - tolerance)

        h_v_lowerbound = h_v * (1 + tolerance)
        h_v_upperbound = h_v * (1 - tolerance)

        # Compute the Hyperbolae
        with np.errstate(invalid='ignore'):
            zone_u_fit = self.hyperbola(self.data["fb_u"], a_u, b_u, h_u)
            zone_u_lowerbound = self.hyperbola(
                self.data["fb_u"], a_u, b_u, h_u_lowerbound)
            zone_u_upperbound = self.hyperbola(
                self.data["fb_u"], a_u, b_u, h_u_upperbound)

            zone_v_fit = self.hyperbola(self.data["fb_v"], a_v, b_v, h_v)
            zone_v_lowerbound = self.hyperbola(
                self.data["fb_v"], a_v, b_v, h_v_lowerbound)
            zone_v_upperbound = self.hyperbola(
                self.data["fb_v"], a_v, b_v, h_v_upperbound)

        zone_u = [zone_u_lowerbound, zone_u_upperbound]
        zone_v = [zone_v_lowerbound, zone_v_upperbound]

        # Apply the masks
        # print("{0: <25}| Hyperbolic masks for U and V axes computed".format(""))

        with np.errstate(invalid='ignore'):
            # print("{0: <25}| Creating U-axis mask".format(""), end=" |")
            between_u = np.logical_not(np.logical_and(
                self.data["fp_u"] < zone_u[1], self.data["fp_u"] > -1 * zone_u[1]))
            not_beyond_u = np.logical_and(
                self.data["fp_u"] < zone_u[0], self.data["fp_u"] > -1 * zone_u[0])
            condition_u_final = np.logical_and(between_u, not_beyond_u)

            # print(" Creating V-axis mask")
            between_v = np.logical_not(np.logical_and(
                self.data["fp_v"] < zone_v[1], self.data["fp_v"] > -1 * zone_v[1]))
            not_beyond_v = np.logical_and(
                self.data["fp_v"] < zone_v[0], self.data["fp_v"] > -1 * zone_v[0])
            condition_v_final = np.logical_and(between_v, not_beyond_v)

        mask_u = condition_u_final
        mask_v = condition_v_final

        hyperzones = {"zone_u_fit": zone_u_fit,
                      "zone_u_lowerbound": zone_u_lowerbound,
                      "zone_u_upperbound": zone_u_upperbound,
                      "zone_v_fit": zone_v_fit,
                      "zone_v_lowerbound": zone_v_lowerbound,
                      "zone_v_upperbound": zone_v_upperbound}

        hypermasks = {"mask_u": mask_u, "mask_v": mask_v}

        # print("{0: <25}| Hyperbolic masks created".format(""))
        # print("{0: <25}| ".format(""))
        return hyperzones, hypermasks
    def image(self, masked_x=None, masked_y=None, xlim=None, ylim=None, detcoords=False, title=None, cmap=None, show=True, rasterized=True, savepath=None, create_subplot=False, ax=None, nbins=(400, 400)):
        """Create a quicklook image, in detector or sky coordinates, of the
        observation. The image will be binned to 400x400 by default.
        Keyword Arguments:
            masked_x {[type]} -- [description] (default: {None})
            masked_y {[type]} -- [description] (default: {None})
            xlim {[type]} -- [description] (default: {None})
            ylim {[type]} -- [description] (default: {None})
            detcoords {bool} -- [description] (default: {False})
            title {[type]} -- [description] (default: {None})
            cmap {[type]} -- [description] (default: {None})
            show {bool} -- [description] (default: {True})
            rasterized {bool} -- [description] (default: {True})
            savepath {[type]} -- [description] (default: {None})
            create_subplot {bool} -- [description] (default: {False})
            ax {[type]} -- [description] (default: {None})
        """

        if masked_x is not None and masked_y is not None:
            x = masked_x
            y = masked_y
            img_data, yedges, xedges = np.histogram2d(y, x, nbins)
        else:
            if detcoords is False:
                x = self.data['x'][self.gtimask]
                y = self.data['y'][self.gtimask]
            elif detcoords is True:
                x = self.data['detx'][self.gtimask]
                y = self.data['dety'][self.gtimask]
            img_data, yedges, xedges = np.histogram2d(y, x, nbins)

        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        # Create the Figure
        styleplots()

        # You can plot the image on axes of a subplot by passing
        # that axis to this function. Here are some switches to enable that.
        if create_subplot is False:
            self.fig, self.ax = plt.subplots()
        elif create_subplot is True:
            if ax is None:
                self.ax = plt.gca()
            else:
                self.ax = ax

        self.ax.grid(False)

        if cmap is None:
            cmap = 'viridis'

        im = self.ax.imshow(img_data, extent=extent, norm=LogNorm(),
                       interpolation=None, rasterized=rasterized, cmap=cmap, origin='lower')
        plt.colorbar(im)
        if title is None:
            self.ax.set_title("ObsID {} | {} | {} | {:,} events".format(
                self.obsid, self.target, self.detector, self.goodtimeevents))
        else:
            self.ax.set_title("{}".format(title))
        if detcoords is False:
            self.ax.set_xlabel("Sky X")
            self.ax.set_ylabel("Sky Y")
        elif detcoords is True:
            self.ax.set_xlabel("Detector X")
            self.ax.set_ylabel("Detector Y")

        if xlim is not None:
            self.ax.set_xlim(xlim)
        if ylim is not None:
            self.ax.set_ylim(ylim)

        if show is True:
            plt.show(block=True)

        if savepath is not None:
            plt.savefig('{}'.format(savepath))
            print("Saved image to {}".format(savepath))

        plt.close()

def styleplots():  # pragma: no cover
    """Make the plots pretty.
    """

    mpl.rcParams['agg.path.chunksize'] = 10000

    # Make things pretty
    plt.style.use('ggplot')

    labelsizes = 10

    plt.rcParams['font.size'] = labelsizes
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = labelsizes
    plt.rcParams['xtick.labelsize'] = labelsizes
    plt.rcParams['ytick.labelsize'] = labelsizes
class HRCevt1:
    """This is a conceptual class representation of a Chandra High Resolution Camera (HRC) Level 1 Event File
    :return: HRCevt1 object
    :rtype: pandas.DataFrame or astropy.table.table.Table
    """

    def __init__(self, evt1file, verbose=False, as_astropy_table=False):
        """The constructor method for the HRCevt1 class
        :param evt1file: A .fits (or fits.gz) file containing the level 1 event list. If downloaded from the Chandra database, this file always has a *evt1.fits extension. This event list includes all events telemetered.
        :type evt1file: .fits or .fits.gz
        :param verbose: Set verbose=True to make the constructor chatty on the command line, defaults to False
        :type verbose: bool, optional
        :param as_astropy_table: Set as_astropy_table to True in order to have the HRCevt1 constructor method return an Astropy Table object, rather than a Pandas DataFrame. Defaults to False.
        :type as_astropy_table: bool, optional
        """

        # Define how chatty to be
        self.verbose = verbose

        if self.verbose is True:
            print(colorama.Fore.BLUE + '\nParsing HRC EVT1 file...', end=" ")
        # Do a standard read in of the EVT1 fits table
        self.filename = evt1file
        self.hdulist = fits.open(evt1file)
        self.data = Table(self.hdulist[1].data)
        self.header = self.hdulist[1].header
        self.gti = self.hdulist[2].data
        self.hdulist.close()  # Don't forget to close your fits file!

        # Make sure the user isn't running this on an ACIS observation!
        if self.header["DETNAM"][:4] == 'ACIS':
            raise Exception(
                "ERROR: HRCevt1 objects can only be initialized for Chandra/HRC observations. This is a Chandra/ACIS observation.")

        # Populate the fp, fb values for ever event
        if self.verbose is True:
            print(colorama.Fore.BLUE + 'Calculating fp, fb values...', end=" ")
        fp_u, fb_u, fp_v, fb_v = self.calculate_fp_fb()

        # Populate the fp, fb values for ever event
        if self.verbose is True:
            print(colorama.Fore.BLUE + 'Applying GTI mask... ', end=" ")
        self.gti.starts = self.gti['START']
        self.gti.stops = self.gti['STOP']

        self.gtimask = (self.data["time"] > self.gti.starts[0]) & (
            self.data["time"] < self.gti.stops[-1])

        # Populate the fp, fb values for every event
        if self.verbose is True:
            print(colorama.Fore.BLUE + 'Populating metadata columns...', end=" ")
        self.data["fp_u"] = fp_u
        self.data["fb_u"] = fb_u
        self.data["fp_v"] = fp_v
        self.data["fb_v"] = fb_v

        # Make individual status bit columns with legible names
        self.data["AV3 corrected for ringing"] = self.data["status"][:, 0]
        self.data["AU3 corrected for ringing"] = self.data["status"][:, 1]
        self.data["Event impacted by prior event (piled up)"] = self.data["status"][:, 2]
        # Bit 4 (Python 3) is spare
        self.data["Shifted event time"] = self.data["status"][:, 4]
        self.data["Event telemetered in NIL mode"] = self.data["status"][:, 5]
        self.data["V axis not triggered"] = self.data["status"][:, 6]
        self.data["U axis not triggered"] = self.data["status"][:, 7]
        self.data["V axis center blank event"] = self.data["status"][:, 8]
        self.data["U axis center blank event"] = self.data["status"][:, 9]
        self.data["V axis width exceeded"] = self.data["status"][:, 10]
        self.data["U axis width exceeded"] = self.data["status"][:, 11]
        self.data["Shield PMT active"] = self.data["status"][:, 12]
        # Bit 14 (Python 13) is hardware spare
        self.data["Upper level discriminator not exceeded"] = self.data["status"][:, 14]
        self.data["Lower level discriminator not exceeded"] = self.data["status"][:, 15]
        self.data["Event in bad region"] = self.data["status"][:, 16]
        self.data["Amp total on V or U = 0"] = self.data["status"][:, 17]
        self.data["Incorrect V center"] = self.data["status"][:, 18]
        self.data["Incorrect U center"] = self.data["status"][:, 19]
        self.data["PHA ratio test failed"] = self.data["status"][:, 20]
        self.data["Sum of 6 taps = 0"] = self.data["status"][:, 21]
        self.data["Grid ratio test failed"] = self.data["status"][:, 22]
        self.data["ADC sum on V or U = 0"] = self.data["status"][:, 23]
        self.data["PI exceeding 255"] = self.data["status"][:, 24]
        self.data["Event time tag is out of sequence"] = self.data["status"][:, 25]
        self.data["V amp flatness test failed"] = self.data["status"][:, 26]
        self.data["U amp flatness test failed"] = self.data["status"][:, 27]
        self.data["V amp saturation test failed"] = self.data["status"][:, 28]
        self.data["U amp saturation test failed"] = self.data["status"][:, 29]
        self.data["V hyperbolic test failed"] = self.data["status"][:, 30]
        self.data["U hyperbolic test failed"] = self.data["status"][:, 31]
        self.data["Hyperbola test passed"] = np.logical_not(np.logical_or(
            self.data['U hyperbolic test failed'], self.data['V hyperbolic test failed']))
        self.data["Hyperbola test failed"] = np.logical_or(
            self.data['U hyperbolic test failed'], self.data['V hyperbolic test failed'])

        self.obsid = self.header["OBS_ID"]
        self.obs_date = self.header["DATE"]
        self.target = self.header["OBJECT"]
        self.detector = self.header["DETNAM"]
        self.grating = self.header["GRATING"]
        self.exptime = self.header["EXPOSURE"]

        self.numevents = len(self.data["time"])
        self.goodtimeevents = len(self.data["time"][self.gtimask])
        self.badtimeevents = self.numevents - self.goodtimeevents

        self.hyperbola_passes = np.sum(np.logical_or(
            self.data['U hyperbolic test failed'], self.data['V hyperbolic test failed']))
        self.hyperbola_failures = np.sum(np.logical_not(np.logical_or(
            self.data['U hyperbolic test failed'], self.data['V hyperbolic test failed'])))

        if self.hyperbola_passes + self.hyperbola_failures != self.numevents:
            warnings.warn("Number of Hyperbola Test Failures and Passes ({}) does not equal total number of events ({}).".format(
                self.hyperbola_passes + self.hyperbola_failures, self.numevents))

        if self.verbose is True:
            print(colorama.Fore.GREEN + 'Done')

        
        if as_astropy_table is False:
            # Multidimensional columns don't grok with Pandas
            self.data.remove_column('status')
            self.data = self.data.to_pandas()

        if self.verbose is True:
            print(colorama.Fore.GREEN + 'Done')

    def __str__(self):
        """This method returns the string representation of the HRCevt1 object. It is called when the print() or str() function is invoked on an HRCevt1 object.
        :return: A string describing the HRCevt1 object
        :rtype: str
        """
        return "HRC EVT1 object with {} events. Data is packaged as a Pandas Dataframe (or an Astropy Table if as_astropy_table=True on initialization.)".format(self.numevents)

    def calculate_fp_fb(self):
        """Method to calculate the Fine Position (f_p) and normalized central tap amplitude (fb) for the HRC U- and V- axes.
        :return: fp_u, fb_u, fp_v, fb_v; the calculated fine positions and normalized central tap amplitudes, respectively, for the HRC U- and V- axes of the I or S detector
        :rtype: float
        """

        a_u = self.data["au1"]  # otherwise known as "a1"
        b_u = self.data["au2"]  # "a2"
        c_u = self.data["au3"]  # "a3"

        a_v = self.data["av1"]
        b_v = self.data["av2"]
        c_v = self.data["av3"]

        with np.errstate(invalid='ignore'):
            # Do the U axis
            fp_u = ((c_u - a_u) / (a_u + b_u + c_u))
            fb_u = b_u / (a_u + b_u + c_u)

            # Do the V axis
            fp_v = ((c_v - a_v) / (a_v + b_v + c_v))
            fb_v = b_v / (a_v + b_v + c_v)

        return fp_u, fb_u, fp_v, fb_v

    def threshold(self, img, bins, softening=None):
        """HyperScreen (a) separates events by both axis and tap, (b) creates an
        image (a 2D histogram, with bin sizes dependent on the total number of counts
        in the observation), then (c) performs efficient image segmentation on that image
        by applying Otsu's Method (https://en.wikipedia.org/wiki/Otsu%27s_method) to
        create a bespoke threshold for each tap image. This efficiently separates the 'boomerang'
        locus for 'real' events with the 'everything else' region for background events. This function
        performs operation (c).
        Arguments:
            img {[type]} -- [description]
            bins {[type]} -- [description]
        """

        # You don't want to be verbose in this function; it's called many times

        thresh_img = img.copy()
        thresh_img[img == 0] = np.nan

        if softening is None:
            thresh = filters.threshold_otsu(img)
        elif isinstance(softening, float):
            otsu_thresh = filters.threshold_otsu(img)
            thresh = otsu_thresh - (otsu_thresh * softening)

        # "If you don't ignore the warning, you'll get a warning" ~~ G. Tremblay
        with np.errstate(invalid='ignore'):
            thresh_img[thresh_img < thresh] = np.nan
        thresh_img[:int(bins[1] / 2), :] = np.nan
    #     thresh_img[:,int(bins[1]-5):] = np.nan

        return thresh_img

    def hyperscreen(self, softening=1.0):
        """[summary]
        Returns:
            [type] -- [description]
        """

        data = self.data[self.data['Hyperbola test passed']]

        # taprange = range(data['crsu'].min(), data['crsu'].max() + 1)
        taprange_u = range(data['crsu'].min() - 1, data['crsu'].max() + 1)
        taprange_v = range(data['crsv'].min() - 1, data['crsv'].max() + 1)

        if self.numevents < 100000:
            bins = [50, 50]  # number of bins
        else:
            bins = [200, 200]

        # Instantiate these empty dictionaries to hold our results
        u_axis_survivals = {}
        v_axis_survivals = {}

        if self.verbose is False:
            progressbar_disable = True
        elif self.verbose is True:
            progressbar_disable = False

        if self.verbose is True:
            print(colorama.Fore.YELLOW + "\nApplying Otsu's Method to every Tap-specific boomerang across U-axis taps {} through {}".format(taprange_u[0] + 1, taprange_u[-1] + 1))

        skiptaps_u = []
        skiptaps_v = []

        for tap in progressbar(taprange_u, disable=progressbar_disable, ascii=False):
            # Do the U axis
            tapmask_u = data[data['crsu'] == tap].index.values
            if len(tapmask_u) < 20:
                skiptaps_u.append((tap + 1, len(tapmask_u)))
                continue
            keep_u = np.isfinite(data['fb_u'][tapmask_u])

            hist_u, xbounds_u, ybounds_u = np.histogram2d(
                data['fb_u'][tapmask_u][keep_u], data['fp_u'][tapmask_u][keep_u], bins=bins)
            thresh_hist_u = self.threshold(
                hist_u, bins=bins, softening=softening)

            posx_u = np.digitize(data['fb_u'][tapmask_u], xbounds_u)
            posy_u = np.digitize(data['fp_u'][tapmask_u], ybounds_u)
            hist_mask_u = (posx_u > 0) & (posx_u <= bins[0]) & (
                posy_u > -1) & (posy_u <= bins[1])

            # Values of the histogram where the points are
            hhsub_u = thresh_hist_u[posx_u[hist_mask_u] -
                                    1, posy_u[hist_mask_u] - 1]
            pass_fb_u = data['fb_u'][tapmask_u][hist_mask_u][np.isfinite(
                hhsub_u)]

            u_axis_survivals["U Axis Tap {:02d}".format(
                tap)] = pass_fb_u.index.values

        if self.verbose is True:
            print("\nThe following {} U-axis taps were skipped due to a (very) low number of counts: ".format(len(skiptaps_u)))
            for skipped_tap in skiptaps_u:
                tapnum, counts = skipped_tap
                print("Skipped U-axis Tap {}, which had {} count(s)".format(tapnum, counts))
            print(colorama.Fore.MAGENTA + "\n... doing the same for the V axis taps {} through {}".format(taprange_v[0] + 1, taprange_v[-1] + 1))

        for tap in progressbar(taprange_v, disable=progressbar_disable, ascii=False):
            # Now do the V axis:
            tapmask_v = data[data['crsv'] == tap].index.values
            if len(tapmask_v) < 20:
                skiptaps_v.append((tap + 1, len(tapmask_v)))
                continue
            keep_v = np.isfinite(data['fb_v'][tapmask_v])

            hist_v, xbounds_v, ybounds_v = np.histogram2d(
                data['fb_v'][tapmask_v][keep_v], data['fp_v'][tapmask_v][keep_v], bins=bins)
            thresh_hist_v = self.threshold(
                hist_v, bins=bins, softening=softening)

            posx_v = np.digitize(data['fb_v'][tapmask_v], xbounds_v)
            posy_v = np.digitize(data['fp_v'][tapmask_v], ybounds_v)
            hist_mask_v = (posx_v > 0) & (posx_v <= bins[0]) & (
                posy_v > -1) & (posy_v <= bins[1])

            # Values of the histogram where the points are
            hhsub_v = thresh_hist_v[posx_v[hist_mask_v] -
                                    1, posy_v[hist_mask_v] - 1]
            pass_fb_v = data['fb_v'][tapmask_v][hist_mask_v][np.isfinite(
                hhsub_v)]

            v_axis_survivals["V Axis Tap {:02d}".format(
                tap)] = pass_fb_v.index.values

        if self.verbose is True:
            print("\nThe following {} V-axis taps were skipped due to a (very) low number of counts: ".format(len(skiptaps_v)))
            for skipped_tap in skiptaps_v:
                tapnum, counts = skipped_tap
                print("Skipped V-axis Tap {}, which had {} count(s)".format(tapnum, counts))

        # Done looping over taps

        if self.verbose is True:
            print(colorama.Fore.BLUE + "\nCollecting events that pass both U- and V-axis HyperScreen tests...", end=" ")

        u_all_survivals = np.concatenate(
            [x for x in u_axis_survivals.values()])
        v_all_survivals = np.concatenate(
            [x for x in v_axis_survivals.values()])

        # If the event passes both U- and V-axis tests, it survives
        all_survivals = np.intersect1d(u_all_survivals, v_all_survivals)
        survival_mask = np.isin(self.data.index.values, all_survivals)
        failure_mask = np.logical_not(survival_mask)

        num_survivals = sum(survival_mask)
        num_failures = sum(failure_mask)

        percent_hyperscreen_rejected = round(
            ((num_failures / self.numevents) * 100), 2)

        # Do a sanity check to look for lost events. Shouldn't be any.
        if num_survivals + num_failures != self.numevents:
            print("WARNING: Total Number of survivals and failures does \
            not equal total events in the EVT1 file. Something is wrong!")

        legacy_hyperbola_test_failures = sum(
            self.data['Hyperbola test failed'])
        percent_legacy_hyperbola_test_rejected = round(
            ((legacy_hyperbola_test_failures / self.numevents) * 100), 2)

        percent_improvement_over_legacy_test = round(
            (percent_hyperscreen_rejected - percent_legacy_hyperbola_test_rejected), 2)

        if self.verbose is True:
            print("Done")
            print(colorama.Fore.GREEN + "HyperScreen rejected" + colorama.Fore.YELLOW + " {}% of all events ({:,} bad events / {:,} total events)".format(percent_hyperscreen_rejected, sum(failure_mask), self.numevents) + colorama.Fore.GREEN +
                  "\nThe Murray+ algorithm rejects" + colorama.Fore.MAGENTA + " {}% of all events ({:,} bad events / {:,} total events)".format(percent_legacy_hyperbola_test_rejected, legacy_hyperbola_test_failures, self.numevents))

            print(colorama.Fore.GREEN + "As long as the results pass sanity checks, this is a POTENTIAL improvement of \n" +
                  colorama.Fore.BLUE + "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ POTENTIAL Improvement ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n" +
                  colorama.Fore.WHITE + "                                      {}%\n".format(percent_improvement_over_legacy_test) +
                  colorama.Fore.BLUE + "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

        hyperscreen_results_dict = {"ObsID": self.obsid,
                                    "Target": self.target,
                                    "Exposure Time": self.exptime,
                                    "Detector": self.detector,
                                    "Number of Events": self.numevents,
                                    "Number of Good Time Events": self.goodtimeevents,
                                    "U Axis Survivals by Tap": u_axis_survivals,
                                    "V Axis Survivals by Tap": v_axis_survivals,
                                    "U Axis All Survivals": u_all_survivals,
                                    "V Axis All Survivals": v_all_survivals,
                                    "All Survivals (event indices)": all_survivals,
                                    "All Survivals (boolean mask)": survival_mask,
                                    "All Failures (boolean mask)": failure_mask,
                                    "Percent rejected by Tapscreen": percent_hyperscreen_rejected,
                                    "Percent rejected by Hyperbola": percent_legacy_hyperbola_test_rejected,
                                    "Percent improvement": percent_improvement_over_legacy_test
                                    }

        return hyperscreen_results_dict

    def hyperbola(self, fb, a, b, h):
        """Given the normalized central tap amplitude, a, b, and h,
        return an array of length len(fb) that gives a hyperbola.
        Arguments:
            fb {[type]} -- [description]
            a {[type]} -- [description]
            b {[type]} -- [description]
            h {[type]} -- [description]
        Returns:
            [type] -- [description]
        """

        hyperbola = b * np.sqrt(((fb - h)**2 / a**2) - 1)

        return hyperbola

    def legacy_hyperbola_test(self, tolerance=0.035):
        """[summary]
        Keyword Arguments:
            tolerance {float} -- [description] (default: {0.035})
        Returns:
            [type] -- [description]
        """

        # Remind the user what tolerance they're using
        # print("{0: <25}| Using tolerance = {1}".format(" ", tolerance))

        # Set hyperbolic coefficients, depending on whether this is HRC-I or -S
        if self.detector == "HRC-I":
            a_u = 0.3110
            b_u = 0.3030
            h_u = 1.0580

            a_v = 0.3050
            b_v = 0.2730
            h_v = 1.1
            # print("{0: <25}| Using HRC-I hyperbolic coefficients: ".format(" "))
            # print("{0: <25}|    Au={1}, Bu={2}, Hu={3}".format(" ", a_u, b_u, h_u))
            # print("{0: <25}|    Av={1}, Bv={2}, Hv={3}".format(" ", a_v, b_v, h_v))

        if self.detector == "HRC-S":
            a_u = 0.2706
            b_u = 0.2620
            h_u = 1.0180

            a_v = 0.2706
            b_v = 0.2480
            h_v = 1.0710
            # print("{0: <25}| Using HRC-S hyperbolic coefficients: ".format(" "))
            # print("{0: <25}|    Au={1}, Bu={2}, Hu={3}".format(" ", a_u, b_u, h_u))
            # print("{0: <25}|    Av={1}, Bv={2}, Hv={3}".format(" ", a_v, b_v, h_v))

        # Set the tolerance boundary ("width" of the hyperbolic region)

        h_u_lowerbound = h_u * (1 + tolerance)
        h_u_upperbound = h_u * (1 - tolerance)

        h_v_lowerbound = h_v * (1 + tolerance)
        h_v_upperbound = h_v * (1 - tolerance)

        # Compute the Hyperbolae
        with np.errstate(invalid='ignore'):
            zone_u_fit = self.hyperbola(self.data["fb_u"], a_u, b_u, h_u)
            zone_u_lowerbound = self.hyperbola(
                self.data["fb_u"], a_u, b_u, h_u_lowerbound)
            zone_u_upperbound = self.hyperbola(
                self.data["fb_u"], a_u, b_u, h_u_upperbound)

            zone_v_fit = self.hyperbola(self.data["fb_v"], a_v, b_v, h_v)
            zone_v_lowerbound = self.hyperbola(
                self.data["fb_v"], a_v, b_v, h_v_lowerbound)
            zone_v_upperbound = self.hyperbola(
                self.data["fb_v"], a_v, b_v, h_v_upperbound)

        zone_u = [zone_u_lowerbound, zone_u_upperbound]
        zone_v = [zone_v_lowerbound, zone_v_upperbound]

        # Apply the masks
        # print("{0: <25}| Hyperbolic masks for U and V axes computed".format(""))

        with np.errstate(invalid='ignore'):
            # print("{0: <25}| Creating U-axis mask".format(""), end=" |")
            between_u = np.logical_not(np.logical_and(
                self.data["fp_u"] < zone_u[1], self.data["fp_u"] > -1 * zone_u[1]))
            not_beyond_u = np.logical_and(
                self.data["fp_u"] < zone_u[0], self.data["fp_u"] > -1 * zone_u[0])
            condition_u_final = np.logical_and(between_u, not_beyond_u)

            # print(" Creating V-axis mask")
            between_v = np.logical_not(np.logical_and(
                self.data["fp_v"] < zone_v[1], self.data["fp_v"] > -1 * zone_v[1]))
            not_beyond_v = np.logical_and(
                self.data["fp_v"] < zone_v[0], self.data["fp_v"] > -1 * zone_v[0])
            condition_v_final = np.logical_and(between_v, not_beyond_v)

        mask_u = condition_u_final
        mask_v = condition_v_final

        hyperzones = {"zone_u_fit": zone_u_fit,
                      "zone_u_lowerbound": zone_u_lowerbound,
                      "zone_u_upperbound": zone_u_upperbound,
                      "zone_v_fit": zone_v_fit,
                      "zone_v_lowerbound": zone_v_lowerbound,
                      "zone_v_upperbound": zone_v_upperbound}

        hypermasks = {"mask_u": mask_u, "mask_v": mask_v}

        # print("{0: <25}| Hyperbolic masks created".format(""))
        # print("{0: <25}| ".format(""))
        return hyperzones, hypermasks
    def image(self, masked_x=None, masked_y=None, xlim=None, ylim=None, detcoords=False, title=None, cmap=None, show=True, rasterized=True, savepath=None, create_subplot=False, ax=None, nbins=(400, 400)):
        """Create a quicklook image, in detector or sky coordinates, of the
        observation. The image will be binned to 400x400 by default.
        Keyword Arguments:
            masked_x {[type]} -- [description] (default: {None})
            masked_y {[type]} -- [description] (default: {None})
            xlim {[type]} -- [description] (default: {None})
            ylim {[type]} -- [description] (default: {None})
            detcoords {bool} -- [description] (default: {False})
            title {[type]} -- [description] (default: {None})
            cmap {[type]} -- [description] (default: {None})
            show {bool} -- [description] (default: {True})
            rasterized {bool} -- [description] (default: {True})
            savepath {[type]} -- [description] (default: {None})
            create_subplot {bool} -- [description] (default: {False})
            ax {[type]} -- [description] (default: {None})
        """

        if masked_x is not None and masked_y is not None:
            x = masked_x
            y = masked_y
            img_data, yedges, xedges = np.histogram2d(y, x, nbins)
        else:
            if detcoords is False:
                x = self.data['x'][self.gtimask]
                y = self.data['y'][self.gtimask]
            elif detcoords is True:
                x = self.data['detx'][self.gtimask]
                y = self.data['dety'][self.gtimask]
            img_data, yedges, xedges = np.histogram2d(y, x, nbins)

        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        # Create the Figure
        styleplots()

        # You can plot the image on axes of a subplot by passing
        # that axis to this function. Here are some switches to enable that.
        if create_subplot is False:
            self.fig, self.ax = plt.subplots()
        elif create_subplot is True:
            if ax is None:
                self.ax = plt.gca()
            else:
                self.ax = ax

        self.ax.grid(False)

        if cmap is None:
            cmap = 'viridis'

        im = self.ax.imshow(img_data, extent=extent, norm=LogNorm(),
                       interpolation=None, rasterized=rasterized, cmap=cmap, origin='lower')
        plt.colorbar(im)
        if title is None:
            self.ax.set_title("ObsID {} | {} | {} | {:,} events".format(
                self.obsid, self.target, self.detector, self.goodtimeevents))
        else:
            self.ax.set_title("{}".format(title))
        if detcoords is False:
            self.ax.set_xlabel("Sky X")
            self.ax.set_ylabel("Sky Y")
        elif detcoords is True:
            self.ax.set_xlabel("Detector X")
            self.ax.set_ylabel("Detector Y")

        if xlim is not None:
            self.ax.set_xlim(xlim)
        if ylim is not None:
            self.ax.set_ylim(ylim)

        if show is True:
            plt.show(block=True)

        if savepath is not None:
            plt.savefig('{}'.format(savepath))
            print("Saved image to {}".format(savepath))

        plt.close()

def styleplots():  # pragma: no cover
    """Make the plots pretty.
    """

    mpl.rcParams['agg.path.chunksize'] = 10000

    # Make things pretty
    plt.style.use('ggplot')

    labelsizes = 10

    plt.rcParams['font.size'] = labelsizes
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = labelsizes
    plt.rcParams['xtick.labelsize'] = labelsizes
    plt.rcParams['ytick.labelsize'] = labelsizes

def plot_from_df(df, name, var1, var2, aspect):
    df = df.dropna()
    
    #print('this is the df you are trying to plot')
    #print(df)
 
    x = df[var1]#[0:n_points]#[np.array(binary_preds)==1]#model_lda.predict(xs)
    y = df[var2]#[0:n_points]#[np.array(binary_preds)==1]

    
    
    nbins=100#int(len(x)/10)
    
    img_data, yedges, xedges = np.histogram2d(y, x, nbins)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    #print('# Real events: ', len(input_data.loc[good_hyperbola]), '# All events: ',len(input_data))
    #print('# bg events: ', len(input_data)-len(input_data.loc[good_hyperbola]))
    try:
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = ax.imshow(ma.masked_where(img_data==0, img_data),  
                        rasterized=True, cmap='viridis',  extent=extent, 
                       norm=matplotlib.colors.LogNorm())
        ax.set_xlabel(var1)
        ax.set_ylabel(var2)
    
        plt.colorbar(im)
        plt.title(name)
        ax.set_aspect(aspect)
        #ax4.set_aspect('equal')
        plt.show()
    except:
        STOP
        plt.clf()
        plt.imshow(ma.masked_where(img_data==0, img_data),  
                        rasterized=True, cmap='viridis',  extent=extent, 
                   norm=matplotlib.colors.LogNorm())
        plt.xlabel(var1)
        plt.ylabel(var2)
        #plt.colorbar()
        plt.title(name)
        plt.set_aspect(aspect)
        plt.show()
    return ma.masked_where(img_data==0, img_data)
 


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap
def comparison_heatmap(df, df1, df2, xaxis, yaxis, title, norm=False):

    xs_1 = df1[xaxis]
    ys_1 = df1[yaxis]

    xs_2 = df2[xaxis]
    ys_2 = df2[yaxis]
    
    if norm:
        xs = df[xaxis]
        ys = df[yaxis]
        heatmap, xedges, yedges = np.histogram2d(ys, xs, bins=100)

    

    heatmap1, xedges1, yedges1 = np.histogram2d(ys_1, xs_1, bins=100)
    
    heatmap2, xedges2, yedges2 = np.histogram2d(ys_2, xs_2, bins=100)

    # Make a plot that is the higher probabilities minus the smaller probabilities
    plt.clf()
    fig = plt.figure()
    ax2=fig.add_subplot(111)

    orig_cmap = matplotlib.cm.coolwarm
    shifted_cmap = shiftedColorMap(orig_cmap, midpoint=0, name='shrunk')

    if norm:
        im2 = ax2.imshow((heatmap1 - heatmap2)/heatmap, 
                     cmap='RdBu_r', 
                     extent=[yedges1[0], yedges1[-1], xedges2[0], xedges2[-1]], vmin=0, vmax=1)#, 
                     #norm = matplotlib.colors.SymLogNorm(linthresh=0.03, linscale=1), 
                     #vmin=-10**-1, vmax=10**0, interpolation ='None')
    else:
        im2 = ax2.imshow((heatmap1 - heatmap2), 
                     cmap='RdBu_r', 
                     extent=[yedges1[0], yedges1[-1], xedges2[0], xedges2[-1]], 
                     norm = matplotlib.colors.SymLogNorm(linthresh=0.3, linscale=1))#, 
                     #vmin=-10**3, vmax=10**3, interpolation ='None')
    #ax2.set_ylim(xedgesmurray[0], xedgesmurray[-1])
    #ax2.set_xlim(yedgesmurray[0], yedgesmurray[-1])
    plt.colorbar(im2, fraction=0.046)
    ax2.set_xlabel(xaxis)
    ax2.set_ylabel(yaxis)
    #ax2.set_aspect((yedgesmurray[-1]-yedgesmurray[0])/(xedgesmurray[-1]-xedgesmurray[0]))

    plt.tight_layout()
    plt.title(title)
    plt.show()

