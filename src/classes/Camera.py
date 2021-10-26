import os
import time
import datetime

from .FileSystem import FileSystem
from .Conversions import dt2jd


class Camera(object):
    """
    Class Description:
    Camera Object used to communicate with cameras connected to the computer.
    Author(s):
    Jacob Taylor Cassady
    """

    def __init__(self, control_cmd_location=None, image_type=None, collection_name="",
                 aperture=None, exposure_control=None, shutter_speed=None, iso=None, camera_type="CloudCamera", timestring=None):
        # Initialize variables
        self.remote_command = 'C:/digiCamControl/CameraControlRemoteCmd.exe /c '
        self.control_cmd_location = self.set_control_cmd_location(control_cmd_location)
        self.image_type = self.set_image_type(image_type)
        self.image_index = 0
        self.image_name = ""
        self.camera_type = camera_type
        self.timestring = timestring
        self.collection_name = collection_name
        self.save_folder = None
        self.script_location = None
        self.script_location = ""

        self.aperture = str(aperture)
        self.exposure_control = str(exposure_control)
        self.shutter_speed = str(shutter_speed)
        self.iso = str(iso)

        self.jd_pre = None
        self.jd_post = None

    def setup(self, setup_script_name="setup.dccscript"):
        """
        Function Desciption:
        Drives the setup of the camera given a set of settings.  Autocodes the setup script and runs it.
        Author(s):
        Jacob Taylor Cassady
        """
        self.generate_setup_script(setup_script_name=setup_script_name)

        self.run_script(script_name=setup_script_name)

    def generate_setup_script(self, setup_script_name):
        """
        Function Description:
        Generates the setup script to set the aperture, exposure_control, shutter_speed, and iso of the camera if any of these values are passed.
            Author(s):
            Jacob Taylor Cassady
        """

        settings = {"aperture": self.aperture,
                    "shutter": self.shutter_speed,
                    "iso": self.iso}

        # Enforce directory location
        FileSystem.enforce_path(self.script_location)

        # Generate the setup script at the script location with the given setup_script_name
        with open(self.script_location + setup_script_name, "w+") as file:
            file.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
            file.write("<dccscript>\n")
            file.write(" " * 2 + "<commands>\n")
            self.write_settings(file, settings)
            file.write(" " * 2 + "</commands>\n")
            file.write("</dccscript>")

    @staticmethod
    def write_settings(file, settings):
        """
        Function Description:
        Writes the passed dictionary of settings to the passed file.  If a setting has a value of None, it is passed over.
        Author(s):
        Jacob Taylor Cassady
        """
        # Write each setting in the settings dictionary to the file as long as the setting is not None
        for setting in settings:
            if settings[setting] is not None:
                file.write(" " * 3 + "<setcamera property=\"" + str(setting) + "\" value=\"" + settings[setting] + "\"/>\n")

    @staticmethod
    def set_control_cmd_location(control_cmd_location=None):
        """
        Function Description:
        Sets the location of CameraControlCmd.exe which is used to command a camera from the command line using the program DigiCamControl.
        Author(s):
        Jacob Taylor Cassady
        """
        # If no control_cmd_location is given, use the default.
        if control_cmd_location is None:
            control_cmd_location = "\"C:\\digiCamControl\\CameraControlCmd.exe\""

        return control_cmd_location

    def get_image_name(self):
        self.jd_pre = dt2jd()
        p1 = str(self.jd_pre).ljust(20, '0')
        return p1 + self.image_type

    def command_camera(self, command):
        """
        Function Description:
        Creates a call to the camera using DigiCamControl
        Author(s):
        Jacob Taylor Cassady
        """
        # # Enforce directory location
        # FileSystem.enforce_path(self.save_folder)

        # Build image name
        # image_name = self.collection_name + "_" + str(self.image_index) + self.image_type

        self.image_name = os.path.join(self.save_folder, self.get_image_name()).replace('/', '\\')
        remote_command_details = 'capture ' + self.image_name
        full_command = self.remote_command + ' ' + remote_command_details

        # Command Camera
        # os.system(self.control_cmd_location + " /filename " + self.image_name + " " + command)
        os.system(full_command)
        self.jd_post = dt2jd()

    def run_script(self, script_name):
        """
        Function Description:
        Runs the passed script within the script location.
        Author(s):
        Jacob Taylor Cassady
        """
        # Enforce directory location
        FileSystem.enforce_path(self.script_location)

        # Make call to operating system
        os.system(self.control_cmd_location + " " + self.script_location + script_name)

    @staticmethod
    def set_image_type(image_type=None):
        """
        Function Description:
        Sets the image type.  If none is given, the default CannonRaw2 image type is used.
        Author(s):
        Jacob Taylor Cassady
        """
        if image_type == "jpeg" or image_type == "jpg":
            return ".jpg"
        elif image_type == 'raw':
            return ".RAW"
        elif image_type == 'png':
            return ".png"
        else:
            return ".CR2"

    def capture_single_image(self, autofocus=False, save_folder=None):
        """
        Function Description:
        Captures a single image.  Iterates the image index to ensure a unique name for each image taken.
        Author(s):
        Jacob Taylor Cassady
        """
        self.save_folder = save_folder if save_folder else None

        # Make the correct camera command depending on autofocus being enabled.
        self.command_camera("/capture") if autofocus else self.command_camera("/capturenoaf")

    def capture_multiple_images(self, image_count, save_folder=None):
        """
        Function Description:
        Captures an n number of images by repeatedly calling the capture_single_image function n times where n is the parameter image_count.
        Author(s):
        Jacob Taylor Cassady
        """
        self.save_folder = save_folder if save_folder else None
        # Iterate over the image_count capturing a single image every time.
        for capture in range(image_count):
            t0 = time.time()
            self.capture_single_image(save_folder=save_folder)
            print('Image # {0:4d} out of {1:4d} :: Time :: {2:4.4f}', capture, image_count, time.time()-t0)
        return True
