from observation import Observation
from camera import CameraModel
import matplotlib.pyplot as plt

import os
import re


BASE_DIRECTORY = os.path.curdir
DATA_DIRECTORY = os.path.join(BASE_DIRECTORY,"data")
CAMERA_FILE = os.path.join(DATA_DIRECTORY,"camera.dat")
MEAS_REGEX = r"^meas-.*\.dat$"

def main():

    meas_files = sorted([
        file for file in os.listdir(DATA_DIRECTORY)
        if re.match(MEAS_REGEX, file)
    ])

    observations = [
        Observation.from_file(os.path.join(DATA_DIRECTORY, filename))
        for filename in meas_files
    ]

    camera = CameraModel.from_file(CAMERA_FILE)


if __name__ == "__main__":
    main()

