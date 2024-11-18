import cv2
import easyocr
from numpy.core.defchararray import isnumeric, isupper, islower

from fencer import Fencer

class Scoreboard:
    def __init__(self, frame, RED_ROI_TOP_LEFT = (), RED_ROI_BOTTOM_RIGHT = ()):
        self.reader = easyocr.Reader(['en'], gpu=True)
        self.frame = frame

        if len(RED_ROI_TOP_LEFT) == 0:
            height = frame.shape[0]
            width = frame.shape[1]
            HEIGHT_CONSTANT = height / 720
            WIDTH_CONSTANT = width / 1280
            center_x = int(640 * WIDTH_CONSTANT)
            center_y = int(595 * HEIGHT_CONSTANT)
            x_offset = int(510 * WIDTH_CONSTANT)
            y_offset = int(30 * HEIGHT_CONSTANT)
            x_mid_offset = int(90 * WIDTH_CONSTANT)

            RED_ROI_TOP_LEFT = (center_x - x_offset, center_y - y_offset)
            RED_ROI_BOTTOM_RIGHT = (center_x - x_mid_offset, center_y + y_offset)

            print(RED_ROI_TOP_LEFT, RED_ROI_BOTTOM_RIGHT)

        self.RED_ROI_TOP_LEFT = RED_ROI_TOP_LEFT
        self.RED_ROI_BOTTOM_RIGHT = RED_ROI_BOTTOM_RIGHT
        self.mirrorGreenToRed()

        print(self.GREEN_ROI_TOP_LEFT, self.GREEN_ROI_BOTTOM_RIGHT)

        self.red_fencer = Fencer()
        self.green_fencer = Fencer()

        self.red_roi = self.getFrame(self.RED_ROI_TOP_LEFT, self.RED_ROI_BOTTOM_RIGHT)
        self.green_roi = self.getFrame(self.GREEN_ROI_TOP_LEFT, self.GREEN_ROI_BOTTOM_RIGHT)

    def shiftedROI(self, key, ROI):
        shift_map = {
            'w': (0, -1),
            'a': (-1, 0),
            's': (0, 1),
            'd': (1, 0)
        }

        if key in shift_map:
            return (ROI[0] + shift_map[key][0] * self.ROI_SHIFT, ROI[1] + shift_map[key][1] * self.ROI_SHIFT)
        return ROI

    def mirrorPosition(self, x):
        return self.frame.shape[1] - x

    def mirrorGreenToRed(self):
        self.GREEN_ROI_TOP_LEFT = (self.mirrorPosition(self.RED_ROI_BOTTOM_RIGHT[0]), self.RED_ROI_TOP_LEFT[1])
        self.GREEN_ROI_BOTTOM_RIGHT = (self.mirrorPosition(self.RED_ROI_TOP_LEFT[0]), self.RED_ROI_BOTTOM_RIGHT[1])

    def editScoreboard(self):
        self.ROI_SHIFT = 5

        while True:
            frame = self.frame.copy()
            cv2.rectangle(frame, self.RED_ROI_TOP_LEFT, self.RED_ROI_BOTTOM_RIGHT, (0, 0, 255), 2)
            cv2.rectangle(frame, self.GREEN_ROI_TOP_LEFT, self.GREEN_ROI_BOTTOM_RIGHT, (0, 255, 0), 2)

            cv2.imshow("Edit Scoreboard", frame)
            key = chr(cv2.waitKey(0) & 0xFF)

            # enter key
            if ord(key) == 13:
                cv2.destroyWindow("Edit Scoreboard")
                return None

            if key == 'W':
                self.RED_ROI_TOP_LEFT = self.shiftedROI(key.lower(), self.RED_ROI_TOP_LEFT)
            elif key == 'A':
                self.RED_ROI_TOP_LEFT = self.shiftedROI(key.lower(), self.RED_ROI_TOP_LEFT)
            elif key == 'S':
                self.RED_ROI_TOP_LEFT = self.shiftedROI(key.lower(), self.RED_ROI_TOP_LEFT)
            elif key == 'D':
                self.RED_ROI_TOP_LEFT = self.shiftedROI(key.lower(), self.RED_ROI_TOP_LEFT)
            else:
                self.RED_ROI_TOP_LEFT = self.shiftedROI(key, self.RED_ROI_TOP_LEFT)
                self.RED_ROI_BOTTOM_RIGHT = self.shiftedROI(key, self.RED_ROI_BOTTOM_RIGHT)

            self.mirrorGreenToRed()

            print(f"Key pressed: {key}")

    def getFrame(self, ROI_TOP_LEFT, ROI_BOTTOM_RIGHT):
        scoreboard_roi = self.frame[ROI_TOP_LEFT[1]:ROI_BOTTOM_RIGHT[1], ROI_TOP_LEFT[0]:ROI_BOTTOM_RIGHT[0]]
        gray_scoreboard = cv2.cvtColor(scoreboard_roi, cv2.COLOR_BGR2GRAY)

        _, thresh_scoreboard = cv2.threshold(gray_scoreboard, 150, 255, cv2.THRESH_BINARY_INV)
        resized_scoreboard = cv2.resize(thresh_scoreboard, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        return resized_scoreboard

    def readText(self, roi):
        results = self.reader.readtext(roi)
        return results

    def printText(self, roi):
        results = self.readText(roi)

        for (bbox, text, prob) in results:
            print(f"Detected Text: {text} (Confidence: {prob:.2f})")

    def __getScore(self, roi, SCORE_ROI_TOP_LEFT, SCORE_ROI_BOTTOM_RIGHT):
        score_roi = roi[SCORE_ROI_TOP_LEFT[1]:SCORE_ROI_BOTTOM_RIGHT[1],
                    SCORE_ROI_TOP_LEFT[0]:SCORE_ROI_BOTTOM_RIGHT[0]]
        cv2.imshow("Score calc", score_roi)

        score_reader = self.readText(score_roi)

        for (bbox, text, prob) in score_reader:
            if isnumeric(text):
                return int(text)

        return None

    def getScore(self, roi):
        # 120 x 840
        height = roi.shape[0]
        width = roi.shape[1]
        HEIGHT_CONSTANT = height / 120
        WIDTH_CONSTANT = width / 840

        len = int(30 * HEIGHT_CONSTANT)
        mid_y = int(60 * HEIGHT_CONSTANT)
        mid_x = int(420 * WIDTH_CONSTANT)
        x_offset = int(360 * WIDTH_CONSTANT)

        RED_SCORE_ROI_TOP_LEFT = (mid_x - x_offset - len, mid_y - len)
        RED_SCORE_ROI_BOTTOM_RIGHT = (mid_x - x_offset + len, mid_y + len)
        score = self.__getScore(roi, RED_SCORE_ROI_TOP_LEFT, RED_SCORE_ROI_BOTTOM_RIGHT)
        if score != None:
            return score

        GREEN_SCORE_ROI_TOP_LEFT = (mid_x + x_offset - len, mid_y - len)
        GREEN_SCORE_ROI_BOTTOM_RIGHT = (mid_x + x_offset + len, mid_y + len)
        score = self.__getScore(roi, GREEN_SCORE_ROI_TOP_LEFT, GREEN_SCORE_ROI_BOTTOM_RIGHT)

        return score

    def getName(self, roi):
        results = self.readText(roi)
        for (bbox, text, prob) in results:
            text_split = text.split()
            if len(text_split) != 2:
                continue

            (first_name, last_name) = text_split

            if not isupper(first_name[0]) or not islower(first_name[1:]):
                continue

            if not isupper(last_name):
                continue

            return text

    def getCountry(self, roi):
        results = self.readText(roi)
        for (bbox, text, prob) in results:
            if len(text) == 3 and isupper(text):
                return text

    def updateFencerInfo(self, fencer, roi):
        fencer.name = self.getName(roi)
        fencer.country = self.getCountry(roi)
        fencer.score = self.getScore(roi)

    def updateGreenFencerInfo(self):
        self.updateFencerInfo(self.green_fencer, self.green_roi)

    def updateRedFencerInfo(self):
        self.updateFencerInfo(self.red_fencer, self.red_roi)

    def printScoreboard(self):
        print(self.red_fencer.country, self.red_fencer.name, ":", self.green_fencer.country, self.green_fencer.name)
        print(self.getScore(self.red_roi), ":", self.getScore(self.green_roi))