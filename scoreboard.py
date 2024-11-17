import cv2
import easyocr
from numpy.core.defchararray import isnumeric, isupper, islower

from fencer import Fencer

class Scoreboard:
    # scoreboard dimensions are 120x840

    def __init__(self, frame):
        center_x = 640
        center_y = 595
        x_offset = 510
        y_offset = 30
        x_mid_offset = 90

        self.red_fencer = Fencer()
        self.green_fencer = Fencer()

        self.RED_ROI_TOP_LEFT = (center_x - x_offset, center_y - y_offset)
        self.RED_ROI_BOTTOM_RIGHT = (center_x - x_mid_offset, center_y + y_offset)
        self.GREEN_ROI_TOP_LEFT = (center_x + x_mid_offset, center_y - y_offset)
        self.GREEN_ROI_BOTTOM_RIGHT = (center_x + x_offset, center_y + y_offset)

        self.reader = easyocr.Reader(['en'], gpu=True)
        self.frame = frame

        self.green_roi = self.getFrame(self.GREEN_ROI_TOP_LEFT, self.GREEN_ROI_BOTTOM_RIGHT)
        self.red_roi = self.getFrame(self.RED_ROI_TOP_LEFT, self.RED_ROI_BOTTOM_RIGHT)

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
        cv2.imshow("Score ROI", score_roi)
        cv2.waitKey(500)

        score_reader = self.readText(score_roi)

        for (bbox, text, prob) in score_reader:
            if isnumeric(text):
                return int(text)

        return None

    def getScore(self, roi):
        # 120 x 840

        len = 30
        mid_y = 60
        mid_x = 420
        x_offset = 360

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