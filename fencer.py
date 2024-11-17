class Fencer:
    def __init__(self, name = "", country = ""):
        self.name = name
        self.country = country

        self.score = None

    def setROI(self, ROI_TOP_LEFT, ROI_BOTTOM_RIGHT):
        self.ROI_TOP_LEFT = ROI_TOP_LEFT
        self.ROI_BOTTOM_RIGHT = ROI_BOTTOM_RIGHT

    def printInfo(self):
        print("Name:", self.name)
        print("Country:", self.country)
        print("Score:", self.score)