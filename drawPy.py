import cv2
import mediapipe as mp

class handTracker():

    counter = 0
    center = []
    permission = False
    thinkness = -1
    colorRec = (255, 0, 0)
    colorRecChange = (2,76,252)
    colorText = (255, 255, 255)
    textRecPen = "Pen"
    textRecEraser = "Eraser"

    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, modelCompexity=1, trackCon=0.5, addressCam=0):
        self.addressCam = addressCam
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelCompexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            self.mode, self.maxHands, self.modelComplex, self.detectionCon, self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def handsFinder(self, image, draw=True):
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)
        if self.result.multi_hand_landmarks:
            for handLms in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
        return image

    def positionFinder(self, image, handNO=0, draw=True):
        lmlist= []
        if self.result.multi_hand_landmarks:
            Hand = self.result.multi_hand_landmarks[handNO]
            for id, lm in enumerate(Hand.landmark):
                h, w, c= image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
            if draw:
                id, cx, cy = lmlist[8]
                cv2.circle(image, (cx, cy), 12, (0, 255, 255), cv2.FILLED)
        return lmlist

    def getAddressCam(self, address):
        self.addressCam = address

    def showRectangle(self, image):
        y, x, c = image.shape
        startRecPen = (x - 100, 10)
        endRecPen = (x - 10, 100)
        orgPen = (x - 80, 65)
        startRecEraser = (x - 100, 110)
        endRecEraser = (x - 10, 200)
        orgEraser = (x - 95, 163)

        cv2.rectangle(image, startRecPen, endRecPen, self.colorRec, thickness=self.thinkness)
        cv2.putText(image, self.textRecPen, orgPen, cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colorText, thickness=2)

        cv2.rectangle(image, startRecEraser, endRecEraser, self.colorRec, thickness=self.thinkness)
        cv2.putText(image, self.textRecEraser, orgEraser, cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colorText, thickness=2)

        return startRecPen, endRecPen, startRecEraser, endRecEraser, orgPen, orgEraser

    def checkPen(self, image, lmlist, startPointX, endPointX, startPointY, endPointY, orgPen):
        try:
            id, cx, cy = lmlist[8]
            if cx >= startPointX and cx <= endPointX and cy <= endPointY and cy >= startPointY:
                cv2.rectangle(image, (startPointX, startPointY), (endPointX, endPointY), self.colorRecChange, thickness=self.thinkness)
                cv2.putText(image, self.textRecPen, orgPen, cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colorText, thickness=2)
                self.permission = True
        except:
            pass

    def checkEraser(self, image, lmlist, startPointX, endPointX, startPointY, endPointY, orgEraser):
        try:
            id, cx, cy = lmlist[8]
            if cx >= startPointX and cx <= endPointX and cy <= endPointY and cy >= startPointY:
                cv2.rectangle(image, (startPointX, startPointY), (endPointX, endPointY), self.colorRecChange, thickness=self.thinkness)
                cv2.putText(image, self.textRecEraser, orgEraser, cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colorText, thickness=2)
                self.permission = False
                self.center = []
        except:
            pass

    def usePen(self, image, lmlist):
        try:
            idIFT, cxIFT, cyIFT = lmlist[8]
            idMFT, cxMFT, cyMFT = lmlist[12]
            if cxIFT > cxMFT:
                diffX = cxIFT - cxMFT
            else:
                diffX = cxMFT - cxIFT

            if cyIFT > cyMFT:
                diffY = cyIFT - cyMFT
            else:
                diffY = cyMFT - cyIFT
            status = (diffX < 30 and diffY < 20)
            if status:
                self.counter += 1
                self.center.append(((cxMFT + cxIFT) // 2, (cyMFT + cyIFT) // 2))
        except:
            pass

    def showDraw(self, image):
        try:
            for i in range(self.counter):
                cv2.circle(image, self.center[i], 10, (0, 0, 0), thickness=-1)
        except:
            pass

    def run(self):
        cap = cv2.VideoCapture(self.addressCam)
        tracker = handTracker()
        while True:
            success, image = cap.read()
            imageCopy = image.copy()
            image = tracker.handsFinder(image)
            lmList = tracker.positionFinder(image)

            startPenPoint, endPenPoint, startEraserPoint, endEraserPoint, orgPen, orgEraser = self.showRectangle(image)

            self.checkPen(image, lmList, startPenPoint[0], endPenPoint[0], startPenPoint[1], endPenPoint[1], orgPen)
            self.checkEraser(image, lmList, startEraserPoint[0], endEraserPoint[0], startEraserPoint[1], endEraserPoint[1], orgEraser)
            if self.permission:
                self.usePen(image, lmList)

            self.showDraw(image)

            imageWithBitwiseAnd = cv2.bitwise_and(imageCopy, image)

            cv2.imshow("Video", imageWithBitwiseAnd)
            cv2.waitKey(1)
            if (cv2.waitKey(1) & 0xFF == ord('q')) or (cv2.waitKey(1) & 0xFF == ord('Q')):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    testProgram = handTracker()
    testProgram.run()