import time
import cv2
import csv
import pygame
from pygame.locals import*
import traceback
import threading
import OpenGL
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL import shaders
import numpy as np
import keras
from keras.models import load_model
from statistics import mode
from Emotion.utils.datasets import get_labels
from Emotion.utils.inference import detect_faces
from Emotion.utils.inference import draw_text
from Emotion.utils.inference import draw_bounding_box
from Emotion.utils.inference import apply_offsets
from Emotion.utils.inference import load_detection_model
from Emotion.utils.preprocessor import preprocess_input

"""
Nina Boelsums
Final Bachelor Project: Spieg

This project shows the user their consumer profiles as assessed from their facial features,
and by counting the appearances of their phone's MAC addresses.
"""

# initiate computer vision models
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

ages = ['(0,2)','(4,6)','(8,12)','(15,20)','(25,32)','(38,43)','(48,53)','(60,100)']
genders = ['male', 'female']

impulseLikelyness = ['not', 'somewhat', 'quite', 'highly']
emotion_model_path = './Emotion/models/emotion_model.hdf5'
emotion_labels = get_labels('fer2013')
frame_window = 10
emotion_offsets = (20, 40)

face_cascade = cv2.CascadeClassifier('./Emotion/models/haarcascade_frontalface_default.xml')
emotion_classifier = load_model(emotion_model_path)
emotion_classifier._make_predict_function()
emotion_target_size = emotion_classifier.input_shape[1:3]
emotion_window = []

mood = 'neutral'
genderFrequency = [0,100]
ageFrequency = [0,0,0,0,100,0,0,0,0]
columnIndex = 0
rowIndex = 0
faceDetected = None

macObjects = []
start_time = time.time()
myphone = None


cap = cv2.VideoCapture(0)
cap.open(0)
if (not cap.isOpened()):
    print("ERROR! Unable to open camera")

width = 748
height = 1024
display = (width,height)
translations = [(0.0, 1.6, -2.5),(-1.3, 1.7, -2.2),(.7, -.9, -1.8),
                (-.8, -1.3, -2),(0.0, -2, -3),(-1.4, 1.0, -2.8),
                (-1.0, -2.0, -4),(1.1, 2.0, -4), (.5, -3.0, -5),(-2.5, 5.0, -6), (-1.5, -3.0, -15),(-2.5, 5.0, -12)]
textures = []
images = []
running = 1





image = pygame.image.load('./graphics/burger.png')

u_time = None

#################### OpenGL SHADERS ############################

vertex_shader = """#version 130
varying vec3 TexCoord0;
out vec2 texcoord;
uniform float time;

void main() {
    float displacement = cos(time/50000);
    
    
    vec4 position = gl_ModelViewMatrix * gl_Vertex;
    position.x = position.x + cos(time/500 + position.y * 2)/10;
    position.x = position.x + sin(time/1000 + position.z * 2)/10;
   
    position.y = position.y + sin(time/10000 + position.x * 2)/5;
    position.y = position.y + cos(time/5000 + position.z * 2)/5;

    position.z = position.z + sin(time/10000 + position.x * 2)/5;
    position.z = position.z + cos(time/5000 + position.y * 2)/5;
    //position = vec3(position.x, gl_Vertex.yz) + gl_Normal * sin(time/1000) * displacement ;
    
    

    texcoord = gl_MultiTexCoord0.xy;
    gl_Position = gl_ProjectionMatrix * position; //gl_Position moet er altijd bij
    //TexCoord0 = gl_Normal;
}"""

fragment_shader = """#version 130
uniform sampler2D Texture0;
out vec4 frag_color;
varying vec3 TexCoord0;
in vec2 texcoord;

void main() {
    frag_color = texture2D(Texture0, texcoord);
}"""

# Class for phone's MAC address    
class macAd:
    numVisits = 1
    
    def __init__(self, address, lastSeen, power):
        self.address = address
        self.lastSeen = lastSeen
        self.power = power
    
    def addMac():
        for obj in macObjects:
            if self.address is obj[0]:
                update(obj)
                return
        firstTime()
    
    def firstTime():
        macObjects.append(self)
        
    def update(self, new):
        ownLastDT = time.strptime(self.lastSeen, ' %Y-%m-%d %H:%M:%S')
        newLastDT = time.strptime(new.lastSeen, ' %Y-%m-%d %H:%M:%S')
        
        elapsed = (time.mktime(newLastDT) - time.mktime(ownLastDT))
        
        if elapsed > 30 : # Demoday setting, needs to be >290 in a real-life situation
            self.numVisits += 1
            
        self.lastSeen = new.lastSeen
        self.power = new.power


# Collect nearby MAC addresses from csvfile created with airodump-ng
def get_mac():    
    bottom = False
    macObj = None

    with open('nearby_devices-01.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        
        for row in reader:
            
            if 01:AB:CD:2E:AB:34 in row: # Should be MAC adress of target device
                macObj = macAd(row[0], row[2], row[3])
    return macObj
 

# Initialize age and gender models
def initialize_caffe_models():
    print('loading models')
    age_net = cv2.dnn.readNetFromCaffe(
        "/home/niinja/age_gender/AGmodel/deploy_age.prototxt",
        "/home/niinja/age_gender/AGmodel/age_net.caffemodel")
    gender_net = cv2.dnn.readNetFromCaffe(
        "/home/niinja/age_gender/AGmodel/deploy_gender.prototxt",
        "/home/niinja/age_gender/AGmodel/gender_net.caffemodel")
    
    return (age_net, gender_net)
    
    
# Recognition of Age, Gender and mood from a video stream
def capture_loop(age_net, gender_net):
    global mood
    global genderFrequency
    global ageFrequency
    global faceDetected
    
    face_cascade = cv2.CascadeClassifier('/usr/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml')    
    myphoneNew = get_mac()
    
    myphone.update(myphoneNew)
    
    start_time = time.time()
    
    while True:
        myphoneNew = get_mac()
        
        if myphoneNew is not None:
            myphone.update(myphoneNew)
        
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        
        maxarea = 0
        x = 0
        y = 0
        w = 0
        h = 0
        
        # Analyze the largest face of the recognized faces
        for (_x, _y, _w, _h) in faces:
            if _w * _h > maxarea:
                x = _x
                y = _y
                w = _w
                h = _h
                maxarea = w * h
            
            if maxarea > 0:
                faceDetected = True
                cv2.rectangle(frame, (x,y), (x+w,y+h),(255,255,0),2)
                face_img = frame[y:y+h, x:x+w].copy()
                blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                          
                
                gender_net.setInput(blob)
                gender_preds = gender_net.forward()
                gi = gender_preds[0].argmax()
                genderFrequency[gi] += 1

                age_net.setInput(blob)
                age_preds = age_net.forward()
                ai = age_preds[0].argmax()
                ageFrequency[ai] += 1

                gray_face = gray[y:y + h, x:x+w]
                try:
                    emotions(gray_face)
                except:
                    continue
        if len(faces) == 0:
            faceDetected = False
    
# Function for recognizing emotions in detected face
def emotions(gray_face):
    global mood
    gray_face = cv2.resize(gray_face, (emotion_target_size))
    gray_face = preprocess_input(gray_face, True)
    gray_face = np.expand_dims(gray_face, 0)
    gray_face = np.expand_dims(gray_face, -1)
    emotion_prediction = emotion_classifier.predict(gray_face)
    emotion_probability = np.max(emotion_prediction)
    emotion_label_arg = np.argmax(emotion_prediction)
    emotion_text = emotion_labels[emotion_label_arg]
    emotion_window.append(emotion_text)

    if len(emotion_window) > frame_window:
        emotion_window.pop(0)

    emotion_mode = mode(emotion_window)

    if emotion_text == 'angry':
        color = emotion_probability * np.asarray((255, 0, 0))
    elif emotion_text == 'sad':
        color = emotion_probability * np.asarray((0, 0, 255))
    elif emotion_text == 'happy':
        color = emotion_probability * np.asarray((255, 255, 0))
    elif emotion_text == 'surprise':
        color = emotion_probability * np.asarray((0, 255, 255))
    else:
        color = emotion_probability * np.asarray((0, 255, 0))

    color = color.astype(int)
    color = color.tolist()

    mood = emotion_mode
    
######################## DISPLAY FUNCTIONS ##################################
    
def monitor():
    global genderFrequency
    global ageFrequency
    global columnIndex
    global rowIndex
    
    angle = 0
    
    startTime = int(time.time())
    lastUpdateTime = 0 
    textTexid = 0
    screen = pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
    pygame.init()
    loadProductTextures()
    gl()
    
    while True:
        elapsedTime = int(time.time()) - startTime
        runtime = pygame.time.get_ticks()        
        glUniform1f(u_time, runtime)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        
        if elapsedTime % 3 == 0 and time.time() - lastUpdateTime > 1 :
            lastUpdateTime = time.time()
            rowIndex = genderFrequency.index(max(genderFrequency))
            columnIndex = ageFrequency.index(max(ageFrequency))
            textureData = text(ages[columnIndex], genders[rowIndex],mood, myphone.numVisits)
            
            textTexid = loadTextures(None, textureData)

            genderFrequency = [0,100]
            ageFrequency = [0,0,0,0,100,0,0,0,0]
        if faceDetected:    
            for count, i in enumerate(textures[columnIndex][rowIndex]):
                wobble(count, angle, i)

            for count, j in enumerate(textures[columnIndex][2], len(textures[columnIndex][rowIndex]) +1) :
                wobble(count, angle, j)
                
            displaytext(textTexid)    
            pygame.display.flip()
    shaders.glUseProgram( 0 )
    
# Load textures of product images to project on bubbles        
def loadProductTextures():
    with open('biasimages.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='"')
        
        for countr, row in enumerate(reader):
            textures.append([])
            
            for countc, column in enumerate(row):
                textures[countr].append([])
                images = column.split(",")
                
                for i in images:
                    try:
                        image = pygame.image.load("./graphics/"+i)  
                    except:
                        continue
                    textureData = pygame.image.tostring(image, "RGBA", 1)
                    texid = loadTextures(image, textureData)
                    
                    textures[countr][countc].append(texid)

# OpenGL setup
def gl():
    global r
    global running
    global vertex_shader
    global fragment_shader
    global quadratic
    global u_time
      
    quadratic = gluNewQuadric()
    gluQuadricNormals(quadratic, GLU_SMOOTH)
    gluQuadricTexture(quadratic, GL_TRUE)		

    
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(90.0, width/float(height), 1.0, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glEnable(GL_DEPTH_TEST)
    
    glTranslatef(0.0,0.0,-5)
    
    vert = shaders.compileShader(vertex_shader, GL_VERTEX_SHADER)
    frag = shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER)
    
    shader = shaders.compileProgram(vert,frag)
    u_time = glGetUniformLocation(shader, 'time')
    shaders.glUseProgram(shader)

# Create image containing information on detected categories
def text(avgAge, avgGender, mood, numVisits):

    startHeight = 100
    textScreen = pygame.Surface((width, width))
    white = (255, 255, 255)
    pygame.display.set_caption(" ")
    myfont = pygame.font.SysFont("Comic Sans MS", 65)
    mysmallfont = pygame.font.SysFont("Comic Sans MS", 30)

    genderLabel = myfont.render("gender: " + avgGender, 1, white)
    textScreen.blit(genderLabel, (100, startHeight))
    ageLabel = myfont.render("age: " + avgAge, 1, white)
    textScreen.blit(ageLabel, (100, startHeight+50))
    moodLabel = myfont.render("mood: " + mood, 1, white)
    textScreen.blit(moodLabel, (100, startHeight+100))
    visitLabel = myfont.render("visited area " + str(numVisits) + " times",1,white)
    textScreen.blit(visitLabel, (100, startHeight+150))
    extraLabel = myfont.render("single, lower middle class", 1, white)
    textScreen.blit(extraLabel, (100, startHeight+230))
    classLabel = myfont.render("not religious, well-educated", 1, white)
    textScreen.blit(classLabel, (100, startHeight+280))
    return pygame.image.tostring(textScreen, "RGBA", 1)


# Create texture out of information image
def loadTextures(image, textureData):   
    if image:
        image_width = image.get_width()
        image_height = image.get_height()
        
    else:
        image_width = width
        image_height = width
        
    glEnable(GL_TEXTURE_2D)
    texid = glGenTextures(1)

    glBindTexture(GL_TEXTURE_2D, texid)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image_width, image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, textureData)

    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    
    return texid


# Creates a sphere wich will be used for the wobbly bubble
def wobble(i, angle, t):
    glLoadIdentity()
    glBindTexture(GL_TEXTURE_2D, t)
    glTranslate(*translations[i])
    glRotate(270, 1.0, 0.0, 0.0)
    glRotate(angle, 0.0, 1, 0.0)
    
    gluSphere(quadratic,.3,32,32)
    

# Create cube on which the information image can be projected
def displaytext(texid):
    glLoadIdentity()
    glBindTexture(GL_TEXTURE_2D, texid)
    glTranslate(0.0, 0.0, -4)
    glRotate(0, 1.0, 0.0, 0.0)
  
    glBegin(GL_QUADS)
    glTexCoord2f(0.0, 0.0)
    glVertex3f(-1.0, -1.0,  1.0)
    glTexCoord2f(1.0, 0.0)
    glVertex3f(1.0, -1.0,  1.0)
    glTexCoord2f(1.0, 1.0)
    glVertex3f(1.0,  1.0,  1.0)
    glTexCoord2f(0.0, 1.0)
    glVertex3f(-1.0,  1.0,  1.0)
    glTexCoord2f(1.0, 0.0)
    glVertex3f(-1.0, -1.0, -1.0)
    glTexCoord2f(1.0, 1.0)
    glVertex3f(-1.0,  1.0, -1.0)
    glTexCoord2f(0.0, 1.0)
    glVertex3f(1.0,  1.0, -1.0)
    glTexCoord2f(0.0, 0.0)
    glVertex3f(1.0, -1.0, -1.0)
    glTexCoord2f(0.0, 1.0)
    glVertex3f(-1.0,  1.0, -1.0)
    glTexCoord2f(0.0, 0.0)
    glVertex3f(-1.0,  1.0,  1.0)
    glTexCoord2f(1.0, 0.0)
    glVertex3f(1.0,  1.0,  1.0)
    glTexCoord2f(1.0, 1.0)
    glVertex3f(1.0,  1.0, -1.0)
    glTexCoord2f(1.0, 1.0)
    glVertex3f(-1.0, -1.0, -1.0)
    glTexCoord2f(0.0, 1.0)
    glVertex3f(1.0, -1.0, -1.0)
    glTexCoord2f(0.0, 0.0)
    glVertex3f(1.0, -1.0,  1.0)
    glTexCoord2f(1.0, 0.0)
    glVertex3f(-1.0, -1.0,  1.0)
    glTexCoord2f(1.0, 0.0)
    glVertex3f(1.0, -1.0, -1.0)
    glTexCoord2f(1.0, 1.0)
    glVertex3f(1.0,  1.0, -1.0)
    glTexCoord2f(0.0, 1.0)
    glVertex3f(1.0,  1.0,  1.0)
    glTexCoord2f(0.0, 0.0)
    glVertex3f(1.0, -1.0,  1.0)
    glTexCoord2f(0.0, 0.0)
    glVertex3f(-1.0, -1.0, -1.0)
    glTexCoord2f(1.0, 0.0)
    glVertex3f(-1.0, -1.0,  1.0)
    glTexCoord2f(1.0, 1.0)
    glVertex3f(-1.0,  1.0,  1.0)
    glTexCoord2f(0.0, 1.0)
    glVertex3f(-1.0,  1.0, -1.0)
    glEnd()

        
if __name__ == '__main__':
    
    myphone = get_mac()
    age_net, gender_net = initialize_caffe_models()
    
    opengl_thread = threading.Thread(target = monitor)
    webcam_thread = threading.Thread(target = capture_loop, args = (age_net, gender_net))
    
    opengl_thread.start()
    webcam_thread.start()

