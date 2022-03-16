import numpy as np
import sys
import PIL
from PIL import Image
from PIL import  ImageFile, ImageDraw, ImageChops, ImageFilter,ImageGrab,ImageEnhance,ImageFont 
import cv2
import regex as re
import pytesseract
import time
import os
#pytesseract.pytesseract.tesseract_cmd =r'/usr/local/Cellar/tesseract/4.0.0_1/bin/tesseract'
pytesseract.pytesseract.tesseract_cmd =r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
gameTime =np.load('xLims.npy')
deathlabels = np.load('label.npy')
deathlabels = np.swapaxes(deathlabels,0,1)


health = np.load('health.npy')
health = np.swapaxes(health,0,1)


lifeData = np.load('lifeData.npy')
lifeData = np.swapaxes(lifeData,0,1)

predictions =np.load('hero.npy')
predictions = np.swapaxes(predictions,0,1)
print(min(gameTime))

basePath = "C:\\Users\\Daniel\\Dropbox\\York\\death_prediction\\speedOImages\\"

dialOriginal = Image.open(basePath+'needle1.png').convert('RGBA')
gaugeOriginal = Image.open(basePath+'gauge.png').convert('RGBA')

sizeMod =16
gaugeOriginal = gaugeOriginal.resize((1650//sizeMod, 856//sizeMod), Image.ANTIALIAS)
dialOriginal = dialOriginal.resize((1650//sizeMod, 856//sizeMod), Image.ANTIALIAS)


x = 1650//(sizeMod*2)
y = 1650//(sizeMod*2)
loc = (x, y)



def get_index_at_time(time):
    
    if len(time) <4:
        return -1
    #if the time contains letters - wrong format
    if re.search('[a-zA-Z]', time):
        return -1
        
    m, s = time.split(':')
    #if the seconds has 3 values - wrong format
    if len(s) !=2:
        return -1

    if len(m) !=1 and len(m) !=2:
        return -1

    time = int(m) * 60 + int(s)
    
    n = np.argmax(gameTime>=time)
    return n

def show_meter_for_player(player,precition,lifeData):
    if lifeData>=1:
        percent = precition#player/9
    else:
        percent = 0
        
    rotation = 180 * percent  # 180 degrees because the gauge is half a circle
    rotation = 90 - rotation  # Factor in the needle graphic pointing to 50 (90 degrees)
    
    dial = dialOriginal.copy()#Image.open(basePath+'needle.png')
    gauge = gaugeOriginal.copy()#Image.open(basePath+'gauge.png')
    
    #if percent >0.8:
    data = np.array(gauge)   # "data" is a height x width x 4 numpy array
    red, green, blue, alpha = data.T # Temporarily unpack the bands for readability



    if lifeData >=1:
        # Replace white with red... (leaves alpha values alone...)
        white_areas = (red > 200) & (blue > 200) & (green > 200)
        data[..., :-1][white_areas.T] = (255, 255-(percent*percent)*255, 255-(percent*percent)*255) # Transpose back needed
        gauge = Image.fromarray(data)

        
    dial = dial.rotate(rotation, resample=PIL.Image.BICUBIC, center=loc)  # Rotate needle
    gauge.paste(dial, mask=dial)  # Paste needle onto gauge
    
    if lifeData < 1:
        gauge = np.array(gauge)
        gauge = cv2.blur(gauge,(9,9))
        
        gauge = Image.fromarray(gauge)
        gauge = gauge.point(lambda p: p * 0.3)      

   
    #print(np.array(gauge).shape)#107 206
    height,width,a = np.array(gauge).shape
   


    playersNames = ['j4','MiLAM','Emperor','33','Bamboe',
                    'Fly','Resolut1on','JerAx','s4','N0tail']

    background = Image.new('RGBA', (width, height+30), (255,255,255,255))
    background.paste(gauge, (0,0))

    fontSize=20
    draw = ImageDraw.Draw(background)  
    font = ImageFont.truetype("C:\\Windows\\WinSxS\\amd64_microsoft-windows-font-truetype-yugothic_31bf3856ad364e35_10.0.17134.1_none_be6df5bb828d2629\\YuGothR.ttc", fontSize)
    
    w, h = draw.textsize(playersNames[player], font=font)
    W = width

    draw.text(((W-w)/2, height),playersNames[player],(0,0,0),font=font)
    return background



def stack_images_to_vis(individualSpeedo,player,time):
    images1 = individualSpeedo[0:5]
    images1 = np.hstack(images1)
    
    images2 = individualSpeedo[5:10]       
    images2 = np.hstack(images2)
       
    imageAsNP = np.hstack((images1,images2))   

    return imageAsNP

def get_time_from_game():    

    im = ImageGrab.grab(bbox=(939,24,980,37))
    
    basewidth=200
    wpercent = (basewidth/float(im.size[0]))
    hsize = int((float(im.size[1])*float(wpercent)))
    im = im.resize((basewidth,hsize), Image.ANTIALIAS)#ANTIALIAS
    
    im = im.filter(ImageFilter.MedianFilter())
    enhancer = ImageEnhance.Contrast(im)
    im = enhancer.enhance(1.5)
    im = im.convert('L')

    im = im.point(lambda x: 0 if x<190 else 255, '1')#180   
    text = pytesseract.image_to_string(im, lang='eng', config='--psm 10 --oem 3 -c tessedit_char_whitelist=:0123456789')
    
    text = text.replace('-',':')
    return text
    

#every x seconds capture the screen and run get_sec

timeTillSync =0
index = 2940
nextTime =0
timeString = get_time_from_game()
index = get_index_at_time(timeString)


gameTimeEstimate = gameTime[index]
while True:
    while True:
        start_time = time.time()
        clear = lambda: os.system('cls')
        clear()


        index = np.argmax(gameTime>=gameTimeEstimate)


        if timeTillSync >1:
            #check if the screen grab is legit
            timeString = get_time_from_game()#'12:32'
            #if index is -1 then the format of the time is fucked - just wait till its right
            index = get_index_at_time(timeString)
            print(index)
            
            if index >=0:
                gameTimeEstimate = gameTime[index]
                timeTillSync =0
                break
            
        
        


        currentPrediction = predictions[index]
        currentTime = gameTime[index]
        if  index+1 <len(gameTime-1):
            nextTime = gameTime[index+1]

        currentLabel = deathlabels[index]
        currenthealth = health[index]
        currentLifeData = lifeData[index]

        print(index)
        print(time.strftime('%H:%M:%S', time.gmtime(currentTime)))
        print(currentLifeData)



        individualSpeedo = []
        for player in  range(len(currentPrediction)):

            img = show_meter_for_player(player,currentPrediction[player],currentLifeData[player])
            individualSpeedo.append(img)

        image = stack_images_to_vis(individualSpeedo,index,time.strftime('%H:%M:%S', time.gmtime(currentTime)))
        
        #image = np.swapaxes(image,0,1)
        cv2.imshow('image',image)
        cv2.waitKey(1)

        #take off the elapsed time from the waiting time
        elapsed_time = time.time() - start_time
        print(nextTime-currentTime  - elapsed_time )

        #if nextTime-currentTime  - elapsed_time>=0:
        #    time.sleep(nextTime-currentTime -elapsed_time )


        
        
        timeTillSync += elapsed_time
        gameTimeEstimate+= elapsed_time
        #maybe 5 seconds too late
        #death labvels dont last as long as htey should