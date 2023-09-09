import speech_recognition as sr
import os
import time 
import pywhatkit
import pyautogui
import numpy as np
import matplotlib.pyplot as plt
from twilio.rest import Client
import cv2
from pynput.keyboard import Key, Controller
from geopy.geocoders import Nominatim
keyboard = Controller()
from PIL import Image, ImageDraw
from googlesearch import search
import boto3
import tkinter as tk
from cvzone.HandTrackingModule import HandDetector
import random
import pyttsx3
from langchain.llms import OpenAI
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from cvzone.HandTrackingModule import HandDetector
import pandas
from sklearn.linear_model import LinearRegression
import webbrowser
from langchain.document_loaders import TextLoader
import threading
import webview

def load_webpage():
    webview.load_url(http://3.111.245.114/hometeam.html')
def ec2_finger():
    
    def genOS():
        ec2=boto3.resource('ec2')
        instances= ec2.create_instances(MinCount=1, MaxCount=1, InstanceType="t2.micro", ImageId="ami-0ded8326293d3201b", SecurityGroupIds=['sg-0c7043809b8957ebd'])
        return instances[0].id

    def delOS(id):
        ec2=boto3.resource('ec2')
        ec2.instances.filter(InstanceIds=[id]).terminate()

    detector = HandDetector(maxHands=1 , detectionCon=0.8 )
    allOS=[]
    cap = cv2.VideoCapture(0)

    while True:
        ret,  photo = cap.read()
        hand = detector.findHands(photo , draw=False)
        if hand:
            detectHand = hand[0]
            if detectHand:
                fingerup = detector.fingersUp(detectHand)
                if detectHand['type'] == 'Left':
                    for i in fingerup:
                        if i==1:
                            allOS.append(genOS())

                else:
                    for i in fingerup:
                        if i==1:
                            delOS(allOS.pop())

        cv2.imshow("my photo", photo)
        if cv2.waitKey(10) == 13:
            break

    cv2.destroyAllWindows()
    cap.release()
    
def linearReg():
    
    dataset = pandas.read_csv("book1.csv")
    model = LinearRegression()
    y = dataset['marks']
    x = dataset['hrs']
    X = x.values.reshape(-1,1)
    model.fit(X,y)
    print("model prediction : ")
    print(model.predict([[3]]))
    print("model Coefficient : ")
    print(model.coef_)
    
def assistant():
    
    def speak(text):
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()

    def recognize_speech():
        recognizer = sr.Recognizer()

        with sr.Microphone() as source:
            print("Listening...")
            audio = recognizer.listen(source)

        try:
            print("Recognizing...")
            text = recognizer.recognize_google(audio)
            print("You said:", text)
            return text.lower()
        except sr.UnknownValueError:
            print("Sorry, could not understand audio.")
        except sr.RequestError as e:
            print(f"Error with the speech recognition service; {e}")

        return None

    def open_whatsapp():
        speak("Opening WhatsApp.")
        webbrowser.open("https://web.whatsapp.com")
        time.sleep(15)  # Wait for 15 seconds to give you time to scan the QR code
        speak("WhatsApp is now open. You can use it on your browser.")

    if __name__ == "__main__":
        while True:
            recognized_text = recognize_speech()
            if recognized_text:
                if "google" in recognized_text:
                    speak("Opening Google.")
                    webbrowser.open("https://www.google.com")
                elif "youtube" in recognized_text:
                    speak("Opening YouTube.")
                    webbrowser.open("https://www.youtube.com")
                    # Wait for a moment before searching for music
                    speak("What music would you like to listen to?")
                    time.sleep(3)  # Wait for 3 seconds to give you time to respond
                    music_name = recognize_speech()
                    if music_name:
                        url = f"https://www.youtube.com/results?search_query={music_name}"
                        webbrowser.open(url)
                        time.sleep(5)  # Wait for the search results page to load
                        # Click on the first video link
                        try:
                            pyautogui.click(x=800, y=380)  # Adjust the coordinates as per your screen resolution
                        except pyautogui.FailSafeException:
                            print("Failed to click the video link. Please click it manually.")

                elif "python" in recognized_text and "code" in recognized_text:
                    speak("Opening Chrome and searching Bill Gates.")
                    webbrowser.open("https://www.google.com/search?q=Bill+Gates")  # Changed the search query as per your requirement
                elif "vimal daga" in recognized_text:  # Added a new condition to directly search for "Bill Gates"
                    speak("Searching Bill Gates on Google.")
                    webbrowser.open("https://www.google.com/search?q=Bill+Gates")
                elif "whatsapp" in recognized_text:
                    open_whatsapp()
                elif "exit" in recognized_text or "stop" in recognized_text:
                    speak("Goodbye!")
                    break

def rekognition(): 
    client = boto3.client('rekognition',region_name='ap-south-1')
    with open("Lokendraa.jpeg",'rb') as imgFile:
        imgData=imgFile.read()
    response=client.detect_labels(Image={'Bytes':imgData},MaxLabels=8)
    response    
    labels= response["Labels"]
    labels
    for label in labels:
            print(f"Label: {label['Name']}, Confidence: {label['Confidence']:.2f}%")
            
def document_loader():
    loader = TextLoader(file_path="personal.txt")
    document = loader.load()
    from langchain.text_splitter import CharacterTextSplitter
    textChunk = CharacterTextSplitter(chunk_size=250)
    texts = textChunk.split_documents(document)
    len(texts)
    myopenkey  = "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"
    from langchain.embeddings import OpenAIEmbeddings
    myembedmodel = OpenAIEmbeddings(openai_api_key=myopenkey)
    from langchain.vectorstores import Pinecone
    import pinecone
    pinecone.init(
            api_key=" #use api KEY ",
            environment="gcp-starter"
    )
    docsearch=Pinecone.from_documents(
                    documents = texts,
                    embedding = myembedmodel,
                    index_name = 'mylwspindex'        
    )
    from langchain.llms import OpenAI
    from langchain.chains import RetrievalQA
    qa =  RetrievalQA.from_chain_type(
            llm=OpenAI(openai_api_key= myopenkey),
            chain_type="stuff",
            retriever=docsearch.as_retriever()
    )
    myquery = "tell me about technology in 10 words"
    qa({"query": myquery}) 
    
def cartoon():

    def cartoonize_image(image, gray_mode=False):
        if gray_mode:
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(image, 9, 300, 300)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        return cartoon

    def cartoonize_video():

        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            cartoon_frame = cartoonize_image(frame)
            stacked_frames = np.hstack((frame, cartoon_frame))
            cv2.imshow("Cartoonizer", stacked_frames)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    if __name__ == "__main__":
        cartoonize_video()

from tkinter import *

def videoDownload():
    from pyyoutube import Api
    from pytube import YouTube
    from threading import Thread
    from tkinter import messagebox


    def get_list_videos():
        global playlist_item_by_id
        # Clear ListBox
        list_box.delete(0, 'end')

        # Create API Object
        api = Api(api_key='AIzaSyDSn9HX5SHGsiyNl_bFDtP1cHaSaI1h1h4')

        if "youtube" in playlistId.get():
            playlist_id = playlistId.get()[len(
                "https://www.youtube.com/playlist?list="):]
        else:
            playlist_id = playlistId.get()

        # Get list of video links
        playlist_item_by_id = api.get_playlist_items(
            playlist_id=playlist_id, count=None, return_json=True)

        # Iterate through all video links and insert into listbox
        for index, videoid in enumerate(playlist_item_by_id['items']):
            list_box.insert(
                END, f" {str(index+1)}. {videoid['contentDetails']['videoId']}")

        download_start.config(state=NORMAL)


    def threading():
        # Call download_videos function
        t1 = Thread(target=download_videos)
        t1.start()


    def download_videos():
        download_start.config(state="disabled")
        get_videos.config(state="disabled")

        # Iterate through all selected videos
        for i in list_box.curselection():
            videoid = playlist_item_by_id['items'][i]['contentDetails']['videoId']

            link = f"https://www.youtube.com/watch?v={videoid}"

            yt_obj = YouTube(link)

            filters = yt_obj.streams.filter(progressive=True, file_extension='mp4')

            # download the highest quality video
            filters.get_highest_resolution().download()

        messagebox.showinfo("Success", "Video Successfully downloaded")
        download_start.config(state="normal")
        get_videos.config(state="normal")


    # Create Object
    root = Tk()
    # Set geometry
    root.geometry('400x400')

    # Add Label
    Label(root, text="Youtube Playlist Downloader",
        font="italic 15 bold").pack(pady=10)
    Label(root, text="Enter Playlist URL:-", font="italic 10").pack()

    # Add Entry box
    playlistId = Entry(root, width=60)
    playlistId.pack(pady=5)

    # Add Button
    get_videos = Button(root, text="Get Videos", command=get_list_videos)
    get_videos.pack(pady=10)

    # Add Scrollbar
    scrollbar = Scrollbar(root)
    scrollbar.pack(side=RIGHT, fill=BOTH)
    list_box = Listbox(root, selectmode="multiple")
    list_box.pack(expand=YES, fill="both")
    list_box.config(yscrollcommand=scrollbar.set)
    scrollbar.config(command=list_box.yview)

    download_start = Button(root, text="Download Start",
                            command=threading, state=DISABLED)
    download_start.pack(pady=10)

    # Execute Tkinter
    root.mainloop()
    
def pomodoro():
    import tkinter as tk
    import time

    # Create the main application window
    root = tk.Tk()
    root.title("Pomodoro Timer")
    root.geometry("300x200")
    root.configure(bg="#f0f0f0")

    # Initialize pomodoro_active flag
    pomodoro_active = False

    # Define Pomodoro functions
    def start_pomodoro():
        work_time = 25 * 60
        short_break_time = 5 * 60
        long_break_time = 15 * 60
        num_work_sessions = 4

        global pomodoro_active
        pomodoro_active = True

        while pomodoro_active and num_work_sessions > 0:
            countdown(work_time, "Work Time")
            if pomodoro_active:
                countdown(short_break_time, "Short Break Time")
                num_work_sessions -= 1

        if pomodoro_active:
            countdown(long_break_time, "Long Break Time")

        pomodoro_active = False
        timer_label.config(text="Pomodoro Stopped", fg="red")

    def stop_pomodoro():
        global pomodoro_active
        pomodoro_active = False
        timer_label.config(text="Pomodoro Stopped", fg="red")

    def countdown(seconds, session_type):
        global pomodoro_active
        while seconds and pomodoro_active:
            mins, secs = divmod(seconds, 60)
            timer_label.config(text=f"{session_type}\n{mins:02d}:{secs:02d}", fg="black")
            root.update()
            time.sleep(1)
            seconds -= 1
        if pomodoro_active:
            timer_label.config(text="Session Complete!", fg="green")
            root.update()
            time.sleep(2)
            timer_label.config(text="")
            root.update()

    # Create and position the buttons
    button_pomodoro = tk.Button(root, text="Start Pomodoro", command=start_pomodoro, padx=10, pady=5, bg="#ff9800", fg="white")
    button_pomodoro.pack(pady=20)

    button_stop_pomodoro = tk.Button(root, text="Stop Pomodoro", command=stop_pomodoro, padx=10, pady=5, bg="#e91e63", fg="white")
    button_stop_pomodoro.pack(pady=10)

    timer_label = tk.Label(root, text="", font=("Helvetica", 20), bg="#f0f0f0")
    timer_label.pack()

    # Start the main event loop
    root.mainloop()
    
def TikTacToe():
    class TicTacToe:
        def __init__(self):
            self.root = tk.Tk()
            self.root.title("Tic Tac Toe")
            self.current_player = 'X'
            self.board = [''] * 9
            self.buttons = []

            for i in range(3):
                row = []
                for j in range(3):
                    button = tk.Button(self.root, text='', font=('normal', 20), width=10, height=3,
                                       command=lambda r=i, c=j: self.make_move(r, c))
                    button.grid(row=i, column=j)
                    row.append(button)
                self.buttons.append(row)

        def make_move(self, row, col):
            index = row * 3 + col
            if self.board[index] == '':
                self.board[index] = self.current_player
                self.buttons[row][col].config(text=self.current_player)
                if self.check_win():
                    self.game_over()
                elif all(cell != '' for cell in self.board):
                    self.game_over(draw=True)
                else:
                    self.current_player = 'O' if self.current_player == 'X' else 'X'

        def check_win(self):
            winning_combinations = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                                    (0, 3, 6), (1, 4, 7), (2, 5, 8),
                                    (0, 4, 8), (2, 4, 6)]
            for combo in winning_combinations:
                if self.board[combo[0]] == self.board[combo[1]] == self.board[combo[2]] != '':
                    return True
            return False

        def game_over(self, draw=False):
            if draw:
                message = "It's a draw!"
            else:
                message = f"Player {self.current_player} wins!"
            messagebox.showinfo("Game Over", message)
            self.root.destroy()

        def start(self):
            self.root.mainloop()

    if __name__ == "__main__":
        game = TicTacToe()
        game.start()

    
def open_software(software_name):
    software_path = {
        "notepad": "notepad.exe",
        "calculator": "calc.exe",
        "paint": "mspaint.exe",
        "chrome":"chrome.exe",
        "command prompt":"cmd.exe",
        "explorer":"explorer.exe",
        "vlc":"vlc.exe",
         "taskmgr":"taskmgr",
        # Add more software names and paths here
    }

    if software_name in software_path:
        try:
            os.startfile(software_path[software_name])
        except Exception as e:
            status_label.config(text=f"Error: {e}")
    else:
        status_label.config(text="Software not found.")
    pass

def whatsapp():
    from pynput.keyboard import Key, Controller
    keyboard = Controller()
    try:
        pywhatkit.sendwhatmsg_instantly(
            phone_no="+917073322741", 
            message="Hello Brother",
            tab_close=True
        )
        time.sleep(20)
        pyautogui.click()
        time.sleep(5)
        keyboard.press(Key.enter)
        keyboard.release(Key.enter)
        print("Message sent!")
    except Exception as e:
        print(str(e))
        
        
def message():
    class SMSSender:
        def __init__(self, root):
            self.root = root
            self.root.title("SMS Sender")

            self.account_sid = "AC2ad5c57917485651904bae4ad6e8e04e"
            self.auth_token = "efe83485fb61bac736982da0c038c0f1"
            self.twilio_phone_number = "+18506128792"

            self.label_to = tk.Label(root, text="To:")
            self.label_to.pack()

            self.entry_to = tk.Entry(root, width=30)
            self.entry_to.pack()

            self.label_message = tk.Label(root, text="Message:")
            self.label_message.pack()

            self.entry_message = tk.Text(root, width=30, height=5)
            self.entry_message.pack()

            self.button_send = tk.Button(root, text="Send SMS", command=self.send_sms)
            self.button_send.pack()

        def send_sms(self):
            to_number = self.entry_to.get()
            message = self.entry_message.get("1.0", tk.END).strip()

            if not to_number or not message:
                messagebox.showerror("Error", "Please enter recipient's number and message.")
                return

            try:
                client = Client(self.account_sid, self.auth_token)
                client.messages.create(
                    to=to_number,
                    from_=self.twilio_phone_number,
                    body=message
                )
                messagebox.showinfo("Success", "SMS sent successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to send SMS: {str(e)}")

        def run(self):
            self.root.mainloop()

    if __name__ == "__main__":
        root = tk.Tk()
        app = SMSSender(root)
        app.run()

def click_photo():
    cap=cv2.VideoCapture(0)
    cap
    status ,photo =cap.read()
    cv2.imwrite("pic.jpg",photo)
    cv2.imshow("My photo",photo)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
    cap.release()
    

def crop_pic():
    cap=cv2.VideoCapture(0)
    cap
    status ,photo =cap.read()
    cv2.imwrite("pic.jpg",photo)
    cv2.imshow("My photo",photo[200:540,200:430])
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
    cap.release()
    
def colorgame():    
    colours = ['Red','Blue','Green','Pink','Black',
            'Yellow','Orange','White','Purple','Brown']
    score = 0

    timeleft = 30

    def startGame(event):

        if timeleft == 30:

            countdown()

        nextColour()

    def nextColour():

        global score
        global timeleft

        if timeleft > 0:

            e.focus_set()

            if e.get().lower() == colours[1].lower():

                score += 1

            e.delete(0, tkinter.END)

            random.shuffle(colours)

            label.config(fg = str(colours[1]), text = str(colours[0]))

            scoreLabel.config(text = "Score: " + str(score))


    def countdown():

        global timeleft

        if timeleft > 0:

            timeleft -= 1

            timeLabel.config(text = "Time left: "
                                + str(timeleft))

            timeLabel.after(1000, countdown)


    root = tkinter.Tk()

    root.title("COLORGAME")

    # set the size
    root.geometry("375x200")

    # add an instructions label
    instructions = tkinter.Label(root, text = "Type in the colour"
                            "of the words, and not the word text!",
                                        font = ('Helvetica', 12))
    instructions.pack()

    # add a score label
    scoreLabel = tkinter.Label(root, text = "Press enter to start",
                                        font = ('Helvetica', 12))
    scoreLabel.pack()

    # add a time left label
    timeLabel = tkinter.Label(root, text = "Time left: " +
                str(timeleft), font = ('Helvetica', 12))

    timeLabel.pack()

    # add a label for displaying the colours
    label = tkinter.Label(root, font = ('Helvetica', 60))
    label.pack()

    e = tkinter.Entry(root)

    root.bind('<Return>', startGame)
    e.pack()

    # set focus on the entry box
    e.focus_set() 
    
def capture_video():
    cap=cv2.VideoCapture(0)
    while True:
        status ,photo=cap.read()
        cv2.imshow("My photo",photo)
        if cv2.waitKey(5)==13:
            break
    cv2.destroyAllWindows()

def capture_crop_video():
    cap=cv2.VideoCapture(0)
    while True:
        status ,photo=cap.read()
        photo[0:200,0:200]=photo[200:400,200:400]
        cv2.imshow("My photo",photo)
        if cv2.waitKey(5)==13:
            break
    cv2.destroyAllWindows()
    
def get_coordinates():
    location_name = input("enter the city name:")
    geolocator = Nominatim(user_agent="location_finder")
    location = geolocator.geocode(location_name)
    if location is None:
        print(f"Coordinates not found for '{location_name}'.")
        return None
    else:
        latitude = location.latitude
        longitude = location.longitude
        print(f"Coordinates for '{location_name}': Latitude = {latitude}, Longitude = {longitude}.")
        return latitude, longitude

    # Replace 'New York City' with your desired location.
    location_name = input("enter the city name:")
    

def top_10_google_searches():

    query = input("Enter what you want to search: ")
    result = int(input("How many results you want: "))

    for i in search(query, num=result, stop=result, pause=2):
        print(i)

import tkinter as tk

def create_drawing_app():
    class DrawingApp:
        def __init__(self, root):
            self.root = root
            self.root.title("Drawing App")
            
            self.canvas = tk.Canvas(root, width=800, height=600, bg="white")
            self.canvas.pack()
            
            self.canvas.bind("<Button-1>", self.start_drawing)
            self.canvas.bind("<B1-Motion>", self.draw)

            self.drawing = False
            self.prev_x = None
            self.prev_y = None

        def start_drawing(self, event):
            self.drawing = True
            self.prev_x = event.x
            self.prev_y = event.y

        def draw(self, event):
            if self.drawing and self.prev_x is not None and self.prev_y is not None:
                x, y = event.x, event.y
                self.canvas.create_line(self.prev_x, self.prev_y, x, y, fill="black", width=2)
                self.prev_x = x
                self.prev_y = y

        def stop_drawing(self, event):
            self.drawing = False
            self.prev_x = None
            self.prev_y = None

    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
    
def launch_instance():
    launch = boto3.client('ec2',region_name='ap-south-1')
    launch.run_instances(
        ImageId='ami-0da59f1af71ea4ad2',
        InstanceType='t2.micro',
        MaxCount=1,
        MinCount=1
        )
    describe_instance = boto3.client('ec2')
    describe_instance.describe_instances()
    
def create_bucket():
    bucket = boto3.client('s3',region_name='ap-south-1')
    bucket.create_bucket(
    Bucket='jaspreetbhagat1234567890',
    ACL='private',
    CreateBucketConfiguration={
          'LocationConstraint': 'ap-south-1'}
    )
    
def sendemail():
    class EmailSender:
        def __init__(self, root):
            self.root = root
            self.root.title("Email Sender")

            self.label_sender_email = tk.Label(root, text="Sender Email:")
            self.label_sender_email.pack()

            self.entry_sender_email = tk.Entry(root, width=30)
            self.entry_sender_email.pack()

            self.label_password = tk.Label(root, text="Password:")
            self.label_password.pack()

            self.entry_password = tk.Entry(root, width=30, show="*")
            self.entry_password.pack()

            self.label_receiver_email = tk.Label(root, text="Receiver Email:")
            self.label_receiver_email.pack()

            self.entry_receiver_email = tk.Entry(root, width=30)
            self.entry_receiver_email.pack()

            self.label_subject = tk.Label(root, text="Subject:")
            self.label_subject.pack()

            self.entry_subject = tk.Entry(root, width=30)
            self.entry_subject.pack()

            self.label_body = tk.Label(root, text="Body:")
            self.label_body.pack()

            self.entry_body = tk.Text(root, width=30, height=5)
            self.entry_body.pack()

            self.button_send = tk.Button(root, text="Send Email", command=self.send_email)
            self.button_send.pack()

        def send_email(self):
            sender_email = self.entry_sender_email.get()
            password = self.entry_password.get()
            receiver_email = self.entry_receiver_email.get()
            subject = self.entry_subject.get()
            body = self.entry_body.get("1.0", tk.END).strip()

            if not sender_email or not password or not receiver_email or not subject or not body:
                messagebox.showerror("Error", "Please fill in all fields.")
                return

            message = MIMEText(body)
            message["subject"] = subject
            message["from"] = sender_email
            message["to"] = receiver_email

            try:
                smtp_server = "smtp.gmail.com"
                smtp_port = 587

                server = smtplib.SMTP(smtp_server, smtp_port)
                server.starttls()
                server.login(sender_email, password)
                server.sendmail(sender_email, [receiver_email], message.as_string())
                server.quit()

                messagebox.showinfo("Success", "Email sent successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to send email: {str(e)}")

        def run(self):
            self.root.mainloop()

    if __name__ == "__main__":
        root = tk.Tk()
        app = EmailSender(root)
        app.run()

def use_sns_service():
    sns = boto3.client('sns',region_name='ap-south-1')
    sns.publish(
    Message='Alert! Alert! Alert!',
    Subject='Jaldi Bhaag, Jaldi Bhaag',
    TopicArn='arn:aws:sns:us-east-1:213901744873:Alert'
    )
    print("email sent")
    

def create_button(parent, label, command):
    button = tk.Button(parent,font=("Montserrat",10,"bold"), text=label,width=20,height=2, command=command)
    return button


root = tk.Tk()
root.title("Our Technical Project")
root.geometry("1200x900")
root.configure(bg="#856ff8")
frame = tk.Frame(root)
frame.pack()

button = tk.Button(frame, text="Load webpage", command=load_webpage)
button.pack() 

software_entry = tk.Label(root,font=("Montserrat",14,"bold"), text="MENU",width=20,height=2)
software_entry.pack(pady=20)

buttons_frame = tk.Frame(root, bg="#FDFD96")
buttons_frame.pack(padx=20, pady=20, fill="both", expand=True)

button_notepad = create_button(buttons_frame, "Youtube Playlist Downloader",videoDownload)
button_calculator = create_button(buttons_frame, "Launch EC2 with Gestures", ec2_finger)
button_paint = create_button(buttons_frame, "Voice Assistant", assistant)
button_chrome = create_button(buttons_frame, "Drawing App", create_drawing_app)
button_colorgame = create_button(buttons_frame, "Color Game", colorgame)
button_vlc = create_button(buttons_frame, "REKOGNITON", rekognition)
button_whatsapp = create_button(buttons_frame, "SEND WHATSAPP", whatsapp)
button_message = create_button(buttons_frame, "SEND MESSAGE", message)
button_photo = create_button(buttons_frame, "CLICK PHOTO",click_photo)
button_croppic = create_button(buttons_frame, "CROP PHOTO",crop_pic)
button_video = create_button(buttons_frame, "CAPTURE VIDEO",capture_video)
button_cropvideo = create_button(buttons_frame,"CROP VIDEO",capture_crop_video)
button_sendemail= create_button(buttons_frame,"Send Email",sendemail)
button_coordinates = create_button(buttons_frame,"GEO COORDINATES" ,lambda:get_coordinates())
button_searchresults = create_button(buttons_frame,"GOOGLE SEARCH",lambda:top_10_google_searches())
button_launchinstance = create_button(buttons_frame,"LAUNCH INSTANCE",launch_instance)
button_createbucket = create_button(buttons_frame,"CREATE BUCKET",create_bucket)
button_usesnsservice = create_button(buttons_frame,"USE SNS SERVICE",use_sns_service)
button_TikTacToe = create_button(buttons_frame, "TIC TAC TOE", TikTacToe)
button_linear = create_button(buttons_frame, "LINEAR REGRESSION", linearReg)
button_load = create_button(buttons_frame, "DOCUMENT LOADER", document_loader)
button_pomodoro = create_button(buttons_frame, "POMODORO", pomodoro)
button_cartoon = create_button(buttons_frame, "CARTOON VIDEO", cartoon)

button_notepad.grid(row=0, column=0, padx=20, pady=40)
button_calculator.grid(row=0, column=1, padx=20, pady=20)
button_paint.grid(row=0, column=2, padx=30, pady=20)
button_chrome.grid(row=0, column=3, padx=10, pady=20)
button_colorgame.grid(row=1, column=0, padx=20, pady=20)
button_vlc.grid(row=1, column=1, padx=30, pady=20)
button_whatsapp.grid(row=1, column=2, padx=10, pady=20)
button_message.grid(row=1, column=3, padx=10, pady=20)
button_photo.grid(row=2, column=0, padx=20, pady=20)
button_croppic.grid(row=2, column=1, padx=30, pady=20)
button_video.grid(row=2, column=2, padx=20, pady=20)
button_cropvideo.grid(row=2, column=3, padx=30, pady=20)
button_sendemail.grid(row=3, column=0, padx=40, pady=20)
button_coordinates.grid(row=3, column=1, padx=50, pady=20)
button_searchresults.grid(row=3, column=2,padx=40, pady=20) 
button_launchinstance.grid(row=3, column=3, padx=50, pady=20)
button_createbucket.grid(row=4, column=0, padx=40, pady=20)
button_usesnsservice.grid(row=4, column=1, padx=40, pady=20)
button_TikTacToe.grid(row=4, column=2, padx=40, pady=20)
button_linear.grid(row=4, column=3, padx=40, pady=20)
button_load.grid(row=5, column=0, padx=40, pady=20)
button_pomodoro.grid(row=5, column=1, padx=40, pady=20)
button_cartoon.grid(row=5, column=2, padx=40, pady=20)



root.mainloop()
