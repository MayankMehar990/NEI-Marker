from tkinter import *
from HMM_pos_tagging import hmm_model
from NEI_Marker import crf
from NEI_Marker import extract_features


root = Tk()
root.title("NEI Marker")
root.geometry("560x400")


label1 = Label(root, text="Enter a sentence : ",width=15,font=("Times New Roman",20))
label1.grid(row=0, column=0)

entry1 = Text(root,width=30,height=5)
entry1.grid(row=0, column=1,pady=15)

label2 = Label(root, text="POS Tags : ",width=15,font=("Times New Roman",20))
label2.grid(row=1, column=0)

output_box = Text(root,width=30,height=8,state='disabled')
output_box.grid(row=1,column=1)

def NEI_Marking(sentence):
    words = sentence.split()
    tagged = hmm_model.viterbi(words)  # returns list of (word, POS) tuples

    # Add dummy label 'O' to each tuple to create (word, POS, label)
    sentence_with_tags = [(word, pos, 'O') for word, pos in tagged]

    # Extract features for CRF model
    features = [extract_features(sentence_with_tags, i) for i in range(len(sentence_with_tags))]
    predicted_labels = crf.predict([features])[0]

    # Combine words with predicted NEI tags
    return " ".join(f"{word}/{tag}" for word, tag in zip(words, predicted_labels))


def clicked():
    label1.configure(text="Entered Sentence : ")
    sentence=entry1.get("1.0",END).strip()
    marked_sentence=NEI_Marking(sentence)
    output_box.config(state='normal')  
    output_box.delete("1.0", END) 
    output_box.insert("1.0",marked_sentence)     
    output_box.config(state='disabled')
    

button1 = Button(root, text="NEI_Marker",command=clicked,width=10,font=("Times New Roman",15))
button1.grid(row=0,column=1,pady=40)
button1.place(x=250,y=290)


close=Button(root, text="Close", command=root.destroy,width=10,font=("Times New Roman",15))
close.grid(row=0,column=2,pady=40)
close.place(x=380,y=289)

root.mainloop()
