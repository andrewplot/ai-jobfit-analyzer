import subprocess
import PyPDF2
import re

#Text Extraction and Processing
def extract_pdf_text(doc):
    reader = PyPDF2.PdfFileReader(doc)
    text = ""
    for page in reader.pages:
        curr_word = page.extract_text()
        if curr_word:
            text += curr_word + " "
    return text
    
def preprocess_text(text):
    text = text.replace("\n"," ").replace("\r"," ") #replace \n and \r with spaces
    text = re.sub(r'[^A-Za-z0-9\+\#\.\- ]+', ' ', text)  # replace all characters outside of set with spaces
    text = re.sub(r'\s+', ' ', text).strip() #remove all extra whitespaces
    return text

#Keyword Extraction





def main():
    resume_file_path = "/Users/aplotnik/Downloads/Resume-5.pdf"
    subprocess.run(["open", resume_file_path])
main() 
