"""Copyright 2026 Jacob Parker and Jordi Bella

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""
from dotenv import load_dotenv
from pprint import pprint
import requests, os, re, time, shutil

load_dotenv()

def get_cols(fasta_file, col_txt, min_col_length=10):
    """
    Opens a fasta file, filters for collagen containing proteins and adds them to a .txt file. Deals with ambiguous chars within sequence by replacing them 
    with an X and deals with gap (-) and terminal (*) chars by removing them from the sequence.

    Args: 
    fasta_file = Inputted fasta file
    col_txt = text file outputted containing COL domain proteins
    min_col_length = minimum length of COL domains permitted, measured in GXY triplets (ie. 10 triplets = 30 amino acids)
    """

    #Using regex to find collagen containing sequences and creates a txt file of all the COL domain containing proteins
    header = None
    new_line = [] #Using list and combining for time + memory efficiency
    sequence = ""
    pattern = re.compile(rf'(?:G[^G][^G]){{{min_col_length},}}') #updated regex which excludes GGG repeats and GXG or GGY segments (see if Jodri thinks its a good idea)
    clean1 = re.compile(r'\n|\r|\t')
    clean2 = re.compile(r'\-|\*')
    replaceX = re.compile(r'\s|\~|\.') #compiling regexes for time efficiency

    with open(fasta_file, 'r', encoding='utf-8') as f:
        with open(col_txt, 'w', encoding='utf-8') as output:
            for line in f:
                line = clean1.sub("", line) #remove any special chars within or end of lines
                if not line: #skip empty lines
                    continue
                if line.startswith(">"):
                    if header is not None:
                        sequence = "".join(new_line)
                        if len(new_line)<1: #checking if the header and sequence have been separated
                            header = clean2.sub("", header.upper()) #removing gap and terminal chars from header if header contains sequence
                            header = replaceX.sub("X", header) #replacing ambiguous chars with "X"
                            if pattern.search(header):
                                output.write(f"This header and sequence were not separated\n{header}\n\n") #Saves the header/sequence the pattern is found
                        
                        if len(new_line)>=1:
                            if pattern.search(sequence): #searches the combined lines for the pattern
                                output.write(f"{header}\n{sequence}\n\n") #Saves the header and combined sequence if the pattern is found
                        header = line #sets the new header
                        new_line = [] #resets the new line object for next sequence
                    else:
                        header = line #sets 1st header
                else:
                    line = clean2.sub("", line.upper()) #removing gap and terminal chars
                    line = replaceX.sub("X", line) #replacing ambiguous amino acid chars with "X"
                    new_line.append(line) #capturing sequence lines

            if header is not None: #capturing the last input
                sequence = "".join(new_line)
                if len(new_line)<1:
                    header = clean2.sub("", header.upper())
                    if pattern.search(header):
                        output.write(f"This header and sequence were not separated\n{header}\n\n") #Saves the header and combined sequence if the pattern is found and header and sequence wasn't separated
                if len(new_line)>=1:
                    if pattern.search(sequence): #searches the combined lines for the pattern
                        output.write(f"{header}\n{sequence}\n\n")


def label_cols(col_file, hfile, table, num=12):
    """ 
    Creates an html file with marked COL domains and interruptions within the given .txt file input.
    Additionally creates a tsv file which contains statistics about GXY and interruption prevelance.

    Args: 
    col_file = Inputted .txt file containing proteins with COL domains
    hfile = html file outputted with highlighted COL domains + interruptons
    table = tsv file output containing statistics about GXY and interruption prevelance
    num = the maximum number of amino acids permitted to separate GXY repeats while still counting as a COL domain interruption
    """
    
    #Using regex to identify collagen regions, marking them and then finding interruptions and marking those
    with open(col_file) as fin:
        with open(hfile, 'w', encoding='utf-8') as fout:
            text = fin.read()
            
            #Marking uninterrupted Col seqs
            text = re.sub(r'(G[A-Z][A-Z]){2,}G[A-Z][A-Z]', r'</span><span class="COL">\g<0></span><span class="NC">', text)
            #Marking G0G interruptions
            text = re.sub(r'G(G.)</span><span class="NC">(.)</span><span class="COL">', r'</span><span class="GXG">G</span><span class="COL">\1\2', text)
            #Marking G1G interruptions
            text = re.sub(r'(G.)G</span><span class="NC">(([A-Z]){2}</span><span class="COL">)',r'</span><span class="GXG">\1</span><span class="COL">G\2', text)
            #Marking G3-_G interruptions
            text = re.sub(rf'(G..)</span><span class="NC">(([A-Z]){{1,{num}}}</span><span class="COL">)', r'\1</span><span class="GNG">\2', text)
            #Marking GGG interruptions
            text = re.sub(r'GGG(G*)', r'<span class="GGG">\g<0></span>', text)
            #Marking places where headers and sequences weren't separates
            text = re.sub(r'(This header and sequence were not separated)', r'</span><span class="Warning">\g<0></span><span class="NC">', text)

            #Counting interruptions within COL domains
            interruption_counter = {"GGG": 0,f"G3-{num}G": 0,"G1G": 0, "G0G":0}
            searcher = re.findall(r'<span class="GGG">|<span class="GNG">|<span class="GXG">|<span class="GXG">G<', text)
            for t in searcher:
                if t == '<span class="GGG">':
                    interruption_counter["GGG"] +=1
                elif t == '<span class="GNG">':
                    interruption_counter[f"G3-{num}G"] +=1
                elif t == '<span class="GXG">':
                    interruption_counter["G1G"] +=1
                elif t == '<span class="GXG">G<':
                    interruption_counter["G0G"] +=1
            with open(table, 'w') as t:
                t.write(f"Interruptions within COL domains:\n")
                for k,v in interruption_counter.items():
                    t.write(f"{k}:\t{v},\t")
            
            #Counting GXX' triplets within COL domains
            cols = ""
            filtered_text1 = re.findall(r'<span class="COL">((?:G..)+)</span>', text) #2nd parentheses around the GXY+ unit ensures only that unit is captured while matching the whole thing
            cols = "".join(filtered_text1)
            triplet_counter = {}
            total_trips = 0
            trips = [cols[i:i+3] for i in range(0, len(cols), 3)]
            for trip in trips:    
                if trip not in triplet_counter:
                    triplet_counter[trip] = 1
                else: triplet_counter[trip] += 1
                total_trips +=1
            #Sorting the dictionary by values in decending order so most prevelant GXYs are at the top and adding %s after
            triplet_counter = dict(sorted(triplet_counter.items(), key=lambda item: item[1], reverse=True))
            for k,v in triplet_counter.items():
                triplet_counter[k] =(v,(v/total_trips*100))
            #Creating a TSV containing the data from the dictionary
            with open(table, 'a', encoding='utf-8') as t:
                t.write(f"\n\nGXX' triplets in COL domains:\nGXX':\tTotal\t% of all triplets\n")
                for k,(v,p) in triplet_counter.items():
                    t.write(f"{k}:\t{v}\t{round(p, 3)}%\n")
            
            #Creating the html file
            #Header for html file
            head = """<!DOCTYPE html>
            <html>
            <head>
            <title>Wiki Collagens html output file</title>
            <style type="text/css">
                p.p1 {margin: 0.0px 0.0px 0.0px 0.0px; word-wrap: break-word; white-space: pre-wrap; font: 11.0px Menlo; color: #000000; background-color: #ffffff}
                span.NC {font: 12.0px Menlo; word-wrap: break-word; white-space: pre-wrap; color: #000000}
                span.COL {font: 12.0px Menlo; word-wrap: break-word; white-space: pre-wrap; color: #000000; background-color: #ffff00}
                span.GXG {font: 12.0px Menlo; word-wrap: break-word; white-space: pre-wrap; color: #000000; background-color: #00ff00}
                span.GNG {font: 12.0px Menlo; word-wrap: break-word; white-space: pre-wrap; color: #000000; background-color: #00ffff}
                span.GGG {font: 12.0px Menlo; word-wrap: break-word; white-space: pre-wrap; color: #ff0000}
                span.Warning {font: 12.0px Menlo; word-wrap: break-word; white-space: pre-wrap; color: #ff0000}
            </style>
            </head>
            <body>
            <span class="NC">"""
            
            #footer for html file
            foot = """
                </span>
                </body>
                </html>"""
            
            html = head
            html += text
            html += foot
            
            fout.write(html)

def cleanup_files(folder="sessions", age_limit=3600):
    """
    Function to remove files that are older than 1hr
    
    Args:
    folder = the directory within which you want to look for files older than the age limit
    age_limit = the maximum age of a file/folder before it is deleted, measured in seconds
    """
    now = time.time()
    for session in os.listdir(folder):
        if session == ".gitkeep":
            continue
        
        path = os.path.join(folder, session)
        age = now - os.path.getmtime(path)
        if age > age_limit:
            shutil.rmtree(path)
