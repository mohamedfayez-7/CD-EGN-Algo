from operator import mul
import networkx as nx
import os
import sys
import dendrogram_from_girvan_newman as dgn
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as mpcolors
import matplotlib.cm as mpcm
import gspread
from oauth2client.service_account import ServiceAccountCredentials

class Communities:
    def __init__(self, ipt_txt, dend_png):
        self.ipt_txt = ipt_txt
        self.dend_png = dend_png
        self.nodes = []
        self.nodes_num = []
        self.graph = None
        self.graph_num = None

    def initialize(self):
        if not os.path.isfile(self.ipt_txt):
            self.quit(self.ipt_txt + " doesn't exist or it's not a file")

        # initialize 3rd-party libraries
        self.graph = nx.Graph()
        self.graph_num = nx.Graph()
        # load data
        self.load_txt(self.ipt_txt)
        #self.mySheet()

        
    
    def plot_dend(self):
        fig, ax = plt.subplots(figsize=(20, 10))        #draw the dendrogram (output)
        fig.canvas.draw()
        partitions = dgn.girvan_newman_partitions(self.graph_num)
        agglomerative_mat = dgn.agglomerative_matrix(self.graph_num, partitions)
        #print (agglomerative_mat)
        hierarchy.dendrogram( agglomerative_mat)
        labels = [item.get_text() for item in ax.get_xticklabels()]
        for n in range(len(labels)):
            x = labels[n]
            labels[n] = self.nodes[int(x)]
        ax.set_xticklabels(labels)
        plt.xticks(rotation=0,fontsize=6)
        plt.savefig(self.dend_png)
        plt.show()

   
    def mySheet(self):
        scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/spreadsheets',"https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name("creds.json", scope)
        client = gspread.authorize(creds)
        sheet = client.open("Community Detection (Responses)").sheet1   
        num = 0
        edges_num = []
        edges = []
        for i in range(1,sheet.row_count):
            row = sheet.row_values(i+1)  # Get a specific row
            if (len(row)>2):
                for name in str(row[4]).split(','):
                    self.graph.add_edge(str(row[1]+row[2]), str(name), weight=0.6, len=3.0)
                    if str(name) not in self.nodes:
                        self.nodes.append(str(name))
                        self.nodes_num.append(num)
                        num+=1
                    if str(row[1]+row[2]) not in self.nodes:
                        self.nodes.append(str(row[1]+row[2]))
                        self.nodes_num.append(num)
                        num+=1
                    temp = (self.nodes_num[self.nodes.index(str(name))],self.nodes_num[self.nodes.index(str(row[1]+row[2]))])
                    edges.append((str(name),str(row[1]+row[2])))
                    edges_num.append(temp)
        
        self.graph_num.add_nodes_from(self.nodes_num)
        self.graph_num.add_edges_from(edges_num)
        self.graph.add_nodes_from(self.nodes)
        self.graph.add_edges_from(edges)
        

    def load_txt(self, ipt_txt):
        input_data = open(ipt_txt, 'r')
        num = 0
        edges_num = []
        edges = []
        for line in input_data:
            line = line.strip('\n')
            line = str(line)
            line = line.split(" ")

            if len(line) != 2:
                #self.quit("edge format for input.txt is error")
                continue
            if line[0] not in self.nodes:   # two lists to convert the string input to integer  
                self.nodes.append(line[0])  # because any matrix cannot take two datatypes and then get it back to the main form (string) 
                self.nodes_num.append(num)
                num+=1
            if line[1] not in self.nodes:
                self.nodes.append(line[1])
                self.nodes_num.append(num)
                num+=1
            temp = (self.nodes_num[self.nodes.index(line[0])],self.nodes_num[self.nodes.index(line[1])])
            edges.append((line[0],line[1]))
            edges_num.append(temp)
        self.graph_num.add_nodes_from(self.nodes_num)
        self.graph_num.add_edges_from(edges_num)
        self.graph.add_nodes_from(self.nodes)
        self.graph.add_edges_from(edges)

    def display(self):
        font = {'family' : 'DejaVu Sans',
                'size'   : 12}
        plt.rc('font', **font)
        self.plot_dend()
    def quit(self, err_desc):
        raise SystemExit('\n'+ "PROGRAM EXIT: " + err_desc + ', please check your input' + '\n')


